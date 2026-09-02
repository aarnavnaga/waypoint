// Package ratelimit implements per-key rate limiting shared across gateway
// replicas, backed by an atomic Redis Lua script.
package ratelimit

import (
	"context"
	"errors"
	"fmt"
	"math"
	"time"

	"github.com/redis/go-redis/v9"
)

// Decision is the result of a limit check.
type Decision struct {
	// Allowed reports whether the request may proceed.
	Allowed bool
	// Limit is the bucket capacity that was evaluated.
	Limit int64
	// Remaining is the budget left after this request.
	Remaining int64
	// RetryAfter is how long until enough budget exists, when denied.
	RetryAfter time.Duration
	// Resource names which bucket denied the request ("requests" or "tokens").
	Resource string
	// Degraded reports that Redis was unreachable and the configured fail-open
	// or fail-closed policy was applied instead of a real decision.
	Degraded bool
}

// Limiter checks and consumes rate-limit budget.
type Limiter interface {
	// Allow consumes budget for one request estimated at estTokens.
	Allow(ctx context.Context, key string, estTokens int64) (Decision, error)
}

// tokenBucketScript implements two token buckets — requests and tokens — in one
// atomic operation.
//
// It must be a script, not a GET/compute/SET sequence, because two replicas
// checking the same key concurrently would both read the old value and both
// decide they are under the limit. Redis executes a script atomically on a
// single thread, so check-and-decrement cannot interleave.
//
// Both buckets are checked *before* either is debited: a request that fails the
// token limit must not silently consume request budget, or a client hammering an
// over-quota key would drain a bucket it was never allowed to use.
//
// KEYS[1] = requests bucket key
// KEYS[2] = tokens bucket key
// ARGV[1] = requests per minute
// ARGV[2] = tokens per minute
// ARGV[3] = estimated tokens for this request
// ARGV[4] = current time (milliseconds)
// ARGV[5] = key TTL (seconds)
const tokenBucketScript = `
local rpm      = tonumber(ARGV[1])
local tpm      = tonumber(ARGV[2])
local est      = tonumber(ARGV[3])
local now_ms   = tonumber(ARGV[4])
local ttl      = tonumber(ARGV[5])

-- Refill rates, expressed per millisecond.
local req_rate = rpm / 60000.0
local tok_rate = tpm / 60000.0

-- Load a bucket, refilling it for elapsed time since it was last touched.
local function load_bucket(key, capacity, rate)
  local data = redis.call('HMGET', key, 'tokens', 'ts')
  local tokens = tonumber(data[1])
  local ts     = tonumber(data[2])

  if tokens == nil then
    -- A new key starts full, so a first-time caller is not penalized.
    return capacity, now_ms
  end

  local elapsed = math.max(0, now_ms - ts)
  tokens = math.min(capacity, tokens + elapsed * rate)
  return tokens, now_ms
end

local req_tokens, _ = load_bucket(KEYS[1], rpm, req_rate)
local tok_tokens, _ = load_bucket(KEYS[2], tpm, tok_rate)

-- Evaluate both limits before debiting either.
local req_ok = req_tokens >= 1
local tok_ok = tok_tokens >= est

if not req_ok or not tok_ok then
  local resource, deficit, rate
  if not req_ok then
    resource, deficit, rate = 'requests', 1 - req_tokens, req_rate
  else
    resource, deficit, rate = 'tokens', est - tok_tokens, tok_rate
  end

  local retry_ms = 0
  if rate > 0 then
    retry_ms = math.ceil(deficit / rate)
  end

  -- Persist the refilled state so elapsed time is not recomputed from stale
  -- timestamps on the next call.
  redis.call('HSET', KEYS[1], 'tokens', req_tokens, 'ts', now_ms)
  redis.call('HSET', KEYS[2], 'tokens', tok_tokens, 'ts', now_ms)
  redis.call('EXPIRE', KEYS[1], ttl)
  redis.call('EXPIRE', KEYS[2], ttl)

  return {0, resource, tostring(math.floor(req_tokens)), tostring(math.floor(tok_tokens)), tostring(retry_ms)}
end

req_tokens = req_tokens - 1
tok_tokens = tok_tokens - est

redis.call('HSET', KEYS[1], 'tokens', req_tokens, 'ts', now_ms)
redis.call('HSET', KEYS[2], 'tokens', tok_tokens, 'ts', now_ms)
redis.call('EXPIRE', KEYS[1], ttl)
redis.call('EXPIRE', KEYS[2], ttl)

return {1, 'ok', tostring(math.floor(req_tokens)), tostring(math.floor(tok_tokens)), '0'}
`

// Options configures a RedisLimiter.
type Options struct {
	// RPM is the requests-per-minute capacity.
	RPM int64
	// TPM is the tokens-per-minute capacity.
	TPM int64
	// FailOpen decides behavior when Redis is unreachable.
	FailOpen bool
	// KeyPrefix namespaces keys in a shared Redis.
	KeyPrefix string
	// Timeout bounds each Redis call.
	Timeout time.Duration
	// Now is injectable for tests.
	Now func() time.Time
}

// RedisLimiter is a Redis-backed token bucket limiter.
type RedisLimiter struct {
	client redis.Scripter
	script *redis.Script
	opts   Options
}

// NewRedisLimiter builds a limiter against the given Redis client.
func NewRedisLimiter(client redis.Scripter, opts Options) *RedisLimiter {
	if opts.KeyPrefix == "" {
		opts.KeyPrefix = "waypoint"
	}
	if opts.Timeout <= 0 {
		opts.Timeout = 200 * time.Millisecond
	}
	if opts.Now == nil {
		opts.Now = time.Now
	}
	return &RedisLimiter{
		client: client,
		script: redis.NewScript(tokenBucketScript),
		opts:   opts,
	}
}

// bucketTTL is how long an idle key survives. It must exceed the time to refill
// a full bucket, or a returning client could be handed a fresh full bucket
// sooner than its limit allows.
const bucketTTL = 300 // seconds

// Allow consumes budget for one request.
func (l *RedisLimiter) Allow(ctx context.Context, key string, estTokens int64) (Decision, error) {
	if estTokens < 0 {
		estTokens = 0
	}

	callCtx, cancel := context.WithTimeout(ctx, l.opts.Timeout)
	defer cancel()

	keys := []string{
		fmt.Sprintf("%s:rl:req:%s", l.opts.KeyPrefix, key),
		fmt.Sprintf("%s:rl:tok:%s", l.opts.KeyPrefix, key),
	}
	args := []any{
		l.opts.RPM,
		l.opts.TPM,
		estTokens,
		l.opts.Now().UnixMilli(),
		bucketTTL,
	}

	raw, err := l.script.Run(callCtx, l.client, keys, args...).Result()
	if err != nil {
		return l.degraded(), fmt.Errorf("ratelimit: redis: %w", err)
	}

	return parseDecision(raw, l.opts.RPM, l.opts.TPM)
}

// degraded applies the configured policy when Redis cannot be reached.
//
// Fail-open is the default, and it is a genuine tradeoff rather than laziness.
// Failing closed means a Redis outage takes down the entire API for every
// tenant — converting a degraded *limiting* capability into a total *serving*
// outage. Rate limits exist to protect capacity, and an unprotected but working
// service is almost always better than a protected but dead one. Fail-closed is
// available for deployments where quota enforcement is a billing or
// safety requirement rather than a capacity guard.
func (l *RedisLimiter) degraded() Decision {
	return Decision{
		Allowed:  l.opts.FailOpen,
		Limit:    l.opts.RPM,
		Degraded: true,
		Resource: "redis_unavailable",
	}
}

// parseDecision converts the Lua reply into a Decision.
func parseDecision(raw any, rpm, tpm int64) (Decision, error) {
	values, ok := raw.([]any)
	if !ok || len(values) < 5 {
		return Decision{}, errors.New("ratelimit: unexpected script reply shape")
	}

	allowed, _ := values[0].(int64)
	resource, _ := values[1].(string)
	reqRemaining := parseInt(values[2])
	tokRemaining := parseInt(values[3])
	retryMS := parseInt(values[4])

	d := Decision{
		Allowed:    allowed == 1,
		Resource:   resource,
		RetryAfter: time.Duration(retryMS) * time.Millisecond,
	}

	// Report the limit and remaining budget for whichever resource is scarcer,
	// so clients see the constraint that will actually bite them.
	reqFrac := fraction(reqRemaining, rpm)
	tokFrac := fraction(tokRemaining, tpm)
	if tokFrac < reqFrac {
		d.Limit, d.Remaining = tpm, tokRemaining
	} else {
		d.Limit, d.Remaining = rpm, reqRemaining
	}
	if d.Remaining < 0 {
		d.Remaining = 0
	}
	return d, nil
}

func fraction(remaining, capacity int64) float64 {
	if capacity <= 0 {
		return math.Inf(1)
	}
	return float64(remaining) / float64(capacity)
}

func parseInt(v any) int64 {
	switch value := v.(type) {
	case int64:
		return value
	case string:
		var n int64
		_, err := fmt.Sscanf(value, "%d", &n)
		if err != nil {
			return 0
		}
		return n
	default:
		return 0
	}
}

// AllowAll is a Limiter that permits everything, used when rate limiting is
// disabled so the request path has no nil checks.
type AllowAll struct{}

// Allow implements Limiter.
func (AllowAll) Allow(context.Context, string, int64) (Decision, error) {
	return Decision{Allowed: true, Remaining: math.MaxInt32}, nil
}
