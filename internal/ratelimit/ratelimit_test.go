package ratelimit

import (
	"context"
	"errors"
	"fmt"
	"os"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/redis/go-redis/v9"
)

// redisAddr is where tests expect Redis Stack. Start it with:
//
//	docker run -d --name waypoint-redis -p 6379:6379 redis/redis-stack-server:7.4.0-v1
func redisAddr() string {
	if v := os.Getenv("WAYPOINT_REDIS_ADDR"); v != "" {
		return v
	}
	return "localhost:6379"
}

// newTestClient returns a Redis client, skipping the test if none is reachable.
func newTestClient(t *testing.T) *redis.Client {
	t.Helper()

	client := redis.NewClient(&redis.Options{
		Addr:        redisAddr(),
		DialTimeout: time.Second,
		ReadTimeout: time.Second,
	})

	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()

	if err := client.Ping(ctx).Err(); err != nil {
		_ = client.Close()
		t.Skipf("Redis not reachable at %s (%v); start it with:\n"+
			"  docker run -d --name waypoint-redis -p 6379:6379 redis/redis-stack-server:7.4.0-v1",
			redisAddr(), err)
	}

	t.Cleanup(func() { _ = client.Close() })
	return client
}

// uniqueKey keeps parallel test runs from sharing buckets.
func uniqueKey(t *testing.T) string {
	t.Helper()
	return fmt.Sprintf("test-%s-%d", t.Name(), time.Now().UnixNano())
}

func TestAllowsWithinLimit(t *testing.T) {
	client := newTestClient(t)
	l := NewRedisLimiter(client, Options{RPM: 60, TPM: 100000, FailOpen: true})

	key := uniqueKey(t)
	ctx := context.Background()

	for i := 0; i < 10; i++ {
		d, err := l.Allow(ctx, key, 100)
		if err != nil {
			t.Fatalf("Allow %d: %v", i, err)
		}
		if !d.Allowed {
			t.Fatalf("request %d denied while under the limit: %+v", i, d)
		}
	}
}

func TestDeniesOverRequestLimit(t *testing.T) {
	client := newTestClient(t)
	// A tiny bucket so it is exhausted quickly and refills slowly.
	l := NewRedisLimiter(client, Options{RPM: 5, TPM: 1000000, FailOpen: true})

	key := uniqueKey(t)
	ctx := context.Background()

	allowed := 0
	for i := 0; i < 20; i++ {
		d, err := l.Allow(ctx, key, 1)
		if err != nil {
			t.Fatalf("Allow: %v", err)
		}
		if d.Allowed {
			allowed++
		} else {
			if d.Resource != "requests" {
				t.Errorf("denied on %q, want requests", d.Resource)
			}
			if d.RetryAfter <= 0 {
				t.Error("denied response carries no RetryAfter")
			}
		}
	}

	// The bucket starts full at 5 and refills slowly, so only ~5 should pass.
	if allowed > 7 {
		t.Errorf("%d requests allowed against an RPM of 5", allowed)
	}
	if allowed < 5 {
		t.Errorf("%d requests allowed, want at least the bucket capacity of 5", allowed)
	}
}

// The token limit must bind independently: a few very large requests can exhaust
// TPM while barely touching RPM.
func TestDeniesOverTokenLimit(t *testing.T) {
	client := newTestClient(t)
	l := NewRedisLimiter(client, Options{RPM: 1000, TPM: 1000, FailOpen: true})

	key := uniqueKey(t)
	ctx := context.Background()

	// Two requests of 600 tokens each exceed a 1000-token budget.
	first, err := l.Allow(ctx, key, 600)
	if err != nil {
		t.Fatalf("Allow: %v", err)
	}
	if !first.Allowed {
		t.Fatal("first 600-token request denied against a 1000 budget")
	}

	second, err := l.Allow(ctx, key, 600)
	if err != nil {
		t.Fatalf("Allow: %v", err)
	}
	if second.Allowed {
		t.Error("second 600-token request allowed; token budget was not enforced")
	}
	if second.Resource != "tokens" {
		t.Errorf("denied on %q, want tokens", second.Resource)
	}
}

// A request rejected for tokens must not silently consume request budget, or a
// client hammering an over-quota key would drain a bucket it never used.
func TestDeniedRequestDoesNotConsumeOtherBucket(t *testing.T) {
	client := newTestClient(t)
	l := NewRedisLimiter(client, Options{RPM: 10, TPM: 100, FailOpen: true})

	key := uniqueKey(t)
	ctx := context.Background()

	// Exhaust the token bucket with one oversized request.
	if _, err := l.Allow(ctx, key, 100); err != nil {
		t.Fatalf("Allow: %v", err)
	}

	// Several denials follow; none should debit the request bucket.
	for i := 0; i < 5; i++ {
		d, err := l.Allow(ctx, key, 100)
		if err != nil {
			t.Fatalf("Allow: %v", err)
		}
		if d.Allowed {
			t.Fatalf("request %d allowed over the token budget", i)
		}
	}

	// A small request should still fit in the untouched request budget.
	d, err := l.Allow(ctx, key, 0)
	if err != nil {
		t.Fatalf("Allow: %v", err)
	}
	if !d.Allowed {
		t.Errorf("zero-token request denied: rejected requests consumed request budget (%+v)", d)
	}
}

func TestKeysAreIsolated(t *testing.T) {
	client := newTestClient(t)
	l := NewRedisLimiter(client, Options{RPM: 3, TPM: 1000000, FailOpen: true})

	ctx := context.Background()
	keyA, keyB := uniqueKey(t)+"-a", uniqueKey(t)+"-b"

	for i := 0; i < 5; i++ {
		if _, err := l.Allow(ctx, keyA, 1); err != nil {
			t.Fatalf("Allow: %v", err)
		}
	}

	// keyB has its own bucket and must be unaffected.
	d, err := l.Allow(ctx, keyB, 1)
	if err != nil {
		t.Fatalf("Allow: %v", err)
	}
	if !d.Allowed {
		t.Error("exhausting one key's budget denied a different key")
	}
}

func TestBucketRefills(t *testing.T) {
	client := newTestClient(t)

	// 60 RPM = 1 token per second. A small bucket drains fast enough that
	// refill during the drain loop is negligible (~60ms of draining restores
	// 0.06 tokens), which keeps the assertion deterministic.
	l := NewRedisLimiter(client, Options{RPM: 60, TPM: 100000000, FailOpen: true})
	key := uniqueKey(t)
	ctx := context.Background()

	drained := false
	for i := 0; i < 80; i++ {
		d, err := l.Allow(ctx, key, 1)
		if err != nil {
			t.Fatalf("Allow: %v", err)
		}
		if !d.Allowed {
			drained = true
			if d.RetryAfter <= 0 || d.RetryAfter > 2*time.Second {
				t.Errorf("RetryAfter = %s, want ~1s at 1 token/sec", d.RetryAfter)
			}
			break
		}
	}
	if !drained {
		t.Fatal("bucket never drained")
	}

	// One second restores exactly one token.
	time.Sleep(1100 * time.Millisecond)

	d, err := l.Allow(ctx, key, 1)
	if err != nil {
		t.Fatalf("Allow: %v", err)
	}
	if !d.Allowed {
		t.Error("bucket did not refill after waiting one refill interval")
	}

	// ...and only one: the next request should be denied again.
	if next, _ := l.Allow(ctx, key, 1); next.Allowed {
		t.Error("refill granted more than the elapsed time earned")
	}
}

// A first-time caller must start with a full bucket, not an empty one.
func TestNewKeyStartsFull(t *testing.T) {
	client := newTestClient(t)
	l := NewRedisLimiter(client, Options{RPM: 100, TPM: 100000, FailOpen: true})

	d, err := l.Allow(context.Background(), uniqueKey(t), 10)
	if err != nil {
		t.Fatalf("Allow: %v", err)
	}
	if !d.Allowed {
		t.Error("a brand new key was immediately rate limited")
	}
	if d.Remaining <= 0 {
		t.Errorf("Remaining = %d on a new key, want a nearly full bucket", d.Remaining)
	}
}

// The property the whole package exists for: the limit must hold across
// replicas. Two limiters sharing one Redis simulate two gateway processes.
func TestLimitHoldsAcrossReplicas(t *testing.T) {
	client := newTestClient(t)

	const rpm = 50
	replicaA := NewRedisLimiter(client, Options{RPM: rpm, TPM: 10000000, FailOpen: true})
	replicaB := NewRedisLimiter(client, Options{RPM: rpm, TPM: 10000000, FailOpen: true})

	key := uniqueKey(t)
	ctx := context.Background()

	var allowed atomic.Int64
	var wg sync.WaitGroup

	// Hammer both replicas concurrently with the same key.
	for _, l := range []*RedisLimiter{replicaA, replicaB} {
		for g := 0; g < 8; g++ {
			wg.Add(1)
			go func(l *RedisLimiter) {
				defer wg.Done()
				for i := 0; i < 20; i++ {
					d, err := l.Allow(ctx, key, 1)
					if err != nil {
						return
					}
					if d.Allowed {
						allowed.Add(1)
					}
				}
			}(l)
		}
	}
	wg.Wait()

	total := allowed.Load()
	t.Logf("2 replicas x 8 goroutines x 20 requests: %d allowed against an RPM of %d", total, rpm)

	// 320 requests were attempted against a bucket of 50. Refill during the run
	// adds a little headroom, but nothing close to double.
	if total > rpm+15 {
		t.Errorf("%d requests allowed against a shared limit of %d: the limit did not hold across replicas",
			total, rpm)
	}
	if total < rpm-5 {
		t.Errorf("%d requests allowed, want close to the full budget of %d", total, rpm)
	}
}

// Atomicity check: without a Lua script, concurrent check-then-set from many
// callers would over-admit badly.
func TestConcurrentCallersDoNotOverAdmit(t *testing.T) {
	client := newTestClient(t)

	const rpm = 20
	l := NewRedisLimiter(client, Options{RPM: rpm, TPM: 10000000, FailOpen: true})
	key := uniqueKey(t)

	var allowed atomic.Int64
	var wg sync.WaitGroup
	start := make(chan struct{})

	for i := 0; i < 100; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			<-start // release everyone at once to maximize contention
			d, err := l.Allow(context.Background(), key, 1)
			if err == nil && d.Allowed {
				allowed.Add(1)
			}
		}()
	}
	close(start)
	wg.Wait()

	if got := allowed.Load(); got > rpm+3 {
		t.Errorf("%d of 100 simultaneous requests allowed against RPM %d; check-and-decrement is not atomic",
			got, rpm)
	}
}

func TestFailOpenWhenRedisUnavailable(t *testing.T) {
	// Nothing is listening on this port.
	client := redis.NewClient(&redis.Options{
		Addr:        "127.0.0.1:1",
		DialTimeout: 100 * time.Millisecond,
	})
	defer func() { _ = client.Close() }()

	l := NewRedisLimiter(client, Options{
		RPM: 10, TPM: 1000, FailOpen: true, Timeout: 200 * time.Millisecond,
	})

	d, err := l.Allow(context.Background(), "any-key", 1)
	if err == nil {
		t.Error("Allow returned nil error against a dead Redis")
	}
	if !d.Allowed {
		t.Error("fail-open limiter denied a request when Redis was unavailable")
	}
	if !d.Degraded {
		t.Error("decision is not marked degraded")
	}
}

func TestFailClosedWhenRedisUnavailable(t *testing.T) {
	client := redis.NewClient(&redis.Options{
		Addr:        "127.0.0.1:1",
		DialTimeout: 100 * time.Millisecond,
	})
	defer func() { _ = client.Close() }()

	l := NewRedisLimiter(client, Options{
		RPM: 10, TPM: 1000, FailOpen: false, Timeout: 200 * time.Millisecond,
	})

	d, err := l.Allow(context.Background(), "any-key", 1)
	if err == nil {
		t.Error("Allow returned nil error against a dead Redis")
	}
	if d.Allowed {
		t.Error("fail-closed limiter allowed a request when Redis was unavailable")
	}
	if !d.Degraded {
		t.Error("decision is not marked degraded")
	}
}

// A dead Redis must be detected fast, or fail-open is useless: every request
// would still pay the full timeout.
func TestRedisFailureIsFast(t *testing.T) {
	client := redis.NewClient(&redis.Options{
		Addr:        "127.0.0.1:1",
		DialTimeout: 50 * time.Millisecond,
	})
	defer func() { _ = client.Close() }()

	l := NewRedisLimiter(client, Options{
		RPM: 10, TPM: 1000, FailOpen: true, Timeout: 150 * time.Millisecond,
	})

	start := time.Now()
	_, _ = l.Allow(context.Background(), "k", 1)
	elapsed := time.Since(start)

	if elapsed > time.Second {
		t.Errorf("failed Redis call took %s; fail-open would add that to every request", elapsed)
	}
}

func TestNegativeTokensClamped(t *testing.T) {
	client := newTestClient(t)
	l := NewRedisLimiter(client, Options{RPM: 100, TPM: 1000, FailOpen: true})

	d, err := l.Allow(context.Background(), uniqueKey(t), -500)
	if err != nil {
		t.Fatalf("Allow: %v", err)
	}
	if !d.Allowed {
		t.Error("a negative token estimate was rejected instead of clamped")
	}
}

func TestAllowAllLimiter(t *testing.T) {
	d, err := AllowAll{}.Allow(context.Background(), "k", 999999)
	if err != nil {
		t.Fatalf("AllowAll returned an error: %v", err)
	}
	if !d.Allowed {
		t.Error("AllowAll denied a request")
	}
}

func TestParseDecisionRejectsBadReply(t *testing.T) {
	if _, err := parseDecision("not a list", 10, 10); err == nil {
		t.Error("parseDecision accepted a malformed reply")
	}
	if _, err := parseDecision([]any{int64(1)}, 10, 10); err == nil {
		t.Error("parseDecision accepted a short reply")
	}
}

func TestLimiterInterface(t *testing.T) {
	var _ Limiter = (*RedisLimiter)(nil)
	var _ Limiter = AllowAll{}
}

func TestContextCancellationPropagates(t *testing.T) {
	client := newTestClient(t)
	l := NewRedisLimiter(client, Options{RPM: 100, TPM: 1000, FailOpen: true})

	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	_, err := l.Allow(ctx, uniqueKey(t), 1)
	if err == nil {
		t.Error("Allow ignored a cancelled context")
	}
	if !errors.Is(err, context.Canceled) && err != nil {
		// go-redis may wrap differently; the important part is that it errored.
		t.Logf("cancellation surfaced as: %v", err)
	}
}
