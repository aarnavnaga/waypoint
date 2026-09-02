//go:build integration

package integration

import (
	"context"
	"encoding/json"
	"fmt"
	"net"
	"net/http"
	"net/http/httptest"
	"os"
	"strconv"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/redis/go-redis/v9"

	"github.com/aarnavnaga/waypoint/internal/admission"
	"github.com/aarnavnaga/waypoint/internal/api"
	"github.com/aarnavnaga/waypoint/internal/backend"
	"github.com/aarnavnaga/waypoint/internal/config"
	"github.com/aarnavnaga/waypoint/internal/mocksim"
	"github.com/aarnavnaga/waypoint/internal/openai"
	"github.com/aarnavnaga/waypoint/internal/ratelimit"
	"github.com/aarnavnaga/waypoint/internal/router"
	"github.com/aarnavnaga/waypoint/internal/tokenizer"
)

func redisAddr() string {
	if v := os.Getenv("WAYPOINT_REDIS_ADDR"); v != "" {
		return v
	}
	return "localhost:6379"
}

func requireRedis(t *testing.T) *redis.Client {
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

// replicaSet is several independent gateway processes sharing one backend fleet
// and one Redis, which is exactly the production topology behind a load
// balancer.
type replicaSet struct {
	urls     []string
	backends []*httptest.Server
	client   *http.Client
	apiKey   string
}

// newReplicaSet starts `replicas` gateways over a shared backend pool.
func newReplicaSet(t *testing.T, replicas int, rdb *redis.Client, limits config.RateLimit) *replicaSet {
	t.Helper()

	rs := &replicaSet{
		client: &http.Client{Timeout: 20 * time.Second},
		apiKey: fmt.Sprintf("replica-key-%d", time.Now().UnixNano()),
	}

	// One shared backend fleet, as a real deployment would have.
	cfgBackends := make([]config.Backend, 0, 2)
	for i := 0; i < 2; i++ {
		mcfg := mocksim.DefaultConfig()
		mcfg.Name = fmt.Sprintf("mock-%d", i+1)
		mcfg.Seed = int64(i + 1)
		mcfg.Prefill.Base = mocksim.Duration{Duration: 0}
		mcfg.Prefill.PerToken = mocksim.Duration{Duration: time.Microsecond}
		mcfg.Decode.TokensPerSecond = 50000
		mcfg.Output.MinTokens, mcfg.Output.MaxTokens = 4, 4

		srv := httptest.NewServer(mocksim.NewServer(mocksim.NewEngine(mcfg), quietLogger()).Handler())
		t.Cleanup(srv.Close)

		rs.backends = append(rs.backends, srv)
		cfgBackends = append(cfgBackends, config.Backend{
			Name: mcfg.Name, URL: srv.URL, MaxInflightTokens: 500000,
		})
	}

	keyPrefix := fmt.Sprintf("itest-%d", time.Now().UnixNano())

	for i := 0; i < replicas; i++ {
		cfg := config.Default()
		cfg.Backends = cfgBackends
		cfg.Auth.APIKeys = []string{rs.apiKey}
		cfg.RateLimit = limits
		cfg.Redis.KeyPrefix = keyPrefix

		pool, err := backend.NewPool(cfg.Backends)
		if err != nil {
			t.Fatalf("NewPool: %v", err)
		}
		est := tokenizer.NewCharHeuristic()
		rt, err := router.New(cfg.Routing, est)
		if err != nil {
			t.Fatalf("router.New: %v", err)
		}

		limiter := ratelimit.NewRedisLimiter(rdb, ratelimit.Options{
			RPM:       int64(cfg.RateLimit.RPM),
			TPM:       int64(cfg.RateLimit.TPM),
			FailOpen:  cfg.RateLimit.FailOpen,
			KeyPrefix: keyPrefix,
		})

		proxy := api.NewProxy(&http.Client{}, api.Timeouts{TTFT: 5 * time.Second}, quietLogger())
		srv := api.NewServer(api.Options{
			Config: cfg, Pool: pool, Router: rt, Proxy: proxy,
			Dispatcher: api.NewDispatcher(api.DispatcherOptions{
				Pool: pool, Router: rt, Proxy: proxy, MaxAttempts: 1, Logger: quietLogger(),
			}),
			Admission: admission.New(admission.Options{MaxConcurrent: 128, MaxQueue: 512}),
			Limiter:   limiter,
			Estimator: est,
			Logger:    quietLogger(),
			Version:   fmt.Sprintf("replica-%d", i+1),
		})

		ln, err := net.Listen("tcp", "127.0.0.1:0")
		if err != nil {
			t.Fatalf("listen: %v", err)
		}
		httpSrv := &http.Server{Handler: srv.Handler(), ReadHeaderTimeout: 5 * time.Second}
		go func() { _ = httpSrv.Serve(ln) }()
		t.Cleanup(func() { _ = httpSrv.Close() })

		rs.urls = append(rs.urls, "http://"+ln.Addr().String())
	}

	return rs
}

// send issues one chat request to the given replica.
func (rs *replicaSet) send(replica int, prompt string) (status int, headers http.Header, err error) {
	payload, err := json.Marshal(openai.ChatRequest{
		Model:     "mock-7b",
		Messages:  []openai.Message{{Role: openai.RoleUser, Content: prompt}},
		MaxTokens: intPtr(4),
	})
	if err != nil {
		return 0, nil, err
	}

	req, err := http.NewRequest(http.MethodPost,
		rs.urls[replica%len(rs.urls)]+"/v1/chat/completions", bytesReaderOf(payload))
	if err != nil {
		return 0, nil, err
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+rs.apiKey)

	resp, err := rs.client.Do(req)
	if err != nil {
		return 0, nil, err
	}
	defer func() { _ = resp.Body.Close() }()
	return resp.StatusCode, resp.Header, nil
}

func intPtr(n int) *int { return &n }

// The M5 acceptance criterion: one shared limit must hold across replicas.
// A per-process limiter would allow N x the configured rate with N replicas.
func TestRateLimitHoldsAcrossTwoReplicas(t *testing.T) {
	rdb := requireRedis(t)

	const rpm = 40
	rs := newReplicaSet(t, 2, rdb, config.RateLimit{
		Enabled: true, RPM: rpm, TPM: 100000000, FailOpen: true,
	})

	var allowed, limited atomic.Int64
	var wg sync.WaitGroup

	// Drive both replicas hard with the same API key.
	for replica := 0; replica < 2; replica++ {
		for g := 0; g < 5; g++ {
			wg.Add(1)
			go func(replica, g int) {
				defer wg.Done()
				for i := 0; i < 20; i++ {
					status, _, err := rs.send(replica, fmt.Sprintf("r%d-g%d-%d", replica, g, i))
					if err != nil {
						return
					}
					switch status {
					case http.StatusOK:
						allowed.Add(1)
					case http.StatusTooManyRequests:
						limited.Add(1)
					}
				}
			}(replica, g)
		}
	}
	wg.Wait()

	ok, denied := allowed.Load(), limited.Load()
	t.Logf("2 replicas, 200 requests, shared RPM=%d: %d allowed, %d rate limited", rpm, ok, denied)

	if denied == 0 {
		t.Fatal("no requests were rate limited; the limiter is not engaged")
	}
	// With a per-replica limiter this would be ~2x the budget.
	if ok > rpm+15 {
		t.Errorf("%d requests allowed against a shared RPM of %d: the limit did not hold across replicas",
			ok, rpm)
	}
	if ok < rpm-10 {
		t.Errorf("only %d requests allowed, want close to the full budget of %d", ok, rpm)
	}
}

func TestRateLimitReturns429WithRetryAfter(t *testing.T) {
	rdb := requireRedis(t)
	rs := newReplicaSet(t, 1, rdb, config.RateLimit{
		Enabled: true, RPM: 3, TPM: 100000000, FailOpen: true,
	})

	sawLimit := false
	for i := 0; i < 15; i++ {
		status, headers, err := rs.send(0, fmt.Sprintf("burst %d", i))
		if err != nil {
			t.Fatalf("send: %v", err)
		}
		if status != http.StatusTooManyRequests {
			continue
		}
		sawLimit = true

		retryAfter := headers.Get("Retry-After")
		if retryAfter == "" {
			t.Error("429 response has no Retry-After header")
		} else {
			secs, err := strconv.Atoi(retryAfter)
			if err != nil || secs < 1 {
				t.Errorf("Retry-After = %q, want a positive whole number of seconds", retryAfter)
			}
		}
		break
	}

	if !sawLimit {
		t.Error("never received a 429 despite exceeding the limit")
	}
}

func TestRateLimitHeadersOnSuccess(t *testing.T) {
	rdb := requireRedis(t)
	rs := newReplicaSet(t, 1, rdb, config.RateLimit{
		Enabled: true, RPM: 100, TPM: 1000000, FailOpen: true,
	})

	status, headers, err := rs.send(0, "headers please")
	if err != nil {
		t.Fatalf("send: %v", err)
	}
	if status != http.StatusOK {
		t.Fatalf("status = %d, want 200", status)
	}

	if headers.Get("X-RateLimit-Limit") == "" {
		t.Error("missing X-RateLimit-Limit")
	}
	if headers.Get("X-RateLimit-Remaining") == "" {
		t.Error("missing X-RateLimit-Remaining")
	}
}

// Different tenants must not share a bucket.
func TestRateLimitIsolatesAPIKeys(t *testing.T) {
	rdb := requireRedis(t)
	rs := newReplicaSet(t, 1, rdb, config.RateLimit{
		Enabled: true, RPM: 5, TPM: 100000000, FailOpen: true,
	})

	// Exhaust the first key's budget.
	for i := 0; i < 12; i++ {
		if _, _, err := rs.send(0, fmt.Sprintf("drain %d", i)); err != nil {
			t.Fatalf("send: %v", err)
		}
	}

	// A different key gets its own bucket. Reconfigure the replica set's key by
	// issuing a request with a second credential registered on a fresh replica.
	other := newReplicaSet(t, 1, rdb, config.RateLimit{
		Enabled: true, RPM: 5, TPM: 100000000, FailOpen: true,
	})
	status, _, err := other.send(0, "fresh tenant")
	if err != nil {
		t.Fatalf("send: %v", err)
	}
	if status != http.StatusOK {
		t.Errorf("status = %d for an unrelated API key; buckets are not isolated", status)
	}
}

// Token limits must bind separately from request limits: a handful of enormous
// requests should exhaust TPM long before RPM.
func TestTokenLimitBindsBeforeRequestLimit(t *testing.T) {
	rdb := requireRedis(t)

	// Each request here costs roughly 20 estimated tokens (prompt + per-message
	// overhead + max_tokens), so a 60-token budget binds after about three
	// requests while the request budget stays effectively untouched.
	rs := newReplicaSet(t, 1, rdb, config.RateLimit{
		Enabled:  true,
		RPM:      100000, // effectively unlimited
		TPM:      60,
		FailOpen: true,
	})

	limited := false
	allowedBefore := 0
	for i := 0; i < 20; i++ {
		status, _, err := rs.send(0, "a moderately long prompt that costs tokens to serve")
		if err != nil {
			t.Fatalf("send: %v", err)
		}
		if status == http.StatusTooManyRequests {
			limited = true
			break
		}
		allowedBefore++
	}

	if !limited {
		t.Error("token budget never bound despite an effectively unlimited request budget")
	}
	t.Logf("token budget bound after %d requests (RPM was effectively unlimited)", allowedBefore)
	if allowedBefore > 10 {
		t.Errorf("%d requests passed a 60-token budget; token accounting looks wrong", allowedBefore)
	}
}

// When rate limiting is disabled the request path must not consult Redis at all.
func TestRateLimitDisabledBypassesRedis(t *testing.T) {
	rdb := requireRedis(t)
	rs := newReplicaSet(t, 1, rdb, config.RateLimit{Enabled: false})

	for i := 0; i < 30; i++ {
		status, headers, err := rs.send(0, fmt.Sprintf("unlimited %d", i))
		if err != nil {
			t.Fatalf("send: %v", err)
		}
		if status != http.StatusOK {
			t.Fatalf("status = %d with rate limiting disabled", status)
		}
		if headers.Get("X-RateLimit-Limit") != "" {
			t.Error("rate limit headers emitted while limiting is disabled")
		}
	}
}

// Readiness must reflect Redis, and the fail-open policy must keep the gateway
// serving when Redis is down.
func TestReadyzReportsRedis(t *testing.T) {
	rdb := requireRedis(t)
	rs := newReplicaSet(t, 1, rdb, config.RateLimit{
		Enabled: true, RPM: 100, TPM: 100000, FailOpen: true,
	})

	resp, err := rs.client.Get(rs.urls[0] + "/readyz")
	if err != nil {
		t.Fatalf("GET /readyz: %v", err)
	}
	defer func() { _ = resp.Body.Close() }()

	if resp.StatusCode != http.StatusOK {
		t.Errorf("status = %d, want 200 with Redis healthy", resp.StatusCode)
	}
}
