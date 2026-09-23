package redisclient

import (
	"context"
	"os"
	"testing"
	"time"

	"github.com/aarnavnaga/waypoint/internal/config"
)

func redisAddr() string {
	if v := os.Getenv("WAYPOINT_REDIS_ADDR"); v != "" {
		return v
	}
	return "localhost:6379"
}

func TestNewAppliesDefaultTimeout(t *testing.T) {
	c := New(config.Redis{Addr: redisAddr()})
	defer func() { _ = c.Close() }()

	if c.timeout <= 0 {
		t.Errorf("timeout = %s, want a positive default", c.timeout)
	}
}

func TestNewHonorsConfiguredTimeout(t *testing.T) {
	c := New(config.Redis{
		Addr:    redisAddr(),
		Timeout: config.Duration{Duration: 50 * time.Millisecond},
	})
	defer func() { _ = c.Close() }()

	if c.timeout != 50*time.Millisecond {
		t.Errorf("timeout = %s, want 50ms", c.timeout)
	}
}

func TestReadyAgainstLiveRedis(t *testing.T) {
	c := New(config.Redis{Addr: redisAddr(), Timeout: config.Duration{Duration: time.Second}})
	defer func() { _ = c.Close() }()

	if err := c.Ready(context.Background()); err != nil {
		t.Skipf("Redis not reachable at %s (%v); start it with:\n"+
			"  docker run -d --name waypoint-redis -p 6379:6379 redis/redis-stack-server:7.4.0-v1",
			redisAddr(), err)
	}
}

// Readiness must fail fast rather than hanging: the limiter's fail-open policy
// is only useful if an unreachable Redis is detected quickly.
func TestReadyFailsFastWhenUnreachable(t *testing.T) {
	c := New(config.Redis{
		Addr:    "127.0.0.1:1",
		Timeout: config.Duration{Duration: 100 * time.Millisecond},
	})
	defer func() { _ = c.Close() }()

	start := time.Now()
	err := c.Ready(context.Background())
	elapsed := time.Since(start)

	if err == nil {
		t.Error("Ready reported healthy against a dead address")
	}
	if elapsed > 2*time.Second {
		t.Errorf("Ready took %s to fail; fail-open would add that to every request", elapsed)
	}
}

func TestReadyHonorsContextCancellation(t *testing.T) {
	c := New(config.Redis{Addr: "127.0.0.1:1", Timeout: config.Duration{Duration: 5 * time.Second}})
	defer func() { _ = c.Close() }()

	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	if err := c.Ready(ctx); err == nil {
		t.Error("Ready ignored a cancelled context")
	}
}
