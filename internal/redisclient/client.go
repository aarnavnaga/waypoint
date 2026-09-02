// Package redisclient builds the shared Redis connection used for rate limits
// and caches.
package redisclient

import (
	"context"
	"fmt"
	"time"

	"github.com/redis/go-redis/v9"

	"github.com/aarnavnaga/waypoint/internal/config"
)

// Client wraps a go-redis client with Waypoint's readiness semantics.
type Client struct {
	*redis.Client
	timeout time.Duration
}

// New builds a Redis client from configuration.
//
// Timeouts are deliberately short and uniform. Redis sits on the request path
// for rate limiting and cache lookups, so a slow Redis must degrade into a fast
// local decision rather than adding its latency to every request. The limiter's
// fail-open policy depends on that: failing open is only useful if the failure
// is detected quickly.
func New(cfg config.Redis) *Client {
	timeout := cfg.Timeout.Duration
	if timeout <= 0 {
		timeout = 200 * time.Millisecond
	}

	return &Client{
		Client: redis.NewClient(&redis.Options{
			Addr:            cfg.Addr,
			Password:        cfg.Password,
			DB:              cfg.DB,
			DialTimeout:     timeout,
			ReadTimeout:     timeout,
			WriteTimeout:    timeout,
			PoolSize:        32,
			MinIdleConns:    4,
			ConnMaxIdleTime: 5 * time.Minute,
			// Retries are handled by Waypoint's own policy, not silently here.
			MaxRetries: 0,
		}),
		timeout: timeout,
	}
}

// Ready reports whether Redis is reachable, satisfying api.Readiness.
func (c *Client) Ready(ctx context.Context) error {
	pingCtx, cancel := context.WithTimeout(ctx, c.timeout)
	defer cancel()

	if err := c.Ping(pingCtx).Err(); err != nil {
		return fmt.Errorf("redis ping: %w", err)
	}
	return nil
}
