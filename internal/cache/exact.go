package cache

import (
	"context"
	"errors"
	"fmt"
	"time"

	"github.com/redis/go-redis/v9"
)

// ErrMiss indicates nothing was found.
var ErrMiss = errors.New("cache: miss")

// ExactCache stores completions keyed by an exact hash of the request.
//
// This is the cheap, safe half of caching: a hit is byte-for-byte the response
// the same request produced before, so there is no quality question — only a
// freshness one, handled by the TTL.
type ExactCache struct {
	client    redis.Cmdable
	ttl       time.Duration
	keyPrefix string
	timeout   time.Duration
}

// ExactOptions configures an ExactCache.
type ExactOptions struct {
	TTL       time.Duration
	KeyPrefix string
	Timeout   time.Duration
}

// NewExactCache builds an ExactCache.
func NewExactCache(client redis.Cmdable, opts ExactOptions) *ExactCache {
	if opts.TTL <= 0 {
		opts.TTL = time.Hour
	}
	if opts.KeyPrefix == "" {
		opts.KeyPrefix = "waypoint"
	}
	if opts.Timeout <= 0 {
		opts.Timeout = 200 * time.Millisecond
	}
	return &ExactCache{
		client:    client,
		ttl:       opts.TTL,
		keyPrefix: opts.KeyPrefix,
		timeout:   opts.Timeout,
	}
}

func (c *ExactCache) redisKey(hash string) string {
	return fmt.Sprintf("%s:cache:exact:%s", c.keyPrefix, hash)
}

// Get looks up a cached entry, returning ErrMiss when absent.
func (c *ExactCache) Get(ctx context.Context, hash string) (*Entry, error) {
	callCtx, cancel := context.WithTimeout(ctx, c.timeout)
	defer cancel()

	data, err := c.client.Get(callCtx, c.redisKey(hash)).Bytes()
	if errors.Is(err, redis.Nil) {
		return nil, ErrMiss
	}
	if err != nil {
		return nil, fmt.Errorf("cache: get: %w", err)
	}

	entry, err := DecodeEntry(data)
	if err != nil {
		// A corrupt entry is treated as a miss rather than an error: the
		// request can still be served live, and the bad key will expire.
		return nil, ErrMiss
	}
	entry.Source = SourceExact
	return entry, nil
}

// Set stores an entry under hash.
func (c *ExactCache) Set(ctx context.Context, hash string, entry *Entry) error {
	data, err := entry.Encode()
	if err != nil {
		return err
	}

	callCtx, cancel := context.WithTimeout(ctx, c.timeout)
	defer cancel()

	if err := c.client.Set(callCtx, c.redisKey(hash), data, c.ttl).Err(); err != nil {
		return fmt.Errorf("cache: set: %w", err)
	}
	return nil
}

// Delete removes an entry, used by tests and cache invalidation.
func (c *ExactCache) Delete(ctx context.Context, hash string) error {
	callCtx, cancel := context.WithTimeout(ctx, c.timeout)
	defer cancel()
	return c.client.Del(callCtx, c.redisKey(hash)).Err()
}
