package cache

import (
	"context"
	"errors"
	"log/slog"
	"sync/atomic"

	"github.com/aarnavnaga/waypoint/internal/openai"
)

// Cache coordinates the exact and semantic layers.
//
// Lookup order is exact first, then semantic. Exact is strictly cheaper (one
// Redis GET, no embedding round trip) and strictly safer (identical request, so
// no quality question), so there is never a reason to try semantic first.
type Cache struct {
	exact    *ExactCache
	semantic *SemanticCache
	logger   *slog.Logger

	exactHits    atomic.Uint64
	exactMisses  atomic.Uint64
	semanticHits atomic.Uint64
	semanticMiss atomic.Uint64
	errors       atomic.Uint64
	stores       atomic.Uint64
}

// Options configures a Cache.
type Options struct {
	Exact    *ExactCache
	Semantic *SemanticCache
	Logger   *slog.Logger
}

// New builds a Cache. Either layer may be nil to disable it.
func New(opts Options) *Cache {
	logger := opts.Logger
	if logger == nil {
		logger = slog.Default()
	}
	return &Cache{exact: opts.Exact, semantic: opts.Semantic, logger: logger}
}

// Enabled reports whether any cache layer is configured.
func (c *Cache) Enabled() bool {
	return c != nil && (c.exact != nil || c.semantic != nil)
}

// SemanticEnabled reports whether the semantic layer is configured.
func (c *Cache) SemanticEnabled() bool { return c != nil && c.semantic != nil }

// Semantic exposes the semantic layer for runtime threshold tuning.
func (c *Cache) Semantic() *SemanticCache { return c.semantic }

// Lookup tries both layers, returning ErrMiss when nothing usable is found.
//
// A cache failure is never fatal: Redis being down or the embedder timing out
// degrades to a miss, and the request is served live. A cache that can take down
// the service it accelerates is worse than no cache.
func (c *Cache) Lookup(ctx context.Context, req *openai.ChatRequest) (*Entry, error) {
	if !c.Enabled() || !Cacheable(req) {
		return nil, ErrMiss
	}

	if c.exact != nil {
		entry, err := c.exact.Get(ctx, ExactKey(req))
		switch {
		case err == nil:
			c.exactHits.Add(1)
			return entry, nil
		case errors.Is(err, ErrMiss):
			c.exactMisses.Add(1)
		default:
			c.errors.Add(1)
			c.logger.Warn("exact cache lookup failed", "error", err)
		}
	}

	if c.semantic == nil {
		return nil, ErrMiss
	}

	text, contextHash, ok := SemanticText(req)
	if !ok {
		return nil, ErrMiss
	}

	entry, err := c.semantic.Lookup(ctx, text, req.Model, ParamsFingerprint(req), contextHash)
	switch {
	case err == nil:
		c.semanticHits.Add(1)
		return entry, nil
	case errors.Is(err, ErrMiss):
		c.semanticMiss.Add(1)
	default:
		c.errors.Add(1)
		c.logger.Warn("semantic cache lookup failed", "error", err)
	}
	return nil, ErrMiss
}

// Store writes a completion to both layers.
//
// Callers run this asynchronously: the client already has its response, so
// cache writes must never be on the path that delays it.
func (c *Cache) Store(ctx context.Context, req *openai.ChatRequest, entry *Entry) {
	if !c.Enabled() || !Cacheable(req) {
		return
	}
	// An empty completion is not worth caching and would poison future hits.
	if entry == nil || entry.Content == "" {
		return
	}

	if c.exact != nil {
		if err := c.exact.Set(ctx, ExactKey(req), entry); err != nil {
			c.errors.Add(1)
			c.logger.Warn("exact cache store failed", "error", err)
		}
	}

	if c.semantic != nil {
		if text, contextHash, ok := SemanticText(req); ok {
			if err := c.semantic.Store(ctx, text, req.Model, ParamsFingerprint(req), contextHash, entry); err != nil {
				c.errors.Add(1)
				c.logger.Warn("semantic cache store failed", "error", err)
			}
		}
	}
	c.stores.Add(1)
}

// Stats is a snapshot of cache behavior.
type Stats struct {
	ExactHits      uint64  `json:"exact_hits"`
	ExactMisses    uint64  `json:"exact_misses"`
	SemanticHits   uint64  `json:"semantic_hits"`
	SemanticMisses uint64  `json:"semantic_misses"`
	Stores         uint64  `json:"stores"`
	Errors         uint64  `json:"errors"`
	HitRate        float64 `json:"hit_rate"`
}

// Stats returns cache counters.
func (c *Cache) Stats() Stats {
	s := Stats{
		ExactHits:      c.exactHits.Load(),
		ExactMisses:    c.exactMisses.Load(),
		SemanticHits:   c.semanticHits.Load(),
		SemanticMisses: c.semanticMiss.Load(),
		Stores:         c.stores.Load(),
		Errors:         c.errors.Load(),
	}

	hits := s.ExactHits + s.SemanticHits
	// A semantic miss only happens after an exact miss, so counting exact
	// misses alone would double-count lookups.
	total := hits + s.ExactMisses
	if s.ExactMisses == 0 {
		total = hits + s.SemanticMisses
	}
	if total > 0 {
		s.HitRate = float64(hits) / float64(total)
	}
	return s
}

// ResetStats clears counters between benchmark runs.
func (c *Cache) ResetStats() {
	c.exactHits.Store(0)
	c.exactMisses.Store(0)
	c.semanticHits.Store(0)
	c.semanticMiss.Store(0)
	c.errors.Store(0)
	c.stores.Store(0)
}
