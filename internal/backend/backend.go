// Package backend models the gateway's view of an upstream model server:
// its address, its current load, its health, and the models it serves.
package backend

import (
	"fmt"
	"net/url"
	"sort"
	"sync"
	"sync/atomic"

	"github.com/aarnavnaga/waypoint/internal/config"
)

// Backend is one upstream model server.
//
// Load is tracked in *estimated tokens*, not request count. Two requests can
// differ by three orders of magnitude in cost, so counting requests would make a
// server holding one 32k-token prompt look idler than one holding five 100-token
// prompts.
type Backend struct {
	Name              string
	URL               *url.URL
	MaxInflightTokens int64

	inflightTokens atomic.Int64
	inflightReqs   atomic.Int64

	// healthy is maintained by the health checker (M4). Until then it starts
	// true so a fresh pool is immediately usable.
	healthy atomic.Bool

	// models holds the discovered model list, swapped atomically.
	models atomic.Pointer[[]string]

	// Breaker is attached in M4; nil means "no breaker configured".
	breaker atomic.Pointer[breakerHandle]

	totalRequests atomic.Uint64
	totalTokens   atomic.Uint64
}

// breakerHandle is an indirection so internal/backend does not import
// internal/resilience (which would be a cycle). M4 registers an implementation.
type breakerHandle struct {
	// Allow reports whether a new request may be sent.
	Allow func() bool
	// State returns a human-readable breaker state for /admin and metrics.
	State func() string
}

// New builds a Backend from its configuration entry.
func New(cfg config.Backend) (*Backend, error) {
	u, err := url.Parse(cfg.URL)
	if err != nil {
		return nil, fmt.Errorf("backend %s: parse url: %w", cfg.Name, err)
	}
	if u.Scheme == "" || u.Host == "" {
		return nil, fmt.Errorf("backend %s: url %q must be absolute", cfg.Name, cfg.URL)
	}

	b := &Backend{
		Name:              cfg.Name,
		URL:               u,
		MaxInflightTokens: int64(cfg.MaxInflightTokens),
	}
	b.healthy.Store(true)

	models := append([]string(nil), cfg.Models...)
	b.models.Store(&models)
	return b, nil
}

// InflightTokens reports the current estimated token load.
func (b *Backend) InflightTokens() int64 { return b.inflightTokens.Load() }

// InflightRequests reports how many requests are in flight.
func (b *Backend) InflightRequests() int64 { return b.inflightReqs.Load() }

// Reserve accounts for a request's estimated cost and returns a release
// function that must be called exactly once when the request finishes.
func (b *Backend) Reserve(tokens int64) (release func()) {
	if tokens < 0 {
		tokens = 0
	}
	b.inflightTokens.Add(tokens)
	b.inflightReqs.Add(1)
	b.totalRequests.Add(1)
	b.totalTokens.Add(uint64(tokens))

	var once sync.Once
	return func() {
		once.Do(func() {
			b.inflightTokens.Add(-tokens)
			b.inflightReqs.Add(-1)
		})
	}
}

// HasCapacity reports whether the backend is below its token ceiling.
// A MaxInflightTokens of 0 means unbounded.
func (b *Backend) HasCapacity() bool {
	if b.MaxInflightTokens <= 0 {
		return true
	}
	return b.inflightTokens.Load() < b.MaxInflightTokens
}

// Healthy reports the backend's current health.
func (b *Backend) Healthy() bool { return b.healthy.Load() }

// SetHealthy updates health, returning true if the state changed.
func (b *Backend) SetHealthy(v bool) bool { return b.healthy.Swap(v) != v }

// Available reports whether the backend may receive a new request: healthy,
// below its token ceiling, and not cut off by an open circuit breaker.
func (b *Backend) Available() bool {
	if !b.Healthy() || !b.HasCapacity() {
		return false
	}
	if h := b.breaker.Load(); h != nil && h.Allow != nil {
		return h.Allow()
	}
	return true
}

// SetBreaker attaches circuit-breaker callbacks (M4).
func (b *Backend) SetBreaker(allow func() bool, state func() string) {
	b.breaker.Store(&breakerHandle{Allow: allow, State: state})
}

// BreakerState returns the breaker state, or "none" when unconfigured.
func (b *Backend) BreakerState() string {
	if h := b.breaker.Load(); h != nil && h.State != nil {
		return h.State()
	}
	return "none"
}

// Models returns the backend's known model IDs.
func (b *Backend) Models() []string {
	if p := b.models.Load(); p != nil {
		return append([]string(nil), *p...)
	}
	return nil
}

// SetModels replaces the discovered model list.
func (b *Backend) SetModels(models []string) {
	cp := append([]string(nil), models...)
	b.models.Store(&cp)
}

// ServesModel reports whether the backend can serve the named model. A backend
// with no known models is assumed to serve anything, so model discovery failing
// degrades to "route anywhere" rather than "route nowhere".
func (b *Backend) ServesModel(model string) bool {
	models := b.Models()
	if len(models) == 0 {
		return true
	}
	for _, m := range models {
		if m == model {
			return true
		}
	}
	return false
}

// Stats is a snapshot of one backend, used by /admin/backends.
type Stats struct {
	Name              string   `json:"name"`
	URL               string   `json:"url"`
	Healthy           bool     `json:"healthy"`
	Available         bool     `json:"available"`
	BreakerState      string   `json:"breaker_state"`
	InflightTokens    int64    `json:"inflight_tokens"`
	InflightRequests  int64    `json:"inflight_requests"`
	MaxInflightTokens int64    `json:"max_inflight_tokens"`
	TotalRequests     uint64   `json:"total_requests"`
	TotalTokens       uint64   `json:"total_tokens"`
	Models            []string `json:"models"`
}

// Stats returns a snapshot of this backend.
func (b *Backend) Stats() Stats {
	return Stats{
		Name:              b.Name,
		URL:               b.URL.String(),
		Healthy:           b.Healthy(),
		Available:         b.Available(),
		BreakerState:      b.BreakerState(),
		InflightTokens:    b.InflightTokens(),
		InflightRequests:  b.InflightRequests(),
		MaxInflightTokens: b.MaxInflightTokens,
		TotalRequests:     b.totalRequests.Load(),
		TotalTokens:       b.totalTokens.Load(),
		Models:            b.Models(),
	}
}

// Pool is the set of configured backends. The set itself is fixed at startup;
// only per-backend state changes at runtime, so reads need no locking.
type Pool struct {
	backends []*Backend
	byName   map[string]*Backend
}

// NewPool builds a Pool from configuration.
func NewPool(cfgs []config.Backend) (*Pool, error) {
	if len(cfgs) == 0 {
		return nil, fmt.Errorf("pool requires at least one backend")
	}

	p := &Pool{
		backends: make([]*Backend, 0, len(cfgs)),
		byName:   make(map[string]*Backend, len(cfgs)),
	}
	for _, c := range cfgs {
		b, err := New(c)
		if err != nil {
			return nil, err
		}
		if _, dup := p.byName[b.Name]; dup {
			return nil, fmt.Errorf("duplicate backend name %q", b.Name)
		}
		p.backends = append(p.backends, b)
		p.byName[b.Name] = b
	}
	return p, nil
}

// All returns every backend in configuration order.
func (p *Pool) All() []*Backend { return p.backends }

// Len reports the number of configured backends.
func (p *Pool) Len() int { return len(p.backends) }

// Get looks up a backend by name.
func (p *Pool) Get(name string) (*Backend, bool) {
	b, ok := p.byName[name]
	return b, ok
}

// Available returns backends that may receive a request for the given model.
// Pass an empty model to skip the model filter.
func (p *Pool) Available(model string) []*Backend {
	out := make([]*Backend, 0, len(p.backends))
	for _, b := range p.backends {
		if !b.Available() {
			continue
		}
		if model != "" && !b.ServesModel(model) {
			continue
		}
		out = append(out, b)
	}
	return out
}

// Healthy returns every healthy backend, ignoring capacity and breaker state.
// Used by /readyz, which asks "could we serve anything?" rather than "can we
// serve right now?".
func (p *Pool) Healthy() []*Backend {
	out := make([]*Backend, 0, len(p.backends))
	for _, b := range p.backends {
		if b.Healthy() {
			out = append(out, b)
		}
	}
	return out
}

// TotalInflightTokens sums estimated load across the pool.
func (p *Pool) TotalInflightTokens() int64 {
	var total int64
	for _, b := range p.backends {
		total += b.InflightTokens()
	}
	return total
}

// AverageInflightTokens returns mean load across the given backends, which the
// load-aware routing policy compares against.
func AverageInflightTokens(backends []*Backend) float64 {
	if len(backends) == 0 {
		return 0
	}
	var total int64
	for _, b := range backends {
		total += b.InflightTokens()
	}
	return float64(total) / float64(len(backends))
}

// UnionModels returns the sorted, deduplicated set of models served by the
// given backends. This backs GET /v1/models.
func UnionModels(backends []*Backend) []string {
	seen := make(map[string]struct{})
	for _, b := range backends {
		for _, m := range b.Models() {
			seen[m] = struct{}{}
		}
	}

	out := make([]string, 0, len(seen))
	for m := range seen {
		out = append(out, m)
	}
	sort.Strings(out)
	return out
}
