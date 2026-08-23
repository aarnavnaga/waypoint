package router

import (
	"context"
	"math"

	"github.com/aarnavnaga/waypoint/internal/backend"
	"github.com/aarnavnaga/waypoint/internal/config"
)

// LeastLoaded picks the backend with the fewest outstanding estimated tokens.
//
// Tokens rather than requests: a server holding one 32k-token prompt is far
// busier than one holding five 100-token prompts, and request counts cannot see
// that. This is the right policy when requests are independent, and it is the
// fallback every affinity policy uses when locality would cost too much.
type LeastLoaded struct{}

// NewLeastLoaded returns a LeastLoaded policy.
func NewLeastLoaded() *LeastLoaded { return &LeastLoaded{} }

// Name implements Policy.
func (*LeastLoaded) Name() string { return config.PolicyLeastLoaded }

// Pick implements Policy.
func (*LeastLoaded) Pick(_ context.Context, _ *Request, backends []*backend.Backend) (*backend.Backend, error) {
	return leastLoaded(backends)
}

// leastLoaded returns the backend with the lowest in-flight token count.
// Ties go to the earlier backend, which keeps selection deterministic.
func leastLoaded(backends []*backend.Backend) (*backend.Backend, error) {
	if len(backends) == 0 {
		return nil, ErrNoBackends
	}

	best := backends[0]
	bestLoad := best.InflightTokens()
	for _, b := range backends[1:] {
		if load := b.InflightTokens(); load < bestLoad {
			best, bestLoad = b, load
		}
	}
	return best, nil
}

// PrefixAffinity routes by prompt prefix using consistent hashing with bounded
// loads (Mirrokni, Thorup & Zadimoghaddam, 2016).
//
// Plain consistent hashing maximizes cache locality but has no safety valve: if
// one system prompt is far more popular than the others — and prompt popularity
// is heavily skewed in practice — its backend is buried while its peers idle.
// Bounded loads add a capacity ceiling of c x average. A key goes to its
// preferred backend only if that backend is under the ceiling; otherwise the
// walk continues clockwise to the next one. This preserves locality for most
// keys while guaranteeing no backend exceeds the ceiling, so the tail latency
// caused by a hot prefix is bounded rather than unbounded.
type PrefixAffinity struct {
	vnodes     int
	loadFactor float64
	rings      ringCache
}

// NewPrefixAffinity returns a PrefixAffinity policy. loadFactor is c, the
// multiple of average load a backend may carry before keys spill to its
// successor. c must be >= 1; values near 1 balance aggressively at the cost of
// locality, large values approach plain consistent hashing.
func NewPrefixAffinity(vnodes int, loadFactor float64) *PrefixAffinity {
	if loadFactor < 1 {
		loadFactor = 1
	}
	return &PrefixAffinity{vnodes: vnodes, loadFactor: loadFactor}
}

// Name implements Policy.
func (*PrefixAffinity) Name() string { return config.PolicyPrefixAffinity }

// Pick implements Policy.
func (p *PrefixAffinity) Pick(_ context.Context, req *Request, backends []*backend.Backend) (*backend.Backend, error) {
	if len(backends) == 0 {
		return nil, ErrNoBackends
	}
	if len(backends) == 1 {
		return backends[0], nil
	}

	ring := p.rings.get(backends, p.vnodes)
	capacity := boundedLoadCapacity(backends, req.EstimatedTokens, p.loadFactor)

	var chosen *backend.Backend
	ring.walk(hashString(req.PrefixKey), func(idx int) bool {
		candidate := backends[idx]
		if candidate.InflightTokens() <= capacity {
			chosen = candidate
			return false // stop walking
		}
		return true // over capacity: try the next backend clockwise
	})

	if chosen != nil {
		return chosen, nil
	}
	// Every backend is over the ceiling, which means the whole fleet is
	// saturated rather than imbalanced. Fall back to the least bad option.
	return leastLoaded(backends)
}

// PrefixAffinityLoadAware routes by prefix, but abandons affinity entirely when
// the preferred backend is more than a factor c above the pool average.
//
// The difference from PrefixAffinity is where the request goes when the
// preferred backend is too hot. PrefixAffinity walks to the *next* backend on
// the ring, keeping some locality and a stable fallback order. This policy jumps
// to the globally least loaded backend instead, trading locality for the best
// available balance. Which wins is an empirical question, which is why both
// exist and M8 measures them.
type PrefixAffinityLoadAware struct {
	vnodes     int
	loadFactor float64
	rings      ringCache
}

// NewPrefixAffinityLoadAware returns a PrefixAffinityLoadAware policy.
func NewPrefixAffinityLoadAware(vnodes int, loadFactor float64) *PrefixAffinityLoadAware {
	if loadFactor < 1 {
		loadFactor = 1
	}
	return &PrefixAffinityLoadAware{vnodes: vnodes, loadFactor: loadFactor}
}

// Name implements Policy.
func (*PrefixAffinityLoadAware) Name() string { return config.PolicyPrefixAffinityLoadAware }

// Pick implements Policy.
func (p *PrefixAffinityLoadAware) Pick(_ context.Context, req *Request, backends []*backend.Backend) (*backend.Backend, error) {
	if len(backends) == 0 {
		return nil, ErrNoBackends
	}
	if len(backends) == 1 {
		return backends[0], nil
	}

	ring := p.rings.get(backends, p.vnodes)

	// The preferred backend is simply the first on the ring; no capacity check
	// happens during the walk.
	var preferred *backend.Backend
	ring.walk(hashString(req.PrefixKey), func(idx int) bool {
		preferred = backends[idx]
		return false
	})
	if preferred == nil {
		return leastLoaded(backends)
	}

	avg := backend.AverageInflightTokens(backends)
	// With an idle pool there is no imbalance to correct, so take the locality.
	if avg <= 0 {
		return preferred, nil
	}

	if float64(preferred.InflightTokens()) > p.loadFactor*avg {
		return leastLoaded(backends)
	}
	return preferred, nil
}

// boundedLoadCapacity computes the per-backend ceiling c x average.
//
// The incoming request's own cost is included in the average so an otherwise
// idle pool does not produce a zero ceiling that rejects every backend.
func boundedLoadCapacity(backends []*backend.Backend, incoming int64, c float64) int64 {
	if len(backends) == 0 {
		return math.MaxInt64
	}

	var total int64
	for _, b := range backends {
		total += b.InflightTokens()
	}
	total += incoming

	avg := float64(total) / float64(len(backends))
	capacity := int64(math.Ceil(c * avg))
	if capacity < 1 {
		// A completely idle pool: any backend will do, so take pure affinity.
		return math.MaxInt64
	}
	return capacity
}
