package resilience

import (
	"context"
	"log/slog"
	"net/http"
	"sync"
	"time"

	"github.com/aarnavnaga/waypoint/internal/backend"
)

// HealthCheckerOptions configures active probing.
type HealthCheckerOptions struct {
	// Interval between probe rounds.
	Interval time.Duration
	// Timeout bounds a single probe.
	Timeout time.Duration
	// Path is the probe endpoint, e.g. /healthz.
	Path string
	// HealthyThreshold is how many consecutive successes restore a backend.
	HealthyThreshold int
	// UnhealthyThreshold is how many consecutive failures remove one.
	UnhealthyThreshold int
	Client             *http.Client
	Logger             *slog.Logger
}

func (o *HealthCheckerOptions) applyDefaults() {
	if o.Interval <= 0 {
		o.Interval = 2 * time.Second
	}
	if o.Timeout <= 0 {
		o.Timeout = time.Second
	}
	if o.Path == "" {
		o.Path = "/healthz"
	}
	if o.HealthyThreshold < 1 {
		o.HealthyThreshold = 2
	}
	if o.UnhealthyThreshold < 1 {
		o.UnhealthyThreshold = 3
	}
	if o.Client == nil {
		o.Client = http.DefaultClient
	}
	if o.Logger == nil {
		o.Logger = slog.Default()
	}
}

// HealthChecker probes backends and maintains their health flag.
//
// Health is tracked with hysteresis — N consecutive results, not one — because
// a single probe is a noisy sample. Flipping a backend out of rotation on one
// timeout causes traffic to oscillate, and oscillating traffic is itself a cause
// of timeouts. Requiring consecutive agreement damps that feedback loop.
//
// Active probing complements the passive signal from the circuit breaker. They
// answer different questions: a breaker observes *real* traffic, so it notices
// problems that only appear under load, but it cannot tell when a backend has
// recovered because it has stopped sending traffic there. Active probes keep
// testing a backend nobody is using, which is exactly what is needed to bring
// one back.
type HealthChecker struct {
	pool *backend.Pool
	opts HealthCheckerOptions

	mu    sync.Mutex
	state map[string]*probeState
}

type probeState struct {
	consecutiveOK   int
	consecutiveFail int
}

// NewHealthChecker builds a HealthChecker.
func NewHealthChecker(pool *backend.Pool, opts HealthCheckerOptions) *HealthChecker {
	opts.applyDefaults()

	hc := &HealthChecker{
		pool:  pool,
		opts:  opts,
		state: make(map[string]*probeState),
	}
	for _, b := range pool.All() {
		hc.state[b.Name] = &probeState{}
	}
	return hc
}

// Run probes until ctx is cancelled.
func (h *HealthChecker) Run(ctx context.Context) {
	ticker := time.NewTicker(h.opts.Interval)
	defer ticker.Stop()

	h.CheckAll(ctx)

	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			h.CheckAll(ctx)
		}
	}
}

// CheckAll probes every backend in parallel.
func (h *HealthChecker) CheckAll(ctx context.Context) {
	var wg sync.WaitGroup
	for _, b := range h.pool.All() {
		wg.Add(1)
		go func(b *backend.Backend) {
			defer wg.Done()
			h.Check(ctx, b)
		}(b)
	}
	wg.Wait()
}

// Check probes one backend and updates its health.
//
// A probe aborted because the *caller's* context was cancelled — shutdown, say —
// is not evidence about the backend, so it is discarded rather than recorded as
// a failure. Counting it would let a restart leave every backend looking
// unhealthy in the final moments before exit.
func (h *HealthChecker) Check(ctx context.Context, b *backend.Backend) {
	ok := h.probe(ctx, b)
	if !ok && ctx.Err() != nil {
		return
	}
	h.record(b, ok)
}

func (h *HealthChecker) probe(ctx context.Context, b *backend.Backend) bool {
	probeCtx, cancel := context.WithTimeout(ctx, h.opts.Timeout)
	defer cancel()

	target := b.URL.JoinPath(h.opts.Path)
	req, err := http.NewRequestWithContext(probeCtx, http.MethodGet, target.String(), nil)
	if err != nil {
		return false
	}

	resp, err := h.opts.Client.Do(req)
	if err != nil {
		return false
	}
	defer func() { _ = resp.Body.Close() }()

	return resp.StatusCode >= 200 && resp.StatusCode < 300
}

// record applies hysteresis and flips health when a threshold is crossed.
func (h *HealthChecker) record(b *backend.Backend, ok bool) {
	h.mu.Lock()
	st, exists := h.state[b.Name]
	if !exists {
		st = &probeState{}
		h.state[b.Name] = st
	}

	if ok {
		st.consecutiveOK++
		st.consecutiveFail = 0
	} else {
		st.consecutiveFail++
		st.consecutiveOK = 0
	}

	okCount, failCount := st.consecutiveOK, st.consecutiveFail
	h.mu.Unlock()

	switch {
	case ok && okCount >= h.opts.HealthyThreshold && !b.Healthy():
		if b.SetHealthy(true) {
			h.opts.Logger.Info("backend restored to rotation",
				"backend", b.Name, "consecutive_successes", okCount)
		}
	case !ok && failCount >= h.opts.UnhealthyThreshold && b.Healthy():
		if b.SetHealthy(false) {
			h.opts.Logger.Warn("backend removed from rotation",
				"backend", b.Name, "consecutive_failures", failCount)
		}
	}
}

// ProbeState reports consecutive counters for a backend, for tests and /admin.
func (h *HealthChecker) ProbeState(name string) (consecutiveOK, consecutiveFail int) {
	h.mu.Lock()
	defer h.mu.Unlock()

	if st, ok := h.state[name]; ok {
		return st.consecutiveOK, st.consecutiveFail
	}
	return 0, 0
}
