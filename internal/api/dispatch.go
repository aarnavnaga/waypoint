package api

import (
	"context"
	"errors"
	"log/slog"
	"net/http"

	"github.com/aarnavnaga/waypoint/internal/backend"
	"github.com/aarnavnaga/waypoint/internal/config"
	"github.com/aarnavnaga/waypoint/internal/openai"
	"github.com/aarnavnaga/waypoint/internal/resilience"
	"github.com/aarnavnaga/waypoint/internal/router"
	"github.com/aarnavnaga/waypoint/internal/telemetry"
)

// Dispatcher turns one client request into up to MaxAttempts backend attempts,
// enforcing circuit breakers, a retry budget and failover rules.
type Dispatcher struct {
	pool     *backend.Pool
	router   *router.Router
	proxy    *Proxy
	breakers *Breakers
	budget   *resilience.RetryBudget
	backoff  *resilience.Backoff

	maxAttempts int
	metrics     *telemetry.Metrics
	logger      *slog.Logger
}

// DispatcherOptions configures a Dispatcher.
type DispatcherOptions struct {
	Pool        *backend.Pool
	Router      *router.Router
	Proxy       *Proxy
	Breakers    *Breakers
	Budget      *resilience.RetryBudget
	Backoff     *resilience.Backoff
	MaxAttempts int
	Metrics     *telemetry.Metrics
	Logger      *slog.Logger
}

// NewDispatcher builds a Dispatcher.
func NewDispatcher(opts DispatcherOptions) *Dispatcher {
	if opts.MaxAttempts < 1 {
		opts.MaxAttempts = 1
	}
	if opts.Logger == nil {
		opts.Logger = slog.Default()
	}
	return &Dispatcher{
		pool:        opts.Pool,
		router:      opts.Router,
		proxy:       opts.Proxy,
		breakers:    opts.Breakers,
		budget:      opts.Budget,
		backoff:     opts.Backoff,
		maxAttempts: opts.MaxAttempts,
		metrics:     opts.Metrics,
		logger:      opts.Logger,
	}
}

// Result summarizes a completed dispatch across all attempts.
type Result struct {
	Outcome  *Outcome
	Attempts int
	// Tried lists the backends attempted, in order.
	Tried []string
}

// Do routes and dispatches req, retrying on other backends when it is safe.
//
// The governing rule is that a retry is only legal while nothing has been
// written to the client. Once the first token is streamed the response is
// committed, so a failure past that point is terminal no matter how many
// attempts remain — retrying would splice two independent generations into one
// stream.
func (d *Dispatcher) Do(
	ctx context.Context,
	w http.ResponseWriter,
	req *openai.ChatRequest,
	routeReq *router.Request,
	body []byte,
	headers http.Header,
) (*Result, error) {
	result := &Result{}
	tried := make(map[string]bool, d.maxAttempts)

	var lastErr error

	for attempt := 0; attempt < d.maxAttempts; attempt++ {
		if err := ctx.Err(); err != nil {
			return result, err
		}

		target, err := d.selectBackend(ctx, routeReq, tried)
		if err != nil {
			if lastErr != nil {
				// Report the backend failure rather than "nothing left to try",
				// which is a consequence rather than a cause.
				return result, lastErr
			}
			return result, err
		}

		tried[target.Name] = true

		// Reserve the breaker slot at the moment of dispatch, not during
		// filtering. In half-open state this consumes one of the limited probe
		// slots, and the matching Record below releases it.
		if breaker := d.breakers.For(target.Name); breaker != nil && !breaker.Allow() {
			d.logger.Debug("breaker refused at dispatch time", "backend", target.Name)
			continue
		}

		result.Tried = append(result.Tried, target.Name)
		result.Attempts = attempt + 1

		// Account for this attempt's estimated cost for the duration of the
		// call, so concurrent routing decisions see it.
		release := target.Reserve(routeReq.EstimatedTokens)
		outcome, err := d.proxy.Dispatch(ctx, w, target, req, body, headers)
		release()

		result.Outcome = outcome
		d.recordOutcome(target, err)

		if err == nil {
			return result, nil
		}
		lastErr = err

		// The client is gone; further attempts would serve nobody.
		if ctx.Err() != nil {
			return result, err
		}

		// Past the first streamed byte, failover is impossible.
		if outcome != nil && !outcome.Retryable() {
			if d.metrics != nil {
				d.metrics.RetriesTotal.WithLabelValues("not_retryable").Inc()
			}
			d.logger.WarnContext(ctx, "backend failed after the response began; cannot fail over",
				"backend", target.Name, "error", err)
			return result, err
		}

		if attempt+1 >= d.maxAttempts {
			break
		}

		// A retry must be affordable. Under a broad outage the budget empties
		// and retries stop, which is the entire point: retrying everything is
		// how a partial outage becomes a total one.
		if d.budget != nil && !d.budget.Withdraw() {
			if d.metrics != nil {
				d.metrics.RetriesTotal.WithLabelValues("budget_exhausted").Inc()
			}
			d.logger.WarnContext(ctx, "retry budget exhausted; not retrying",
				"backend", target.Name, "error", err)
			break
		}

		d.recordRetry(err)
		d.logger.InfoContext(ctx, "retrying on another backend",
			"failed_backend", target.Name, "attempt", attempt+1, "error", err)

		if d.backoff != nil && !d.backoff.Wait(ctx, attempt) {
			return result, ctx.Err()
		}
	}

	return result, lastErr
}

// selectBackend picks an eligible backend that has not already been tried.
func (d *Dispatcher) selectBackend(
	ctx context.Context,
	routeReq *router.Request,
	tried map[string]bool,
) (*backend.Backend, error) {
	candidates := d.pool.Available(routeReq.Model)

	remaining := make([]*backend.Backend, 0, len(candidates))
	for _, b := range candidates {
		if !tried[b.Name] {
			remaining = append(remaining, b)
		}
	}
	if len(remaining) == 0 {
		return nil, router.ErrNoBackends
	}

	return d.router.Pick(ctx, routeReq, remaining)
}

// recordOutcome feeds the result back into the backend's circuit breaker.
//
// This is the passive half of health detection: it observes real traffic, so it
// notices failures that only appear under load, which an active probe of
// /healthz would never see.
func (d *Dispatcher) recordOutcome(b *backend.Backend, err error) {
	if d.breakers == nil {
		return
	}
	breaker := d.breakers.For(b.Name)
	if breaker == nil {
		return
	}

	switch {
	case err == nil:
		breaker.Record(resilience.OutcomeSuccess)
	case errors.Is(err, ErrTTFTTimeout), errors.Is(err, context.DeadlineExceeded):
		breaker.Record(resilience.OutcomeTimeout)
		d.countUpstreamError(b.Name, "timeout")
	case errors.Is(err, context.Canceled):
		// The client hung up. That is not the backend's fault and must not
		// count against it, or a burst of impatient clients would trip every
		// breaker in the fleet.
	default:
		breaker.Record(resilience.OutcomeFailure)
		d.countUpstreamError(b.Name, upstreamErrorKind(err))
	}
}

// upstreamErrorKind classifies a failure for the metric label, keeping
// cardinality bounded — raw error strings would create an unbounded label set.
func upstreamErrorKind(err error) string {
	switch {
	case errors.Is(err, ErrUpstreamUnavailable):
		return "unreachable"
	case errors.Is(err, ErrUpstreamServerError):
		return "server_error"
	default:
		return "other"
	}
}

func (d *Dispatcher) countUpstreamError(backendName, kind string) {
	if d.metrics != nil {
		d.metrics.UpstreamErrors.WithLabelValues(backendName, kind).Inc()
	}
}

// recordRetry counts a retry by the reason that triggered it.
func (d *Dispatcher) recordRetry(err error) {
	if d.metrics == nil {
		return
	}
	reason := "upstream_error"
	switch {
	case errors.Is(err, ErrTTFTTimeout):
		reason = "ttft_timeout"
	case errors.Is(err, ErrUpstreamUnavailable):
		reason = "unreachable"
	case errors.Is(err, ErrUpstreamServerError):
		reason = "server_error"
	}
	d.metrics.RetriesTotal.WithLabelValues(reason).Inc()
}

// Breakers owns one circuit breaker per backend and wires them into the pool so
// routing can skip a backend whose circuit is open.
type Breakers struct {
	byName map[string]*resilience.Breaker
}

// NewBreakers builds a breaker per backend and attaches it.
func NewBreakers(pool *backend.Pool, cfg config.Breaker) *Breakers {
	b := &Breakers{byName: make(map[string]*resilience.Breaker, pool.Len())}

	for _, bk := range pool.All() {
		breaker := resilience.NewBreaker(resilience.BreakerOptions{
			Window:           cfg.Window.Duration,
			Buckets:          cfg.Buckets,
			MinRequests:      cfg.MinRequests,
			ErrorRate:        cfg.ErrorRate,
			TimeoutRate:      cfg.TimeoutRate,
			OpenDuration:     cfg.OpenDuration.Duration,
			HalfOpenRequests: cfg.HalfOpenRequests,
		})
		b.byName[bk.Name] = breaker

		// backend.Available() consults this during candidate filtering, which
		// happens for every backend on every request. It must therefore be the
		// *non-mutating* query: using Allow here would consume half-open probe
		// slots for backends that are never dispatched to, and the breaker
		// would never close again.
		bk.SetBreaker(
			breaker.Permits,
			func() string { return breaker.State().String() },
		)
	}
	return b
}

// For returns the breaker for a backend, or nil.
func (b *Breakers) For(name string) *resilience.Breaker {
	if b == nil {
		return nil
	}
	return b.byName[name]
}

// Stats returns every breaker's state, for /admin/backends.
func (b *Breakers) Stats() map[string]resilience.Stats {
	out := make(map[string]resilience.Stats, len(b.byName))
	for name, breaker := range b.byName {
		out[name] = breaker.Stats()
	}
	return out
}

// Reset clears every breaker, used between benchmark runs.
func (b *Breakers) Reset() {
	for _, breaker := range b.byName {
		breaker.Reset()
	}
}
