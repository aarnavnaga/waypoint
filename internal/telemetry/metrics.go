// Package telemetry owns Waypoint's metrics, tracing and structured logging.
package telemetry

import (
	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/collectors"
	"github.com/prometheus/client_golang/prometheus/promhttp"
	"net/http"
)

// Metrics holds every instrument the gateway exports.
//
// Label cardinality is the thing to be careful about: every distinct label
// combination is a separate time series, so model IDs and backend names (bounded
// sets) are labels, while API keys, prompt hashes and request IDs are not — those
// belong in traces and logs, which are sampled and indexed differently.
type Metrics struct {
	registry *prometheus.Registry

	RequestsTotal   *prometheus.CounterVec
	TTFTSeconds     *prometheus.HistogramVec
	DurationSeconds *prometheus.HistogramVec
	TokensPerSecond *prometheus.HistogramVec

	QueueDepth           prometheus.Gauge
	AdmissionActive      prometheus.Gauge
	BackendInflightToken *prometheus.GaugeVec
	BackendHealthy       *prometheus.GaugeVec

	CacheHitsTotal   *prometheus.CounterVec
	CacheMissesTotal *prometheus.CounterVec

	BreakerState   *prometheus.GaugeVec
	RetriesTotal   *prometheus.CounterVec
	UpstreamErrors *prometheus.CounterVec

	RateLimitRejections *prometheus.CounterVec
}

// latencyBuckets span the range that matters for inference: a cache hit lands
// near 1ms, a cold prefill in the hundreds of milliseconds, and a long
// generation in tens of seconds. Prometheus's default buckets top out at 10s,
// which would collapse every slow generation into +Inf and make p99 unusable.
var latencyBuckets = []float64{
	0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5,
	1, 2.5, 5, 10, 20, 30, 60, 120,
}

// throughputBuckets cover plausible decode rates in tokens/second.
var throughputBuckets = []float64{1, 5, 10, 25, 50, 100, 200, 400, 800, 1600}

// NewMetrics builds and registers all instruments.
func NewMetrics() *Metrics {
	reg := prometheus.NewRegistry()

	m := &Metrics{
		registry: reg,

		RequestsTotal: prometheus.NewCounterVec(prometheus.CounterOpts{
			Name: "waypoint_requests_total",
			Help: "Chat completion requests by outcome, serving backend, routing policy and cache result.",
		}, []string{"status", "backend", "policy", "cache"}),

		TTFTSeconds: prometheus.NewHistogramVec(prometheus.HistogramOpts{
			Name:    "waypoint_ttft_seconds",
			Help:    "Time to first content token, the latency users actually perceive.",
			Buckets: latencyBuckets,
		}, []string{"backend", "policy", "cache"}),

		DurationSeconds: prometheus.NewHistogramVec(prometheus.HistogramOpts{
			Name:    "waypoint_request_duration_seconds",
			Help:    "End-to-end request duration including generation.",
			Buckets: latencyBuckets,
		}, []string{"status", "backend", "policy"}),

		TokensPerSecond: prometheus.NewHistogramVec(prometheus.HistogramOpts{
			Name:    "waypoint_tokens_per_second",
			Help:    "Output throughput per request, which degrades as backend batches fill.",
			Buckets: throughputBuckets,
		}, []string{"backend"}),

		QueueDepth: prometheus.NewGauge(prometheus.GaugeOpts{
			Name: "waypoint_queue_depth",
			Help: "Requests waiting for an admission slot.",
		}),

		AdmissionActive: prometheus.NewGauge(prometheus.GaugeOpts{
			Name: "waypoint_admission_active",
			Help: "Requests currently dispatched.",
		}),

		BackendInflightToken: prometheus.NewGaugeVec(prometheus.GaugeOpts{
			Name: "waypoint_backend_inflight_tokens",
			Help: "Estimated tokens in flight per backend, the gateway's load signal.",
		}, []string{"backend"}),

		BackendHealthy: prometheus.NewGaugeVec(prometheus.GaugeOpts{
			Name: "waypoint_backend_healthy",
			Help: "1 when a backend is eligible for routing, 0 otherwise.",
		}, []string{"backend"}),

		CacheHitsTotal: prometheus.NewCounterVec(prometheus.CounterOpts{
			Name: "waypoint_cache_hits_total",
			Help: "Cache hits by layer.",
		}, []string{"type"}),

		CacheMissesTotal: prometheus.NewCounterVec(prometheus.CounterOpts{
			Name: "waypoint_cache_misses_total",
			Help: "Cache misses by layer.",
		}, []string{"type"}),

		BreakerState: prometheus.NewGaugeVec(prometheus.GaugeOpts{
			Name: "waypoint_breaker_state",
			Help: "Circuit breaker state per backend (0=closed, 1=half_open, 2=open).",
		}, []string{"backend"}),

		RetriesTotal: prometheus.NewCounterVec(prometheus.CounterOpts{
			Name: "waypoint_retries_total",
			Help: "Retry attempts by reason.",
		}, []string{"reason"}),

		UpstreamErrors: prometheus.NewCounterVec(prometheus.CounterOpts{
			Name: "waypoint_upstream_errors_total",
			Help: "Backend failures by backend and kind.",
		}, []string{"backend", "kind"}),

		RateLimitRejections: prometheus.NewCounterVec(prometheus.CounterOpts{
			Name: "waypoint_ratelimit_rejections_total",
			Help: "Requests rejected by rate limiting, by the resource that bound.",
		}, []string{"resource"}),
	}

	reg.MustRegister(
		m.RequestsTotal, m.TTFTSeconds, m.DurationSeconds, m.TokensPerSecond,
		m.QueueDepth, m.AdmissionActive, m.BackendInflightToken, m.BackendHealthy,
		m.CacheHitsTotal, m.CacheMissesTotal,
		m.BreakerState, m.RetriesTotal, m.UpstreamErrors,
		m.RateLimitRejections,
	)

	// Go runtime and process metrics make GC pauses and FD exhaustion visible,
	// both of which show up as latency the application metrics cannot explain.
	reg.MustRegister(
		collectors.NewGoCollector(),
		collectors.NewProcessCollector(collectors.ProcessCollectorOpts{}),
	)

	return m
}

// InitSeries pre-creates the series whose label values are known at startup.
//
// Prometheus does not export a metric family that has never been observed, so
// without this a fresh gateway reports *nothing* for retries, cache misses or
// per-backend load until the first one happens. Dashboards then show "No data"
// rather than zero, and an alert on rate(...) cannot distinguish "no traffic"
// from "the metric does not exist" — which is exactly the ambiguity you do not
// want at 3am.
func (m *Metrics) InitSeries(backends []string) {
	for _, name := range backends {
		m.BackendInflightToken.WithLabelValues(name).Set(0)
		m.BackendHealthy.WithLabelValues(name).Set(0)
		m.BreakerState.WithLabelValues(name).Set(0)
	}

	for _, layer := range []string{"exact", "semantic", "miss"} {
		m.CacheHitsTotal.WithLabelValues(layer).Add(0)
		m.CacheMissesTotal.WithLabelValues(layer).Add(0)
	}

	for _, reason := range []string{
		"ttft_timeout", "unreachable", "server_error",
		"upstream_error", "budget_exhausted", "not_retryable",
	} {
		m.RetriesTotal.WithLabelValues(reason).Add(0)
	}

	for _, resource := range []string{"requests", "tokens", "redis_unavailable"} {
		m.RateLimitRejections.WithLabelValues(resource).Add(0)
	}

	m.QueueDepth.Set(0)
	m.AdmissionActive.Set(0)
}

// Registry exposes the registry for the /metrics handler.
func (m *Metrics) Registry() *prometheus.Registry { return m.registry }

// Handler returns the Prometheus scrape handler.
func (m *Metrics) Handler() http.Handler {
	return promhttp.HandlerFor(m.registry, promhttp.HandlerOpts{
		// A broken collector should not take down the scrape endpoint.
		ErrorHandling: promhttp.ContinueOnError,
	})
}

// BreakerStateValue maps a breaker state name to its numeric gauge value.
//
// Gauges must be numeric, but breaker state is categorical. Encoding it as an
// ordered scale (closed < half_open < open) rather than arbitrary codes means
// `max_over_time` and alert thresholds behave sensibly.
func BreakerStateValue(state string) float64 {
	switch state {
	case "closed":
		return 0
	case "half_open":
		return 1
	case "open":
		return 2
	default:
		return -1
	}
}
