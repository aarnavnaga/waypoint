package telemetry

import (
	"context"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/prometheus/client_golang/prometheus/testutil"
)

func TestMetricsRegisterAndExport(t *testing.T) {
	m := NewMetrics()
	m.InitSeries([]string{"mock-1", "mock-2"})

	m.RequestsTotal.WithLabelValues("200", "mock-1", "round_robin", "miss").Inc()
	m.TTFTSeconds.WithLabelValues("mock-1", "round_robin", "miss").Observe(0.25)
	m.DurationSeconds.WithLabelValues("200", "mock-1", "round_robin").Observe(1.5)
	m.TokensPerSecond.WithLabelValues("mock-1").Observe(120)
	m.CacheHitsTotal.WithLabelValues("exact").Inc()
	m.RateLimitRejections.WithLabelValues("requests").Inc()
	m.BreakerState.WithLabelValues("mock-1").Set(2)

	rec := httptest.NewRecorder()
	m.Handler().ServeHTTP(rec, httptest.NewRequest(http.MethodGet, "/metrics", nil))

	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200", rec.Code)
	}

	body := rec.Body.String()
	// Every metric SPEC §6.7 names must be present in the exposition.
	required := []string{
		"waypoint_requests_total",
		"waypoint_ttft_seconds",
		"waypoint_request_duration_seconds",
		"waypoint_tokens_per_second",
		"waypoint_queue_depth",
		"waypoint_backend_inflight_tokens",
		"waypoint_cache_hits_total",
		"waypoint_cache_misses_total",
		"waypoint_breaker_state",
		"waypoint_retries_total",
		"waypoint_ratelimit_rejections_total",
	}
	for _, name := range required {
		if !strings.Contains(body, name) {
			t.Errorf("metric %s missing from the exposition", name)
		}
	}
}

// A fresh gateway must already export the series dashboards graph, so panels
// show zero instead of "No data" before the first request.
func TestInitSeriesPreCreatesFamilies(t *testing.T) {
	m := NewMetrics()
	m.InitSeries([]string{"mock-1"})

	rec := httptest.NewRecorder()
	m.Handler().ServeHTTP(rec, httptest.NewRequest(http.MethodGet, "/metrics", nil))
	body := rec.Body.String()

	for _, name := range []string{
		"waypoint_backend_inflight_tokens",
		"waypoint_cache_misses_total",
		"waypoint_retries_total",
		"waypoint_ratelimit_rejections_total",
		"waypoint_breaker_state",
	} {
		if !strings.Contains(body, name) {
			t.Errorf("%s is absent before any traffic; dashboards would show No data", name)
		}
	}
}

func TestMetricsCountersIncrement(t *testing.T) {
	m := NewMetrics()

	for i := 0; i < 3; i++ {
		m.RequestsTotal.WithLabelValues("200", "mock-1", "round_robin", "miss").Inc()
	}
	if got := testutil.ToFloat64(m.RequestsTotal.WithLabelValues("200", "mock-1", "round_robin", "miss")); got != 3 {
		t.Errorf("counter = %v, want 3", got)
	}
}

// Latency buckets must extend past Prometheus's 10s default, or every long
// generation collapses into +Inf and p99 becomes unusable.
func TestLatencyBucketsCoverLongGenerations(t *testing.T) {
	last := latencyBuckets[len(latencyBuckets)-1]
	if last < 60 {
		t.Errorf("largest latency bucket is %vs; long generations would be unmeasurable", last)
	}
	if latencyBuckets[0] > 0.005 {
		t.Errorf("smallest latency bucket is %vs; cache hits would be unmeasurable", latencyBuckets[0])
	}
}

func TestBreakerStateValue(t *testing.T) {
	tests := map[string]float64{
		"closed":    0,
		"half_open": 1,
		"open":      2,
		"nonsense":  -1,
	}
	for state, want := range tests {
		if got := BreakerStateValue(state); got != want {
			t.Errorf("BreakerStateValue(%q) = %v, want %v", state, got, want)
		}
	}

	// The encoding must be ordered so alert thresholds behave sensibly.
	if BreakerStateValue("closed") >= BreakerStateValue("half_open") ||
		BreakerStateValue("half_open") >= BreakerStateValue("open") {
		t.Error("breaker state encoding is not monotonically ordered")
	}
}

// Tracing must be optional: with no endpoint the gateway runs on a no-op tracer
// rather than failing to start.
func TestInitTracingWithoutEndpoint(t *testing.T) {
	tracer, shutdown, err := InitTracing(context.Background(), TracingOptions{
		ServiceName: "test", Logger: quietLogger(),
	})
	if err != nil {
		t.Fatalf("InitTracing: %v", err)
	}
	if tracer == nil {
		t.Fatal("tracer is nil")
	}

	_, span := tracer.Start(context.Background(), "noop-span")
	span.End()

	if err := shutdown(context.Background()); err != nil {
		t.Errorf("shutdown: %v", err)
	}
}

func TestTraceIDFromContextWhenUntraced(t *testing.T) {
	if got := TraceIDFromContext(context.Background()); got != "" {
		t.Errorf("TraceIDFromContext = %q, want empty for an untraced context", got)
	}
	if got := SpanIDFromContext(context.Background()); got != "" {
		t.Errorf("SpanIDFromContext = %q, want empty", got)
	}
}

func TestLoggerEmitsJSON(t *testing.T) {
	logger := NewLogger("debug")
	if logger == nil {
		t.Fatal("NewLogger returned nil")
	}
	// Must not panic with or without a span in context.
	logger.InfoContext(context.Background(), "hello", "key", "value")
}

func TestLoggerLevelParsing(t *testing.T) {
	// An unparseable level must fall back to info rather than failing.
	if l := NewLogger("not-a-level"); l == nil {
		t.Fatal("NewLogger returned nil for an invalid level")
	}
	if !NewLogger("not-a-level").Enabled(context.Background(), slog.LevelInfo) {
		t.Error("invalid level did not fall back to info")
	}
	if NewLogger("error").Enabled(context.Background(), slog.LevelInfo) {
		t.Error("error level logger is emitting info records")
	}
}

func TestSpanNamesCoverRequestLifecycle(t *testing.T) {
	// SPEC §5 lists the stages a trace should show.
	stages := []string{SpanRequest, SpanRateLimit, SpanCache, SpanAdmission, SpanDispatch}
	seen := map[string]bool{}
	for _, s := range stages {
		if s == "" {
			t.Error("empty span name constant")
		}
		if seen[s] {
			t.Errorf("duplicate span name %q", s)
		}
		seen[s] = true
	}
}

// A semconv version that disagrees with resource.Default()'s schema makes
// resource.Merge fail, which silently disables tracing: the gateway starts
// normally and simply never exports a span. This asserts the versions agree.
func TestInitTracingBuildsResource(t *testing.T) {
	// Point at an address nothing is listening on. The OTLP gRPC exporter
	// connects lazily, so construction should still succeed — the failure this
	// guards against is the resource schema conflict, not connectivity.
	tracer, shutdown, err := InitTracing(context.Background(), TracingOptions{
		ServiceName: "waypoint-test",
		Version:     "test",
		Endpoint:    "127.0.0.1:4317",
		SampleRatio: 1,
		Logger:      quietLogger(),
	})
	if err != nil {
		t.Fatalf("InitTracing failed to build a resource: %v", err)
	}
	defer func() { _ = shutdown(context.Background()) }()

	ctx, span := tracer.Start(context.Background(), SpanRequest)
	if got := TraceIDFromContext(ctx); got == "" {
		t.Error("a started span produced no trace ID")
	}
	if got := SpanIDFromContext(ctx); got == "" {
		t.Error("a started span produced no span ID")
	}
	span.End()
}
