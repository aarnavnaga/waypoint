package telemetry

import (
	"context"
	"fmt"
	"log/slog"
	"time"

	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/exporters/otlp/otlptrace/otlptracegrpc"
	"go.opentelemetry.io/otel/propagation"
	"go.opentelemetry.io/otel/sdk/resource"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	semconv "go.opentelemetry.io/otel/semconv/v1.43.0"
	"go.opentelemetry.io/otel/trace"
	"go.opentelemetry.io/otel/trace/noop"
)

// TracerName namespaces spans produced by the gateway.
const TracerName = "github.com/aarnavnaga/waypoint"

// Span names for each stage of SPEC §5's request lifecycle.
const (
	SpanRequest    = "waypoint.request"
	SpanRateLimit  = "waypoint.ratelimit"
	SpanCache      = "waypoint.cache_lookup"
	SpanAdmission  = "waypoint.admission"
	SpanRouting    = "waypoint.routing"
	SpanDispatch   = "waypoint.dispatch"
	SpanCacheWrite = "waypoint.cache_write"
)

// TracingOptions configures the trace pipeline.
type TracingOptions struct {
	ServiceName string
	Version     string
	// Endpoint is an OTLP/gRPC collector address. Empty disables exporting.
	Endpoint string
	// SampleRatio in [0,1]; 1 traces everything.
	SampleRatio float64
	Logger      *slog.Logger
}

// Shutdown flushes and stops the trace pipeline.
type Shutdown func(context.Context) error

// InitTracing configures the global tracer provider.
//
// Tracing is strictly optional: if no endpoint is configured, or the collector
// is unreachable, the gateway runs with a no-op tracer. Observability must never
// be able to take down the thing it observes.
func InitTracing(ctx context.Context, opts TracingOptions) (trace.Tracer, Shutdown, error) {
	logger := opts.Logger
	if logger == nil {
		logger = slog.Default()
	}

	// Always install the W3C propagator, even without an exporter, so trace
	// context flows through to backends and a downstream collector can stitch
	// the trace together even when this hop is not exporting.
	otel.SetTextMapPropagator(propagation.NewCompositeTextMapPropagator(
		propagation.TraceContext{},
		propagation.Baggage{},
	))

	if opts.Endpoint == "" {
		logger.Info("tracing disabled (no OTLP endpoint configured)")
		return noop.NewTracerProvider().Tracer(TracerName), func(context.Context) error { return nil }, nil
	}

	exporter, err := otlptracegrpc.New(ctx,
		otlptracegrpc.WithEndpoint(opts.Endpoint),
		otlptracegrpc.WithInsecure(),
		otlptracegrpc.WithTimeout(5*time.Second),
	)
	if err != nil {
		return nil, nil, fmt.Errorf("telemetry: create OTLP exporter: %w", err)
	}

	// resource.Merge fails outright on conflicting schema URLs, so the semconv
	// package imported here must match the one resource.Default() was built
	// with. Getting this wrong is a silent loss of tracing: the gateway starts
	// fine and simply never exports a span.
	res, err := resource.Merge(resource.Default(), resource.NewWithAttributes(
		semconv.SchemaURL,
		semconv.ServiceName(opts.ServiceName),
		semconv.ServiceVersion(opts.Version),
	))
	if err != nil {
		return nil, nil, fmt.Errorf("telemetry: build resource (schema mismatch between semconv and resource.Default): %w", err)
	}

	ratio := opts.SampleRatio
	if ratio <= 0 {
		ratio = 1
	}

	provider := sdktrace.NewTracerProvider(
		// Batching keeps export off the request path; a synchronous exporter
		// would add collector latency to every request.
		sdktrace.WithBatcher(exporter,
			sdktrace.WithMaxQueueSize(2048),
			sdktrace.WithBatchTimeout(2*time.Second),
		),
		sdktrace.WithResource(res),
		// ParentBased keeps a trace whole: if an upstream service sampled a
		// request, this hop records it too, rather than producing traces with
		// holes in them.
		sdktrace.WithSampler(sdktrace.ParentBased(sdktrace.TraceIDRatioBased(ratio))),
	)
	otel.SetTracerProvider(provider)

	logger.Info("tracing enabled", "endpoint", opts.Endpoint, "sample_ratio", ratio)

	shutdown := func(ctx context.Context) error {
		shutdownCtx, cancel := context.WithTimeout(ctx, 5*time.Second)
		defer cancel()
		return provider.Shutdown(shutdownCtx)
	}
	return provider.Tracer(TracerName), shutdown, nil
}

// Attribute keys used on gateway spans.
var (
	AttrBackend     = attribute.Key("waypoint.backend")
	AttrPolicy      = attribute.Key("waypoint.policy")
	AttrModel       = attribute.Key("waypoint.model")
	AttrCache       = attribute.Key("waypoint.cache")
	AttrAttempt     = attribute.Key("waypoint.attempt")
	AttrEstTokens   = attribute.Key("waypoint.estimated_tokens")
	AttrTTFTMillis  = attribute.Key("waypoint.ttft_ms")
	AttrOutTokens   = attribute.Key("waypoint.completion_tokens")
	AttrStreaming   = attribute.Key("waypoint.streaming")
	AttrSimilarity  = attribute.Key("waypoint.cache_similarity")
	AttrBreakerOpen = attribute.Key("waypoint.breaker_open")
)

// TraceIDFromContext returns the current trace ID, or "" when untraced.
//
// This is what ties a log line to a trace: given a slow request in Grafana, you
// copy its trace ID out of the logs and open the exact trace in Jaeger.
func TraceIDFromContext(ctx context.Context) string {
	sc := trace.SpanContextFromContext(ctx)
	if !sc.IsValid() {
		return ""
	}
	return sc.TraceID().String()
}

// SpanIDFromContext returns the current span ID, or "" when untraced.
func SpanIDFromContext(ctx context.Context) string {
	sc := trace.SpanContextFromContext(ctx)
	if !sc.IsValid() {
		return ""
	}
	return sc.SpanID().String()
}
