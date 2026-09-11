package telemetry

import (
	"context"
	"log/slog"
	"os"
	"time"
)

// NewLogger builds the structured JSON logger used throughout the gateway.
func NewLogger(level string) *slog.Logger {
	var lvl slog.Level
	if err := lvl.UnmarshalText([]byte(level)); err != nil {
		lvl = slog.LevelInfo
	}

	handler := slog.NewJSONHandler(os.Stdout, &slog.HandlerOptions{
		Level: lvl,
		ReplaceAttr: func(_ []string, a slog.Attr) slog.Attr {
			// RFC3339 with nanoseconds, in UTC, so logs from different hosts
			// sort correctly against each other and against trace timestamps.
			if a.Key == slog.TimeKey {
				a.Value = slog.StringValue(time.Now().UTC().Format(time.RFC3339Nano))
			}
			return a
		},
	})

	return slog.New(&traceHandler{Handler: handler})
}

// traceHandler injects trace and span IDs into every record that has them.
//
// This is the join key between the three signals. A Grafana panel shows p99
// latency spiking; a log line for a slow request carries its trace_id; pasting
// that into Jaeger shows exactly which stage was slow. Without the correlation
// ID the three systems are three separate investigations.
type traceHandler struct {
	slog.Handler
}

func (h *traceHandler) Handle(ctx context.Context, record slog.Record) error {
	if traceID := TraceIDFromContext(ctx); traceID != "" {
		record.AddAttrs(slog.String("trace_id", traceID))
		if spanID := SpanIDFromContext(ctx); spanID != "" {
			record.AddAttrs(slog.String("span_id", spanID))
		}
	}
	return h.Handler.Handle(ctx, record)
}

func (h *traceHandler) WithAttrs(attrs []slog.Attr) slog.Handler {
	return &traceHandler{Handler: h.Handler.WithAttrs(attrs)}
}

func (h *traceHandler) WithGroup(name string) slog.Handler {
	return &traceHandler{Handler: h.Handler.WithGroup(name)}
}
