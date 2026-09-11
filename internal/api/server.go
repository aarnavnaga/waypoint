// Package api implements the gateway's OpenAI-compatible HTTP surface:
// request parsing, authentication, admission control, dispatch and streaming.
package api

import (
	"context"
	"encoding/json"
	"errors"
	"log/slog"
	"net/http"
	"strings"
	"time"

	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/codes"
	"go.opentelemetry.io/otel/propagation"
	"go.opentelemetry.io/otel/trace"
	"go.opentelemetry.io/otel/trace/noop"

	"github.com/aarnavnaga/waypoint/internal/admission"
	"github.com/aarnavnaga/waypoint/internal/backend"
	"github.com/aarnavnaga/waypoint/internal/cache"
	"github.com/aarnavnaga/waypoint/internal/config"
	"github.com/aarnavnaga/waypoint/internal/openai"
	"github.com/aarnavnaga/waypoint/internal/ratelimit"
	"github.com/aarnavnaga/waypoint/internal/resilience"
	"github.com/aarnavnaga/waypoint/internal/router"
	"github.com/aarnavnaga/waypoint/internal/telemetry"
	"github.com/aarnavnaga/waypoint/internal/tokenizer"
)

// Readiness reports whether dependencies the gateway needs are usable.
// Implemented by the Redis client in M5/M6; nil means "no dependency".
type Readiness interface {
	Ready(ctx context.Context) error
}

// Options configures a Server.
type Options struct {
	Config     config.Config
	Pool       *backend.Pool
	Router     *router.Router
	Proxy      *Proxy
	Dispatcher *Dispatcher
	Breakers   *Breakers
	Budget     *resilience.RetryBudget
	Admission  *admission.Controller
	Limiter    ratelimit.Limiter
	Cache      *cache.Cache
	Metrics    *telemetry.Metrics
	Tracer     trace.Tracer
	Estimator  tokenizer.Estimator
	Logger     *slog.Logger
	// Redis is optional; when set, /readyz requires it to be reachable.
	Redis Readiness
	// Version is reported by /healthz.
	Version string
}

// Server is the gateway's HTTP handler set.
type Server struct {
	cfg        config.Config
	pool       *backend.Pool
	router     *router.Router
	proxy      *Proxy
	dispatcher *Dispatcher
	breakers   *Breakers
	budget     *resilience.RetryBudget
	admission  *admission.Controller
	limiter    ratelimit.Limiter
	cache      *cache.Cache
	metrics    *telemetry.Metrics
	tracer     trace.Tracer
	estimator  tokenizer.Estimator
	logger     *slog.Logger
	redis      Readiness
	version    string

	// apiKeys is a set for O(1) lookup; empty disables authentication.
	apiKeys map[string]struct{}
}

// NewServer builds a Server.
func NewServer(opts Options) *Server {
	logger := opts.Logger
	if logger == nil {
		logger = slog.Default()
	}
	est := opts.Estimator
	if est == nil {
		est = tokenizer.NewCharHeuristic()
	}

	// A nil Dispatcher would panic on the first request. Build a single-attempt
	// one so a partially configured server degrades to "no retries" rather than
	// to a crash loop.
	dispatcher := opts.Dispatcher
	if dispatcher == nil {
		dispatcher = NewDispatcher(DispatcherOptions{
			Pool:        opts.Pool,
			Router:      opts.Router,
			Proxy:       opts.Proxy,
			Breakers:    opts.Breakers,
			MaxAttempts: 1,
			Logger:      logger,
		})
	}

	// A no-op tracer keeps call sites free of nil checks when tracing is off.
	tracer := opts.Tracer
	if tracer == nil {
		tracer = noop.NewTracerProvider().Tracer(telemetry.TracerName)
	}

	// Rate limiting defaults to permissive so a partially configured server
	// still serves rather than denying everything.
	limiter := opts.Limiter
	if limiter == nil {
		limiter = ratelimit.AllowAll{}
	}

	keys := make(map[string]struct{}, len(opts.Config.Auth.APIKeys))
	for _, k := range opts.Config.Auth.APIKeys {
		keys[k] = struct{}{}
	}

	return &Server{
		cfg:        opts.Config,
		pool:       opts.Pool,
		router:     opts.Router,
		proxy:      opts.Proxy,
		dispatcher: dispatcher,
		breakers:   opts.Breakers,
		budget:     opts.Budget,
		admission:  opts.Admission,
		limiter:    limiter,
		cache:      opts.Cache,
		metrics:    opts.Metrics,
		tracer:     tracer,
		estimator:  est,
		logger:     logger,
		redis:      opts.Redis,
		version:    opts.Version,
		apiKeys:    keys,
	}
}

// Handler returns the gateway's routes with middleware applied.
func (s *Server) Handler() http.Handler {
	mux := http.NewServeMux()

	mux.Handle("POST /v1/chat/completions", s.authenticated(http.HandlerFunc(s.handleChatCompletions)))
	mux.Handle("GET /v1/models", s.authenticated(http.HandlerFunc(s.handleModels)))

	if s.metrics != nil {
		mux.Handle("GET /metrics", s.metrics.Handler())
	}

	mux.HandleFunc("GET /healthz", s.handleHealthz)
	mux.HandleFunc("GET /readyz", s.handleReadyz)
	mux.HandleFunc("GET /admin/backends", s.handleAdminBackends)
	mux.HandleFunc("GET /admin/routing", s.handleGetRouting)
	mux.HandleFunc("POST /admin/routing", s.handleSetRouting)

	// Order matters: tracing is outermost (after panic recovery) so every inner
	// layer — including request logging — runs inside the span and can attach
	// the trace ID to its output.
	return s.recoverPanic(s.traced(s.logRequests(mux)))
}

// ---------- middleware ----------

// authenticated enforces API-key auth when any keys are configured.
func (s *Server) authenticated(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if len(s.apiKeys) == 0 {
			next.ServeHTTP(w, r)
			return
		}

		key := bearerToken(r)
		if key == "" {
			writeAPIError(w, http.StatusUnauthorized,
				"missing API key: supply it as 'Authorization: Bearer <key>'",
				openai.ErrTypeAuthentication, "missing_api_key")
			return
		}
		if _, ok := s.apiKeys[key]; !ok {
			writeAPIError(w, http.StatusUnauthorized,
				"invalid API key", openai.ErrTypeAuthentication, "invalid_api_key")
			return
		}

		next.ServeHTTP(w, r.WithContext(withAPIKey(r.Context(), key)))
	})
}

// traced starts the root span for a request and puts it in the context.
//
// This runs *outside* the request logger on purpose. If the span were started
// inside the handler, the logging middleware would execute with the original
// context and every access log line would be missing its trace ID — which
// defeats the point, since the access log is exactly where you look up the
// trace for a slow request.
func (s *Server) traced(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// Health and metrics endpoints are scraped constantly and would bury
		// real traffic in the trace backend.
		switch r.URL.Path {
		case "/healthz", "/readyz", "/metrics":
			next.ServeHTTP(w, r)
			return
		}

		// Continue an upstream trace when the caller supplied one, so the
		// gateway appears as a span inside the client's trace rather than
		// starting a disconnected one.
		ctx := otel.GetTextMapPropagator().Extract(r.Context(),
			propagation.HeaderCarrier(r.Header))

		ctx, span := s.tracer.Start(ctx, telemetry.SpanRequest,
			trace.WithSpanKind(trace.SpanKindServer),
			trace.WithAttributes(
				attribute.String("http.request.method", r.Method),
				attribute.String("url.path", r.URL.Path),
			),
		)
		defer span.End()

		rec := &statusRecorder{ResponseWriter: w, status: http.StatusOK}
		next.ServeHTTP(rec, r.WithContext(ctx))

		span.SetAttributes(attribute.Int("http.response.status_code", rec.status))
		if rec.status >= 500 {
			span.SetStatus(codes.Error, http.StatusText(rec.status))
		}
	})
}

// logRequests emits one structured line per request.
func (s *Server) logRequests(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		start := time.Now()
		rec := &statusRecorder{ResponseWriter: w, status: http.StatusOK}

		next.ServeHTTP(rec, r)

		// Health probes would otherwise dominate the logs.
		if r.URL.Path == "/healthz" || r.URL.Path == "/readyz" {
			return
		}
		// InfoContext, not Info: the trace ID is pulled from the context by
		// the logging handler, so a plain Info() would drop it.
		s.logger.InfoContext(r.Context(), "request",
			"method", r.Method,
			"path", r.URL.Path,
			"status", rec.status,
			"duration_ms", time.Since(start).Milliseconds(),
		)
	})
}

// recoverPanic keeps one bad request from taking down the process.
func (s *Server) recoverPanic(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		defer func() {
			if rec := recover(); rec != nil {
				s.logger.Error("panic serving request",
					"path", r.URL.Path, "panic", rec)
				// If the response already started there is nothing safe to add.
				writeAPIError(w, http.StatusInternalServerError,
					"internal error", openai.ErrTypeServer, "")
			}
		}()
		next.ServeHTTP(w, r)
	})
}

// ---------- handlers ----------

func (s *Server) handleHealthz(w http.ResponseWriter, _ *http.Request) {
	writeJSON(w, http.StatusOK, map[string]string{
		"status":  "ok",
		"version": s.version,
	})
}

// handleReadyz reports whether the gateway can serve traffic right now:
// at least one healthy backend, and Redis reachable when configured.
//
// Liveness and readiness are deliberately different. A gateway with zero healthy
// backends is alive (restarting it would not help) but not ready (it should be
// taken out of the load balancer until a backend returns).
func (s *Server) handleReadyz(w http.ResponseWriter, r *http.Request) {
	healthy := s.pool.Healthy()

	body := map[string]any{
		"healthy_backends": len(healthy),
		"total_backends":   s.pool.Len(),
	}

	if len(healthy) == 0 {
		body["status"] = "no healthy backends"
		writeJSON(w, http.StatusServiceUnavailable, body)
		return
	}

	if s.redis != nil {
		ctx, cancel := context.WithTimeout(r.Context(), s.cfg.Redis.Timeout.Duration)
		defer cancel()
		if err := s.redis.Ready(ctx); err != nil {
			body["status"] = "redis unavailable"
			body["error"] = err.Error()
			// Redis being down does not stop the gateway serving when limits
			// fail open, so this is reported but not fatal to readiness.
			if s.cfg.RateLimit.Enabled && !s.cfg.RateLimit.FailOpen {
				writeJSON(w, http.StatusServiceUnavailable, body)
				return
			}
			body["degraded"] = true
		}
	}

	if _, ok := body["status"]; !ok {
		body["status"] = "ready"
	}
	writeJSON(w, http.StatusOK, body)
}

func (s *Server) handleModels(w http.ResponseWriter, _ *http.Request) {
	models := backend.UnionModels(s.pool.Healthy())

	list := openai.ModelList{Object: "list", Data: make([]openai.Model, 0, len(models))}
	for _, m := range models {
		list.Data = append(list.Data, openai.Model{
			ID:      m,
			Object:  "model",
			Created: time.Now().Unix(),
			OwnedBy: "waypoint",
		})
	}
	writeJSON(w, http.StatusOK, list)
}

func (s *Server) handleAdminBackends(w http.ResponseWriter, _ *http.Request) {
	stats := make([]backend.Stats, 0, s.pool.Len())
	for _, b := range s.pool.All() {
		stats = append(stats, b.Stats())
	}

	body := map[string]any{
		"policy":    s.router.PolicyName(),
		"backends":  stats,
		"admission": s.admission.Stats(),
	}
	if s.breakers != nil {
		body["breakers"] = s.breakers.Stats()
	}
	if s.budget != nil {
		body["retry_budget"] = s.budget.Stats()
	}
	if s.cache != nil {
		body["cache"] = s.cache.Stats()
	}
	writeJSON(w, http.StatusOK, body)
}

func (s *Server) handleGetRouting(w http.ResponseWriter, _ *http.Request) {
	writeJSON(w, http.StatusOK, map[string]any{
		"policy":             s.router.PolicyName(),
		"available_policies": config.ValidPolicies,
	})
}

// handleSetRouting switches policy at runtime, which the M8 benchmark uses to
// compare policies without restarting the stack (a restart would also reset the
// backends' prefix caches and invalidate the comparison).
func (s *Server) handleSetRouting(w http.ResponseWriter, r *http.Request) {
	var body struct {
		Policy string `json:"policy"`
	}
	if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
		writeAPIError(w, http.StatusBadRequest, "invalid JSON: "+err.Error(),
			openai.ErrTypeInvalidRequest, "")
		return
	}

	if err := s.router.SetPolicy(body.Policy); err != nil {
		writeAPIError(w, http.StatusBadRequest, err.Error(),
			openai.ErrTypeInvalidRequest, "")
		return
	}

	s.logger.Info("routing policy changed", "policy", body.Policy)
	writeJSON(w, http.StatusOK, map[string]string{"policy": s.router.PolicyName()})
}

// ---------- helpers ----------

// bearerToken extracts a key from the Authorization header, also accepting the
// x-api-key form some clients use.
func bearerToken(r *http.Request) string {
	if auth := r.Header.Get("Authorization"); auth != "" {
		const prefix = "Bearer "
		if len(auth) > len(prefix) && strings.EqualFold(auth[:len(prefix)], prefix) {
			return strings.TrimSpace(auth[len(prefix):])
		}
		return ""
	}
	return strings.TrimSpace(r.Header.Get("X-API-Key"))
}

type ctxKey int

const ctxKeyAPIKey ctxKey = iota

func withAPIKey(ctx context.Context, key string) context.Context {
	return context.WithValue(ctx, ctxKeyAPIKey, key)
}

// APIKeyFromContext returns the authenticated key, if any.
func APIKeyFromContext(ctx context.Context) string {
	key, _ := ctx.Value(ctxKeyAPIKey).(string)
	return key
}

// statusRecorder captures the status code for logging.
type statusRecorder struct {
	http.ResponseWriter
	status  int
	written bool
}

func (r *statusRecorder) WriteHeader(code int) {
	if !r.written {
		r.status = code
		r.written = true
	}
	r.ResponseWriter.WriteHeader(code)
}

func (r *statusRecorder) Write(b []byte) (int, error) {
	r.written = true
	return r.ResponseWriter.Write(b)
}

// Flush forwards to the underlying writer so streaming still works through the
// middleware chain. Without this, wrapping the ResponseWriter silently disables
// SSE, because the handler's type assertion to http.Flusher fails.
func (r *statusRecorder) Flush() {
	if f, ok := r.ResponseWriter.(http.Flusher); ok {
		f.Flush()
	}
}

func writeJSON(w http.ResponseWriter, status int, body any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(body)
}

func writeAPIError(w http.ResponseWriter, status int, msg, typ, code string) {
	writeJSON(w, status, openai.NewError(msg, typ, code))
}

// errorStatus maps a dispatch error to an HTTP status and OpenAI error type.
func errorStatus(err error) (status int, typ string) {
	switch {
	case errors.Is(err, admission.ErrQueueFull):
		return http.StatusServiceUnavailable, openai.ErrTypeOverloaded
	case errors.Is(err, router.ErrNoBackends):
		return http.StatusServiceUnavailable, openai.ErrTypeOverloaded
	case errors.Is(err, ErrTTFTTimeout), errors.Is(err, context.DeadlineExceeded):
		return http.StatusGatewayTimeout, openai.ErrTypeServer
	case errors.Is(err, ErrUpstreamUnavailable), errors.Is(err, ErrUpstreamServerError):
		return http.StatusBadGateway, openai.ErrTypeServer
	default:
		return http.StatusInternalServerError, openai.ErrTypeServer
	}
}
