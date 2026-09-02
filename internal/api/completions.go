package api

import (
	"encoding/json"
	"errors"
	"io"
	"net/http"

	"github.com/aarnavnaga/waypoint/internal/admission"
	"github.com/aarnavnaga/waypoint/internal/openai"
	"github.com/aarnavnaga/waypoint/internal/router"
)

// maxRequestBytes bounds a request body. Prompts are large but not unbounded,
// and without a cap a single client can drive the gateway out of memory.
const maxRequestBytes = 8 << 20 // 8 MiB

// handleChatCompletions is the gateway's main request path:
// parse → admit → route → reserve → dispatch → release.
func (s *Server) handleChatCompletions(w http.ResponseWriter, r *http.Request) {
	ctx := r.Context()

	req, ok := s.parseChatRequest(w, r)
	if !ok {
		return
	}

	// Rate limiting runs first, before admission and routing. A request that is
	// over quota should cost the gateway as little as possible — admitting it
	// into the queue would let an over-quota tenant occupy capacity that paying
	// traffic needs.
	estTokens := s.router.EstimateTokens(req)
	if !s.checkRateLimit(w, r, estTokens) {
		return
	}

	// Admission control runs before routing: if the gateway cannot serve the
	// request at all, there is no point choosing who would have served it.
	ticket, err := s.admission.Acquire(ctx, admission.ClassInteractive)
	if err != nil {
		if errors.Is(err, admission.ErrQueueFull) {
			w.Header().Set("Retry-After", "1")
			writeAPIError(w, http.StatusServiceUnavailable,
				"gateway is at capacity, retry shortly",
				openai.ErrTypeOverloaded, "queue_full")
			return
		}
		// Context cancelled while queued: the client is already gone.
		return
	}
	defer ticket.Release()

	routeReq := s.router.BuildRequest(req)

	if len(s.pool.Available(req.Model)) == 0 {
		s.writeNoBackendError(w, req.Model)
		return
	}

	body, err := json.Marshal(req)
	if err != nil {
		writeAPIError(w, http.StatusInternalServerError,
			"failed to encode upstream request", openai.ErrTypeServer, "")
		return
	}

	// Every request deposits retry budget, whether or not it ends up retrying.
	// The budget is a fraction of *total* traffic, so deposits must track all
	// requests rather than only failing ones.
	if s.budget != nil {
		s.budget.Deposit()
	}

	result, err := s.dispatcher.Do(ctx, w, req, routeReq, body, passthroughHeaders(r))
	if err != nil {
		started := result != nil && result.Outcome != nil && result.Outcome.ResponseStarted
		s.logger.Warn("dispatch failed",
			"model", req.Model,
			"attempts", result.Attempts,
			"tried", result.Tried,
			"response_started", started,
			"error", err,
		)

		// Once any byte has reached the client, the response is committed:
		// writing an error envelope now would corrupt the stream the client is
		// already parsing.
		if started {
			return
		}
		if ctx.Err() != nil {
			return // client disconnected
		}

		status, typ := errorStatus(err)
		writeAPIError(w, status, err.Error(), typ, "")
		return
	}

	s.logger.Debug("dispatch complete",
		"backend", result.Outcome.Backend.Name,
		"model", req.Model,
		"attempts", result.Attempts,
		"ttft_ms", result.Outcome.TTFT.Milliseconds(),
		"completion_tokens", result.Outcome.CompletionTokens,
		"duration_ms", result.Outcome.Duration.Milliseconds(),
	)
}

// parseChatRequest decodes and validates the request body, writing an
// OpenAI-shaped error and returning false if it is unusable.
func (s *Server) parseChatRequest(w http.ResponseWriter, r *http.Request) (*openai.ChatRequest, bool) {
	limited := http.MaxBytesReader(w, r.Body, maxRequestBytes)
	defer func() { _ = r.Body.Close() }()

	payload, err := io.ReadAll(limited)
	if err != nil {
		var maxErr *http.MaxBytesError
		if errors.As(err, &maxErr) {
			writeAPIError(w, http.StatusRequestEntityTooLarge,
				"request body exceeds the size limit",
				openai.ErrTypeInvalidRequest, "payload_too_large")
			return nil, false
		}
		writeAPIError(w, http.StatusBadRequest, "failed to read request body",
			openai.ErrTypeInvalidRequest, "")
		return nil, false
	}

	var req openai.ChatRequest
	if err := json.Unmarshal(payload, &req); err != nil {
		writeAPIError(w, http.StatusBadRequest, "invalid JSON: "+err.Error(),
			openai.ErrTypeInvalidRequest, "")
		return nil, false
	}
	if err := req.Validate(); err != nil {
		writeAPIError(w, http.StatusBadRequest, err.Error(),
			openai.ErrTypeInvalidRequest, "")
		return nil, false
	}
	return &req, true
}

// writeNoBackendError reports that nothing can serve the request,
// distinguishing "wrong model" from "fleet is down" so the caller knows whether
// retrying will help.
func (s *Server) writeNoBackendError(w http.ResponseWriter, model string) {
	if len(s.pool.Healthy()) == 0 {
		w.Header().Set("Retry-After", "5")
		writeAPIError(w, http.StatusServiceUnavailable,
			"no healthy backend is available",
			openai.ErrTypeOverloaded, "no_healthy_backend")
		return
	}

	// Backends are up but none serves this model, or all are saturated.
	for _, b := range s.pool.Healthy() {
		if b.ServesModel(model) {
			w.Header().Set("Retry-After", "1")
			writeAPIError(w, http.StatusServiceUnavailable,
				"all backends serving this model are at capacity",
				openai.ErrTypeOverloaded, "backends_saturated")
			return
		}
	}

	writeAPIError(w, http.StatusNotFound,
		"model "+model+" is not served by any backend",
		openai.ErrTypeInvalidRequest, "model_not_found")
}

// passthroughHeaders selects request headers that should reach the backend.
//
// Authorization is deliberately *not* forwarded: the client's key authenticates
// it to Waypoint, not to the model servers, which sit on a trusted network.
// Forwarding it would leak tenant credentials into the backend fleet.
var passthroughHeaderNames = []string{
	"traceparent",  // W3C trace context
	"tracestate",   // W3C trace context
	"X-Request-Id", // caller-supplied correlation
	"X-Waypoint-Cache",
}

func passthroughHeaders(r *http.Request) http.Header {
	out := make(http.Header, len(passthroughHeaderNames))
	for _, name := range passthroughHeaderNames {
		if v := r.Header.Get(name); v != "" {
			out.Set(name, v)
		}
	}
	return out
}

// compile-time assertion that the router error is handled by errorStatus.
var _ = router.ErrNoBackends
