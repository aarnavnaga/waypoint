package mocksim

import (
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"net/http"
	"strings"
	"time"

	"github.com/aarnavnaga/waypoint/internal/openai"
)

// Server exposes an Engine over an OpenAI-compatible HTTP API.
type Server struct {
	engine *Engine
	logger *slog.Logger
	now    func() time.Time
	// exit is called by /admin/crash; injectable so tests do not kill the runner.
	exit func(code int)
}

// NewServer wraps an Engine in HTTP handlers.
func NewServer(engine *Engine, logger *slog.Logger) *Server {
	if logger == nil {
		logger = slog.Default()
	}
	return &Server{
		engine: engine,
		logger: logger,
		now:    time.Now,
		exit:   func(int) {},
	}
}

// SetExitFunc overrides the process-exit behavior of /admin/crash.
func (s *Server) SetExitFunc(fn func(int)) { s.exit = fn }

// Handler returns the configured HTTP routes.
func (s *Server) Handler() http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("POST /v1/chat/completions", s.handleChatCompletions)
	mux.HandleFunc("GET /v1/models", s.handleModels)
	mux.HandleFunc("GET /healthz", s.handleHealth)
	mux.HandleFunc("GET /stats", s.handleStats)
	mux.HandleFunc("GET /admin/faults", s.handleGetFaults)
	mux.HandleFunc("POST /admin/faults", s.handleSetFaults)
	mux.HandleFunc("POST /admin/reset", s.handleReset)
	mux.HandleFunc("POST /admin/crash", s.handleCrash)
	return mux
}

func (s *Server) handleHealth(w http.ResponseWriter, _ *http.Request) {
	if s.engine.Faults().ShouldHang() {
		// A hung server should fail its health check too, otherwise the
		// gateway's active probe would keep it in rotation.
		http.Error(w, "hanging", http.StatusServiceUnavailable)
		return
	}
	w.WriteHeader(http.StatusOK)
	_, _ = w.Write([]byte("ok\n"))
}

func (s *Server) handleModels(w http.ResponseWriter, _ *http.Request) {
	cfg := s.engine.Config()
	list := openai.ModelList{Object: "list"}
	for _, m := range cfg.Models {
		list.Data = append(list.Data, openai.Model{
			ID:      m,
			Object:  "model",
			Created: s.now().Unix(),
			OwnedBy: cfg.Name,
		})
	}
	writeJSON(w, http.StatusOK, list)
}

func (s *Server) handleStats(w http.ResponseWriter, _ *http.Request) {
	writeJSON(w, http.StatusOK, s.engine.Stats())
}

func (s *Server) handleGetFaults(w http.ResponseWriter, _ *http.Request) {
	writeJSON(w, http.StatusOK, s.engine.Faults().Get())
}

func (s *Server) handleSetFaults(w http.ResponseWriter, r *http.Request) {
	var state FaultState
	if err := json.NewDecoder(r.Body).Decode(&state); err != nil {
		writeError(w, http.StatusBadRequest, err.Error(), openai.ErrTypeInvalidRequest)
		return
	}
	s.engine.Faults().Set(state)
	s.logger.Warn("fault state updated", "faults", state)
	writeJSON(w, http.StatusOK, s.engine.Faults().Get())
}

func (s *Server) handleReset(w http.ResponseWriter, _ *http.Request) {
	s.engine.Reset()
	s.engine.Faults().Clear()
	writeJSON(w, http.StatusOK, map[string]string{"status": "reset"})
}

func (s *Server) handleCrash(w http.ResponseWriter, _ *http.Request) {
	s.logger.Error("crash requested via admin endpoint")
	writeJSON(w, http.StatusOK, map[string]string{"status": "crashing"})
	if f, ok := w.(http.Flusher); ok {
		f.Flush()
	}
	s.exit(1)
}

func (s *Server) handleChatCompletions(w http.ResponseWriter, r *http.Request) {
	ctx := r.Context()
	faults := s.engine.Faults()

	var req openai.ChatRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeError(w, http.StatusBadRequest, "invalid JSON: "+err.Error(), openai.ErrTypeInvalidRequest)
		return
	}
	if err := req.Validate(); err != nil {
		writeError(w, http.StatusBadRequest, err.Error(), openai.ErrTypeInvalidRequest)
		return
	}

	if faults.ShouldError() {
		writeError(w, http.StatusInternalServerError, "injected fault", openai.ErrTypeServer)
		return
	}
	if d := faults.ExtraLatency(); d > 0 && !sleepCtx(ctx, d) {
		return
	}
	if faults.ShouldHang() {
		// Hold the connection open without writing anything. The client's TTFT
		// deadline, not the backend, decides when this ends.
		<-ctx.Done()
		return
	}

	plan := s.engine.Plan(&req)
	cfg := s.engine.Config()

	if req.Stream {
		s.streamCompletion(w, r, &req, plan, cfg.Name)
		return
	}
	s.bufferedCompletion(w, r, &req, plan, cfg.Name)
}

// streamCompletion emits OpenAI-format SSE chunks as tokens are produced.
func (s *Server) streamCompletion(w http.ResponseWriter, r *http.Request, req *openai.ChatRequest, plan Plan, backend string) {
	flusher, ok := w.(http.Flusher)
	if !ok {
		writeError(w, http.StatusInternalServerError, "streaming unsupported", openai.ErrTypeServer)
		return
	}

	// Wait for a batch slot and prefill *before* writing anything. Until the
	// model would produce its first token there is nothing honest to send, and
	// an early 200 would both hide prefill cost from TTFT measurements and
	// prevent returning a real error if admission fails.
	release, err := s.engine.Admit(r.Context(), plan)
	if err != nil {
		if errors.Is(err, ErrQueueFull) {
			writeError(w, http.StatusServiceUnavailable, "backend queue full", openai.ErrTypeOverloaded)
			return
		}
		// Context cancelled: the client already hung up.
		return
	}
	defer release()

	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")
	w.Header().Set("X-Waypoint-Backend", backend)
	w.WriteHeader(http.StatusOK)

	id := completionID(plan)
	created := s.now().Unix()

	// The first chunk carries the assistant role and no content, matching the
	// OpenAI wire format clients expect.
	roleChunk := openai.ChatChunk{
		ID: id, Object: "chat.completion.chunk", Created: created, Model: req.Model,
		SystemFingerprint: backend,
		Choices: []openai.ChunkChoice{{
			Index: 0,
			Delta: openai.Delta{Role: openai.RoleAssistant},
		}},
	}
	if err := writeSSEChunk(w, roleChunk); err != nil {
		return
	}
	flusher.Flush()

	emit := func(_ int, fragment string) error {
		chunk := openai.ChatChunk{
			ID: id, Object: "chat.completion.chunk", Created: created, Model: req.Model,
			SystemFingerprint: backend,
			Choices: []openai.ChunkChoice{{
				Index: 0,
				Delta: openai.Delta{Content: fragment},
			}},
		}
		if err := writeSSEChunk(w, chunk); err != nil {
			return err
		}
		flusher.Flush()
		return nil
	}

	if err := s.engine.Decode(r.Context(), plan, emit); err != nil {
		// The client is gone or the request was cancelled; nothing useful can
		// be written to a broken stream.
		s.logger.Debug("stream aborted", "error", err)
		return
	}

	stop := "stop"
	finalChunk := openai.ChatChunk{
		ID: id, Object: "chat.completion.chunk", Created: created, Model: req.Model,
		SystemFingerprint: backend,
		Choices: []openai.ChunkChoice{{
			Index: 0, Delta: openai.Delta{}, FinishReason: &stop,
		}},
		Usage: &openai.Usage{
			PromptTokens:     plan.PromptTokens,
			CompletionTokens: plan.OutputTokens,
			TotalTokens:      plan.PromptTokens + plan.OutputTokens,
		},
	}
	if err := writeSSEChunk(w, finalChunk); err != nil {
		return
	}
	if _, err := fmt.Fprint(w, "data: [DONE]\n\n"); err != nil {
		return
	}
	flusher.Flush()
}

// bufferedCompletion runs the same simulation but returns a single JSON body.
func (s *Server) bufferedCompletion(w http.ResponseWriter, r *http.Request, req *openai.ChatRequest, plan Plan, backend string) {
	var sb strings.Builder
	emit := func(_ int, fragment string) error {
		sb.WriteString(fragment)
		return nil
	}

	if err := s.engine.Execute(r.Context(), plan, emit); err != nil {
		if errors.Is(err, ErrQueueFull) {
			writeError(w, http.StatusServiceUnavailable, "backend queue full", openai.ErrTypeOverloaded)
			return
		}
		// Context cancellation means the caller hung up; no response is useful.
		return
	}

	w.Header().Set("X-Waypoint-Backend", backend)
	writeJSON(w, http.StatusOK, openai.ChatResponse{
		ID:                completionID(plan),
		Object:            "chat.completion",
		Created:           s.now().Unix(),
		Model:             req.Model,
		SystemFingerprint: backend,
		Choices: []openai.Choice{{
			Index:        0,
			Message:      openai.Message{Role: openai.RoleAssistant, Content: sb.String()},
			FinishReason: "stop",
		}},
		Usage: openai.Usage{
			PromptTokens:     plan.PromptTokens,
			CompletionTokens: plan.OutputTokens,
			TotalTokens:      plan.PromptTokens + plan.OutputTokens,
		},
	})
}

// completionID derives a stable ID from the plan so repeated identical requests
// produce identical responses, which the cache experiments rely on.
func completionID(plan Plan) string {
	return fmt.Sprintf("chatcmpl-mock-%d-%d", plan.PromptTokens, plan.OutputTokens)
}

func writeSSEChunk(w http.ResponseWriter, chunk openai.ChatChunk) error {
	payload, err := json.Marshal(chunk)
	if err != nil {
		return err
	}
	_, err = fmt.Fprintf(w, "data: %s\n\n", payload)
	return err
}

func writeJSON(w http.ResponseWriter, status int, body any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(body)
}

func writeError(w http.ResponseWriter, status int, msg, typ string) {
	writeJSON(w, status, openai.NewError(msg, typ, ""))
}
