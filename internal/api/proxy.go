package api

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"sync/atomic"
	"time"

	"github.com/aarnavnaga/waypoint/internal/backend"
	"github.com/aarnavnaga/waypoint/internal/openai"
	"github.com/aarnavnaga/waypoint/internal/sse"
)

// Dispatch errors. These are classified so M4's retry logic can decide what is
// worth another attempt.
var (
	// ErrTTFTTimeout means the backend accepted the request but produced no
	// first token in time. Distinct from a transport error: the connection is
	// fine, the server is just too slow or stuck.
	ErrTTFTTimeout = errors.New("backend exceeded time-to-first-token deadline")
	// ErrUpstreamUnavailable means the backend could not be reached at all.
	ErrUpstreamUnavailable = errors.New("backend unreachable")
	// ErrUpstreamServerError means the backend returned 5xx.
	ErrUpstreamServerError = errors.New("backend returned a server error")
)

// Outcome describes what happened on one dispatch attempt.
type Outcome struct {
	Backend *backend.Backend
	// StatusCode is the upstream status, 0 if the request never completed.
	StatusCode int
	// TTFT is the time until the first *content* token reached the client.
	TTFT time.Duration
	// CompletionTokens counts content-bearing chunks forwarded.
	CompletionTokens int
	// PromptTokens is taken from upstream usage when reported.
	PromptTokens int
	// ResponseStarted reports whether anything was written to the client.
	//
	// This is the single most important field for resilience: once a byte of
	// the response has been committed, the request can no longer be failed over
	// to another backend, because a client cannot be told "ignore the tokens I
	// already sent you".
	ResponseStarted bool
	// Duration is the wall time of the attempt.
	Duration time.Duration
}

// Retryable reports whether this attempt may be retried on another backend.
func (o *Outcome) Retryable() bool { return o != nil && !o.ResponseStarted }

// Timeouts bounds each phase of a dispatch.
type Timeouts struct {
	// TTFT bounds the wait for the first content token.
	TTFT time.Duration
	// Request bounds the whole exchange.
	Request time.Duration
}

// Proxy forwards a chat completion to one backend and streams the response back.
type Proxy struct {
	client   *http.Client
	logger   *slog.Logger
	timeouts Timeouts
}

// NewProxy builds a Proxy.
func NewProxy(client *http.Client, timeouts Timeouts, logger *slog.Logger) *Proxy {
	if client == nil {
		client = http.DefaultClient
	}
	if logger == nil {
		logger = slog.Default()
	}
	return &Proxy{client: client, logger: logger, timeouts: timeouts}
}

// Dispatch sends req to b and relays the response to w.
//
// Nothing is written to w until the upstream response is known to be usable, so
// a failed attempt leaves the client response untouched and therefore retryable.
func (p *Proxy) Dispatch(
	ctx context.Context,
	w http.ResponseWriter,
	b *backend.Backend,
	req *openai.ChatRequest,
	body []byte,
	headers http.Header,
) (*Outcome, error) {
	start := time.Now()
	out := &Outcome{Backend: b}

	// The TTFT deadline is enforced by cancelling the request if no first token
	// arrives. It is separate from the overall request deadline so a backend
	// that is merely slow to *start* is detected quickly, while a long but
	// healthy generation is allowed to finish.
	attemptCtx, cancel := context.WithCancel(ctx)
	defer cancel()

	var ttftExceeded atomic.Bool
	var ttftTimer *time.Timer
	if p.timeouts.TTFT > 0 {
		ttftTimer = time.AfterFunc(p.timeouts.TTFT, func() {
			ttftExceeded.Store(true)
			cancel()
		})
		defer ttftTimer.Stop()
	}

	target := b.URL.JoinPath("/v1/chat/completions")
	httpReq, err := http.NewRequestWithContext(attemptCtx, http.MethodPost, target.String(), bytes.NewReader(body))
	if err != nil {
		out.Duration = time.Since(start)
		return out, fmt.Errorf("build upstream request: %w", err)
	}
	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("Accept", "text/event-stream")
	// Propagate tracing and any caller-supplied passthrough headers.
	for k, vs := range headers {
		for _, v := range vs {
			httpReq.Header.Add(k, v)
		}
	}

	resp, err := p.client.Do(httpReq)
	if err != nil {
		out.Duration = time.Since(start)
		if ttftExceeded.Load() {
			return out, ErrTTFTTimeout
		}
		return out, fmt.Errorf("%w: %s: %w", ErrUpstreamUnavailable, b.Name, err)
	}
	defer func() { _ = resp.Body.Close() }()

	out.StatusCode = resp.StatusCode

	// 5xx is the backend's fault and is worth another backend.
	if resp.StatusCode >= http.StatusInternalServerError {
		snippet, _ := io.ReadAll(io.LimitReader(resp.Body, 2048))
		out.Duration = time.Since(start)
		return out, fmt.Errorf("%w: %s returned %d: %s",
			ErrUpstreamServerError, b.Name, resp.StatusCode, bytes.TrimSpace(snippet))
	}

	// 4xx is the client's fault; retrying elsewhere would produce the same
	// result, so pass it straight through.
	if resp.StatusCode >= http.StatusBadRequest {
		out.ResponseStarted = true
		p.relayError(w, resp)
		out.Duration = time.Since(start)
		return out, nil
	}

	if req.Stream {
		err = p.relayStream(w, resp, out, start, ttftTimer)
	} else {
		err = p.relayBuffered(w, resp, out, start, ttftTimer)
	}

	out.Duration = time.Since(start)
	if err != nil && ttftExceeded.Load() {
		return out, ErrTTFTTimeout
	}
	return out, err
}

// relayStream forwards SSE events, counting content tokens and recording TTFT.
func (p *Proxy) relayStream(
	w http.ResponseWriter,
	resp *http.Response,
	out *Outcome,
	start time.Time,
	ttftTimer *time.Timer,
) error {
	flusher, ok := w.(http.Flusher)
	if !ok {
		return errors.New("response writer does not support streaming")
	}

	dec := sse.NewDecoder(resp.Body)
	var enc *sse.Encoder

	for {
		ev, err := dec.Next()
		if errors.Is(err, io.EOF) {
			break
		}
		if err != nil {
			// If the stream already started, the client has partial output and
			// there is nothing better to do than stop.
			return fmt.Errorf("decode upstream stream: %w", err)
		}

		// Headers are written lazily, on the first event that proves the
		// upstream is actually producing. Until then the attempt is retryable.
		if !out.ResponseStarted {
			writeStreamHeaders(w, out.Backend.Name)
			out.ResponseStarted = true
			enc = sse.NewEncoder(&flushWriter{w: w, f: flusher})
		}

		if ev.IsDone() {
			if err := enc.WriteDone(); err != nil {
				return fmt.Errorf("write [DONE]: %w", err)
			}
			break
		}

		if n, prompt := countChunk(ev.Data); n > 0 {
			if out.CompletionTokens == 0 {
				out.TTFT = time.Since(start)
				// First token arrived: the TTFT deadline no longer applies.
				if ttftTimer != nil {
					ttftTimer.Stop()
				}
			}
			out.CompletionTokens += n
			if prompt > 0 {
				out.PromptTokens = prompt
			}
		} else if prompt > 0 {
			out.PromptTokens = prompt
		}

		if err := enc.WriteData(ev.Data); err != nil {
			return fmt.Errorf("write downstream chunk: %w", err)
		}
	}

	// A stream that produced no events at all still owes the client a response.
	if !out.ResponseStarted {
		writeStreamHeaders(w, out.Backend.Name)
		out.ResponseStarted = true
		enc = sse.NewEncoder(&flushWriter{w: w, f: flusher})
		if err := enc.WriteDone(); err != nil {
			return fmt.Errorf("write [DONE]: %w", err)
		}
	}
	return nil
}

// relayBuffered forwards a non-streaming JSON completion.
func (p *Proxy) relayBuffered(
	w http.ResponseWriter,
	resp *http.Response,
	out *Outcome,
	start time.Time,
	ttftTimer *time.Timer,
) error {
	payload, err := io.ReadAll(resp.Body)
	if err != nil {
		return fmt.Errorf("read upstream body: %w", err)
	}
	if ttftTimer != nil {
		ttftTimer.Stop()
	}

	var parsed openai.ChatResponse
	if err := json.Unmarshal(payload, &parsed); err == nil {
		out.CompletionTokens = parsed.Usage.CompletionTokens
		out.PromptTokens = parsed.Usage.PromptTokens
	}
	out.TTFT = time.Since(start)

	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("X-Waypoint-Backend", out.Backend.Name)
	w.WriteHeader(http.StatusOK)
	out.ResponseStarted = true

	if _, err := w.Write(payload); err != nil {
		return fmt.Errorf("write downstream body: %w", err)
	}
	return nil
}

// relayError passes a 4xx upstream response through unchanged.
func (p *Proxy) relayError(w http.ResponseWriter, resp *http.Response) {
	payload, err := io.ReadAll(io.LimitReader(resp.Body, 1<<20))
	if err != nil {
		writeAPIError(w, http.StatusBadGateway, "failed to read upstream error", openai.ErrTypeServer, "")
		return
	}

	contentType := resp.Header.Get("Content-Type")
	if contentType == "" {
		contentType = "application/json"
	}
	w.Header().Set("Content-Type", contentType)
	w.WriteHeader(resp.StatusCode)
	_, _ = w.Write(payload)
}

func writeStreamHeaders(w http.ResponseWriter, backendName string) {
	h := w.Header()
	h.Set("Content-Type", "text/event-stream")
	h.Set("Cache-Control", "no-cache")
	h.Set("Connection", "keep-alive")
	// Nginx buffers proxied responses by default, which would defeat streaming.
	h.Set("X-Accel-Buffering", "no")
	h.Set("X-Waypoint-Backend", backendName)
	w.WriteHeader(http.StatusOK)
}

// countChunk reports how many content tokens a chunk payload carries, and any
// prompt-token count it reports.
//
// The role chunk (delta.role with empty content) must not count: treating it as
// the first token would understate TTFT by the entire prefill duration.
func countChunk(data string) (content, promptTokens int) {
	var chunk openai.ChatChunk
	if err := json.Unmarshal([]byte(data), &chunk); err != nil {
		return 0, 0
	}
	if chunk.Usage != nil {
		promptTokens = chunk.Usage.PromptTokens
	}
	for _, ch := range chunk.Choices {
		if ch.Delta.Content != "" {
			content++
		}
	}
	return content, promptTokens
}

// flushWriter flushes after every write so tokens are not buffered.
type flushWriter struct {
	w io.Writer
	f http.Flusher
}

func (fw *flushWriter) Write(p []byte) (int, error) { return fw.w.Write(p) }
func (fw *flushWriter) Flush()                      { fw.f.Flush() }
