package api

import (
	"context"
	"encoding/json"
	"net/http"
	"strings"
	"time"

	"github.com/aarnavnaga/waypoint/internal/cache"
	"github.com/aarnavnaga/waypoint/internal/openai"
	"github.com/aarnavnaga/waypoint/internal/sse"
)

// captureWriter tees the response to the client while accumulating the
// assistant content, so a successful completion can be cached without buffering
// it first and delaying the client.
//
// Streaming makes this necessary: the response is written incrementally and is
// gone once sent, so the only way to cache it is to observe it in flight.
type captureWriter struct {
	http.ResponseWriter

	status  int
	chunks  []string
	content strings.Builder
	// pending holds bytes not yet parsed into complete SSE events.
	pending strings.Builder
	// buffered collects a non-streaming JSON body.
	buffered  strings.Builder
	streaming bool
}

func newCaptureWriter(w http.ResponseWriter) *captureWriter {
	return &captureWriter{ResponseWriter: w, status: http.StatusOK}
}

func (c *captureWriter) WriteHeader(code int) {
	c.status = code
	c.streaming = strings.HasPrefix(c.Header().Get("Content-Type"), "text/event-stream")
	c.ResponseWriter.WriteHeader(code)
}

func (c *captureWriter) Write(p []byte) (int, error) {
	// Forward first: capture must never delay or block the client.
	n, err := c.ResponseWriter.Write(p)
	if c.status != http.StatusOK {
		return n, err
	}

	if c.streaming {
		c.pending.Write(p[:n])
		c.drainEvents()
	} else {
		c.buffered.Write(p[:n])
	}
	return n, err
}

// Flush forwards to the underlying writer; without it, wrapping the
// ResponseWriter would silently disable streaming.
func (c *captureWriter) Flush() {
	if f, ok := c.ResponseWriter.(http.Flusher); ok {
		f.Flush()
	}
}

// drainEvents parses whatever complete SSE events have accumulated.
//
// Writes do not align with event boundaries, so parsing is done on the buffer
// up to the last complete event terminator and the remainder is kept pending.
func (c *captureWriter) drainEvents() {
	buf := c.pending.String()
	idx := strings.LastIndex(buf, "\n\n")
	if idx < 0 {
		return
	}

	complete := buf[:idx+2]
	remainder := buf[idx+2:]

	c.pending.Reset()
	c.pending.WriteString(remainder)

	dec := sse.NewDecoder(strings.NewReader(complete))
	for {
		ev, err := dec.Next()
		if err != nil {
			return
		}
		if ev.IsDone() {
			continue
		}

		var chunk openai.ChatChunk
		if json.Unmarshal([]byte(ev.Data), &chunk) != nil {
			continue
		}
		for _, choice := range chunk.Choices {
			if choice.Delta.Content != "" {
				c.chunks = append(c.chunks, choice.Delta.Content)
				c.content.WriteString(choice.Delta.Content)
			}
		}
	}
}

// Entry builds a cache entry from what was observed, or nil if the response is
// not cacheable.
func (c *captureWriter) Entry(model, backend string, promptTokens, completionTokens int) *cache.Entry {
	if c.status != http.StatusOK {
		return nil
	}

	content := c.content.String()
	chunks := c.chunks

	if !c.streaming {
		var parsed openai.ChatResponse
		if err := json.Unmarshal([]byte(c.buffered.String()), &parsed); err != nil {
			return nil
		}
		if len(parsed.Choices) == 0 {
			return nil
		}
		content = parsed.Choices[0].Message.Content
		chunks = nil
		if parsed.Usage.PromptTokens > 0 {
			promptTokens = parsed.Usage.PromptTokens
			completionTokens = parsed.Usage.CompletionTokens
		}
	}

	if content == "" {
		return nil
	}

	return &cache.Entry{
		Model:            model,
		Content:          content,
		Chunks:           chunks,
		PromptTokens:     promptTokens,
		CompletionTokens: completionTokens,
		Backend:          backend,
		CreatedAt:        time.Now().UTC(),
	}
}

// storeInCache writes a completed response to cache asynchronously.
//
// The client already has its bytes, so this must never be on the path that
// delays them. A detached context is used deliberately: the request context is
// cancelled the moment the handler returns, which would abort every cache write.
func (s *Server) storeInCache(req *openai.ChatRequest, capture *captureWriter, outcome *Outcome) {
	backendName := ""
	if outcome.Backend != nil {
		backendName = outcome.Backend.Name
	}

	entry := capture.Entry(req.Model, backendName, outcome.PromptTokens, outcome.CompletionTokens)
	if entry == nil {
		return
	}

	go func() {
		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
		s.cache.Store(ctx, req, entry)
	}()
}
