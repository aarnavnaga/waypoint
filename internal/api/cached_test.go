package api

import (
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/aarnavnaga/waypoint/internal/cache"
	"github.com/aarnavnaga/waypoint/internal/openai"
	"github.com/aarnavnaga/waypoint/internal/sse"
)

func cachedEntry() *cache.Entry {
	return &cache.Entry{
		Model:            "mock-7b",
		Content:          "One two three four.",
		Chunks:           []string{"One", " two", " three", " four."},
		PromptTokens:     12,
		CompletionTokens: 4,
		Backend:          "mock-2",
		CreatedAt:        time.Now().UTC(),
		Source:           cache.SourceExact,
	}
}

// replayRecorder captures what serveFromCache writes.
func replayCachedResponse(t *testing.T, req *openai.ChatRequest, entry *cache.Entry) *httptest.ResponseRecorder {
	t.Helper()

	s := &Server{} // serveFromCache needs no dependencies
	rec := httptest.NewRecorder()
	s.serveFromCache(rec, req, entry)
	return rec
}

func TestServeFromCacheBuffered(t *testing.T) {
	entry := cachedEntry()
	req := &openai.ChatRequest{Model: "mock-7b"}

	rec := replayCachedResponse(t, req, entry)

	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200", rec.Code)
	}
	if got := rec.Header().Get("X-Waypoint-Cache"); got != cache.SourceExact {
		t.Errorf("X-Waypoint-Cache = %q, want exact", got)
	}
	if got := rec.Header().Get("X-Waypoint-Backend"); got != "mock-2" {
		t.Errorf("X-Waypoint-Backend = %q, want the backend that produced the entry", got)
	}

	var out openai.ChatResponse
	if err := json.Unmarshal(rec.Body.Bytes(), &out); err != nil {
		t.Fatalf("decode: %v", err)
	}
	if out.Choices[0].Message.Content != entry.Content {
		t.Errorf("content = %q, want %q", out.Choices[0].Message.Content, entry.Content)
	}
	if out.Usage.TotalTokens != 16 {
		t.Errorf("TotalTokens = %d, want 16", out.Usage.TotalTokens)
	}
	if out.Choices[0].FinishReason != "stop" {
		t.Errorf("FinishReason = %q", out.Choices[0].FinishReason)
	}
}

// The stored chunk boundaries must be reproduced, or a cached stream arrives as
// one giant chunk and feels worse than an uncached one despite being faster.
func TestServeFromCacheReplaysChunkBoundaries(t *testing.T) {
	entry := cachedEntry()
	req := &openai.ChatRequest{Model: "mock-7b", Stream: true}

	rec := replayCachedResponse(t, req, entry)

	if ct := rec.Header().Get("Content-Type"); ct != "text/event-stream" {
		t.Errorf("Content-Type = %q, want text/event-stream", ct)
	}

	var (
		content strings.Builder
		chunks  int
		sawRole bool
		sawDone bool
		sawStop bool
	)

	dec := sse.NewDecoder(rec.Body)
	for {
		ev, err := dec.Next()
		if errors.Is(err, io.EOF) {
			break
		}
		if err != nil {
			t.Fatalf("sse decode: %v", err)
		}
		if ev.IsDone() {
			sawDone = true
			continue
		}

		var chunk openai.ChatChunk
		if err := json.Unmarshal([]byte(ev.Data), &chunk); err != nil {
			t.Fatalf("decode chunk %q: %v", ev.Data, err)
		}
		for _, c := range chunk.Choices {
			if c.Delta.Role == openai.RoleAssistant {
				sawRole = true
			}
			if c.Delta.Content != "" {
				content.WriteString(c.Delta.Content)
				chunks++
			}
			if c.FinishReason != nil && *c.FinishReason == "stop" {
				sawStop = true
			}
		}
	}

	if !sawRole {
		t.Error("replay omitted the assistant role chunk")
	}
	if !sawStop {
		t.Error("replay omitted a finish_reason")
	}
	if !sawDone {
		t.Error("replay did not terminate with [DONE]")
	}
	if chunks != len(entry.Chunks) {
		t.Errorf("replayed %d chunks, want %d: boundaries were lost", chunks, len(entry.Chunks))
	}
	if content.String() != entry.Content {
		t.Errorf("replayed content = %q, want %q", content.String(), entry.Content)
	}
}

// An entry stored without chunk boundaries must still replay correctly.
func TestServeFromCacheWithoutChunks(t *testing.T) {
	entry := cachedEntry()
	entry.Chunks = nil

	rec := replayCachedResponse(t, &openai.ChatRequest{Model: "m", Stream: true}, entry)

	var content strings.Builder
	dec := sse.NewDecoder(rec.Body)
	for {
		ev, err := dec.Next()
		if errors.Is(err, io.EOF) {
			break
		}
		if err != nil || ev.IsDone() {
			continue
		}
		var chunk openai.ChatChunk
		if json.Unmarshal([]byte(ev.Data), &chunk) != nil {
			continue
		}
		for _, c := range chunk.Choices {
			content.WriteString(c.Delta.Content)
		}
	}

	if content.String() != entry.Content {
		t.Errorf("content = %q, want %q", content.String(), entry.Content)
	}
}

func TestServeFromCacheSemanticReportsSimilarity(t *testing.T) {
	entry := cachedEntry()
	entry.Source = cache.SourceSemantic
	entry.Similarity = 0.8734

	rec := replayCachedResponse(t, &openai.ChatRequest{Model: "m"}, entry)

	if got := rec.Header().Get("X-Waypoint-Cache"); got != cache.SourceSemantic {
		t.Errorf("X-Waypoint-Cache = %q, want semantic", got)
	}
	if got := rec.Header().Get("X-Waypoint-Cache-Similarity"); got != "0.8734" {
		t.Errorf("similarity header = %q, want 0.8734", got)
	}
}

// ---------- captureWriter ----------

func TestCaptureWriterBufferedResponse(t *testing.T) {
	rec := httptest.NewRecorder()
	c := newCaptureWriter(rec)

	body, err := json.Marshal(openai.ChatResponse{
		Choices: []openai.Choice{{
			Message: openai.Message{Role: openai.RoleAssistant, Content: "buffered answer"},
		}},
		Usage: openai.Usage{PromptTokens: 7, CompletionTokens: 3},
	})
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}

	c.WriteHeader(http.StatusOK)
	if _, err := c.Write(body); err != nil {
		t.Fatalf("Write: %v", err)
	}

	entry := c.Entry("mock-7b", "mock-1", 0, 0)
	if entry == nil {
		t.Fatal("Entry returned nil for a successful buffered response")
	}
	if entry.Content != "buffered answer" {
		t.Errorf("Content = %q", entry.Content)
	}
	// Usage from the body should win over the passed-in estimates.
	if entry.PromptTokens != 7 || entry.CompletionTokens != 3 {
		t.Errorf("tokens = %d/%d, want 7/3 from the response body",
			entry.PromptTokens, entry.CompletionTokens)
	}
}

// Writes do not align with SSE event boundaries, so the capture buffer must
// parse only up to the last complete event and carry the remainder forward.
func TestCaptureWriterHandlesSplitEvents(t *testing.T) {
	rec := httptest.NewRecorder()
	c := newCaptureWriter(rec)
	c.Header().Set("Content-Type", "text/event-stream")
	c.WriteHeader(http.StatusOK)

	chunk := func(content string) string {
		payload, _ := json.Marshal(openai.ChatChunk{
			Choices: []openai.ChunkChoice{{Delta: openai.Delta{Content: content}}},
		})
		return "data: " + string(payload) + "\n\n"
	}

	full := chunk("Hello") + chunk(" world") + chunk("!")

	// Feed the stream in deliberately awkward slices that cut through events.
	for i := 0; i < len(full); i += 7 {
		end := i + 7
		if end > len(full) {
			end = len(full)
		}
		if _, err := c.Write([]byte(full[i:end])); err != nil {
			t.Fatalf("Write: %v", err)
		}
	}

	entry := c.Entry("mock-7b", "mock-1", 10, 3)
	if entry == nil {
		t.Fatal("Entry returned nil")
	}
	if entry.Content != "Hello world!" {
		t.Errorf("Content = %q, want %q", entry.Content, "Hello world!")
	}
	if len(entry.Chunks) != 3 {
		t.Errorf("captured %d chunks, want 3", len(entry.Chunks))
	}

	// The client must have received the bytes unchanged.
	if rec.Body.String() != full {
		t.Error("captureWriter altered the bytes forwarded to the client")
	}
}

func TestCaptureWriterSkipsNonOKResponses(t *testing.T) {
	rec := httptest.NewRecorder()
	c := newCaptureWriter(rec)

	c.WriteHeader(http.StatusBadGateway)
	if _, err := c.Write([]byte(`{"error":{"message":"nope"}}`)); err != nil {
		t.Fatalf("Write: %v", err)
	}

	if entry := c.Entry("m", "b", 0, 0); entry != nil {
		t.Error("a failed response was captured for caching")
	}
}

func TestCaptureWriterSkipsEmptyContent(t *testing.T) {
	rec := httptest.NewRecorder()
	c := newCaptureWriter(rec)
	c.Header().Set("Content-Type", "text/event-stream")
	c.WriteHeader(http.StatusOK)

	if _, err := c.Write([]byte("data: [DONE]\n\n")); err != nil {
		t.Fatalf("Write: %v", err)
	}

	if entry := c.Entry("m", "b", 0, 0); entry != nil {
		t.Error("an empty completion was captured; it would poison future hits")
	}
}

func TestCaptureWriterRejectsMalformedBufferedBody(t *testing.T) {
	rec := httptest.NewRecorder()
	c := newCaptureWriter(rec)
	c.WriteHeader(http.StatusOK)

	if _, err := c.Write([]byte("not json at all")); err != nil {
		t.Fatalf("Write: %v", err)
	}

	if entry := c.Entry("m", "b", 0, 0); entry != nil {
		t.Error("an unparseable body was captured for caching")
	}
}

// Without a forwarded Flush, wrapping the ResponseWriter silently disables
// streaming — the same class of bug as the logging middleware in M2.
func TestCaptureWriterForwardsFlush(t *testing.T) {
	rec := httptest.NewRecorder()
	c := newCaptureWriter(rec)

	if _, ok := interface{}(c).(http.Flusher); !ok {
		t.Fatal("captureWriter does not implement http.Flusher")
	}
	c.Flush() // must not panic against a recorder
	if !rec.Flushed {
		t.Error("Flush was not forwarded to the underlying writer")
	}
}

func TestStatusRecorderForwardsFlush(t *testing.T) {
	rec := httptest.NewRecorder()
	sr := &statusRecorder{ResponseWriter: rec, status: http.StatusOK}

	sr.Flush()
	if !rec.Flushed {
		t.Error("statusRecorder.Flush was not forwarded")
	}

	sr.WriteHeader(http.StatusTeapot)
	sr.WriteHeader(http.StatusOK) // first status wins
	if sr.status != http.StatusTeapot {
		t.Errorf("status = %d, want the first one written", sr.status)
	}
}

func TestClientIPPrefersForwardingHeaders(t *testing.T) {
	tests := []struct {
		name    string
		headers map[string]string
		remote  string
		want    string
	}{
		{"remote addr", nil, "10.0.0.5:1234", "10.0.0.5:1234"},
		{"x-real-ip", map[string]string{"X-Real-IP": "203.0.113.9"}, "10.0.0.5:1234", "203.0.113.9"},
		{"x-forwarded-for single", map[string]string{"X-Forwarded-For": "203.0.113.7"}, "10.0.0.5:1", "203.0.113.7"},
		{"x-forwarded-for chain", map[string]string{"X-Forwarded-For": "203.0.113.7, 10.0.0.1"}, "10.0.0.5:1", "203.0.113.7"},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			r := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", http.NoBody)
			r.RemoteAddr = tc.remote
			for k, v := range tc.headers {
				r.Header.Set(k, v)
			}
			if got := clientIP(r); got != tc.want {
				t.Errorf("clientIP = %q, want %q", got, tc.want)
			}
		})
	}
}
