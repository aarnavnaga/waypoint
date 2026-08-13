package mocksim

import (
	"bytes"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/aarnavnaga/waypoint/internal/openai"
	"github.com/aarnavnaga/waypoint/internal/sse"
)

func newTestServer(t *testing.T, cfg Config) *httptest.Server {
	t.Helper()
	srv := httptest.NewServer(NewServer(NewEngine(cfg), discardLogger()).Handler())
	t.Cleanup(srv.Close)
	return srv
}

func postChat(t *testing.T, srv *httptest.Server, body any) *http.Response {
	t.Helper()
	payload, err := json.Marshal(body)
	if err != nil {
		t.Fatalf("marshal request: %v", err)
	}
	resp, err := srv.Client().Post(srv.URL+"/v1/chat/completions", "application/json", bytes.NewReader(payload))
	if err != nil {
		t.Fatalf("POST /v1/chat/completions: %v", err)
	}
	t.Cleanup(func() { _ = resp.Body.Close() })
	return resp
}

func TestServerHealthz(t *testing.T) {
	srv := newTestServer(t, testConfig())

	resp, err := srv.Client().Get(srv.URL + "/healthz")
	if err != nil {
		t.Fatalf("GET /healthz: %v", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		t.Errorf("status = %d, want 200", resp.StatusCode)
	}
}

func TestServerModels(t *testing.T) {
	cfg := testConfig()
	cfg.Models = []string{"mock-7b", "mock-13b"}
	srv := newTestServer(t, cfg)

	resp, err := srv.Client().Get(srv.URL + "/v1/models")
	if err != nil {
		t.Fatalf("GET /v1/models: %v", err)
	}
	defer resp.Body.Close()

	var list openai.ModelList
	if err := json.NewDecoder(resp.Body).Decode(&list); err != nil {
		t.Fatalf("decode: %v", err)
	}
	if len(list.Data) != 2 {
		t.Fatalf("got %d models, want 2", len(list.Data))
	}
	if list.Data[0].ID != "mock-7b" {
		t.Errorf("first model = %q", list.Data[0].ID)
	}
}

func TestServerNonStreamingCompletion(t *testing.T) {
	srv := newTestServer(t, testConfig())

	resp := postChat(t, srv, openai.ChatRequest{
		Model:    "mock-7b",
		Messages: []openai.Message{{Role: openai.RoleUser, Content: "hello"}},
	})
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("status = %d, want 200", resp.StatusCode)
	}

	var out openai.ChatResponse
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
		t.Fatalf("decode: %v", err)
	}

	if out.Object != "chat.completion" {
		t.Errorf("Object = %q", out.Object)
	}
	if len(out.Choices) != 1 {
		t.Fatalf("got %d choices, want 1", len(out.Choices))
	}
	if out.Choices[0].Message.Content == "" {
		t.Error("empty completion content")
	}
	if out.Choices[0].Message.Role != openai.RoleAssistant {
		t.Errorf("role = %q, want assistant", out.Choices[0].Message.Role)
	}
	if out.Usage.TotalTokens != out.Usage.PromptTokens+out.Usage.CompletionTokens {
		t.Error("usage totals are inconsistent")
	}
	if out.Choices[0].FinishReason != "stop" {
		t.Errorf("FinishReason = %q", out.Choices[0].FinishReason)
	}
}

func TestServerStreamingCompletion(t *testing.T) {
	srv := newTestServer(t, testConfig())

	resp := postChat(t, srv, openai.ChatRequest{
		Model:    "mock-7b",
		Stream:   true,
		Messages: []openai.Message{{Role: openai.RoleUser, Content: "hello"}},
	})
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("status = %d, want 200", resp.StatusCode)
	}
	if ct := resp.Header.Get("Content-Type"); ct != "text/event-stream" {
		t.Errorf("Content-Type = %q, want text/event-stream", ct)
	}

	dec := sse.NewDecoder(resp.Body)
	var (
		chunks    []openai.ChatChunk
		sawDone   bool
		content   strings.Builder
		finishSet bool
	)

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
		chunks = append(chunks, chunk)
		if len(chunk.Choices) > 0 {
			content.WriteString(chunk.Choices[0].Delta.Content)
			if chunk.Choices[0].FinishReason != nil {
				finishSet = true
			}
		}
	}

	if !sawDone {
		t.Error("stream did not end with [DONE]")
	}
	if !finishSet {
		t.Error("no chunk carried a finish_reason")
	}
	if content.Len() == 0 {
		t.Error("stream produced no content")
	}
	if len(chunks) < 3 {
		t.Errorf("got %d chunks, want role + content + final", len(chunks))
	}
	if chunks[0].Choices[0].Delta.Role != openai.RoleAssistant {
		t.Error("first chunk must carry the assistant role")
	}
	if chunks[0].Choices[0].Delta.Content != "" {
		t.Error("role chunk should not carry content")
	}
}

// Streaming and buffered modes must agree, otherwise a cached streamed response
// could not be replayed as a buffered one (and vice versa) in M6.
func TestServerStreamingMatchesBuffered(t *testing.T) {
	cfg := testConfig()

	streamSrv := newTestServer(t, cfg)
	bufferSrv := newTestServer(t, cfg)

	req := openai.ChatRequest{
		Model:    "mock-7b",
		Messages: []openai.Message{{Role: openai.RoleUser, Content: "deterministic?"}},
	}

	req.Stream = true
	streamResp := postChat(t, streamSrv, req)
	var streamed strings.Builder
	dec := sse.NewDecoder(streamResp.Body)
	for {
		ev, err := dec.Next()
		if errors.Is(err, io.EOF) {
			break
		}
		if err != nil || ev.IsDone() {
			if err != nil {
				t.Fatalf("sse decode: %v", err)
			}
			continue
		}
		var chunk openai.ChatChunk
		if err := json.Unmarshal([]byte(ev.Data), &chunk); err != nil {
			t.Fatalf("decode chunk: %v", err)
		}
		if len(chunk.Choices) > 0 {
			streamed.WriteString(chunk.Choices[0].Delta.Content)
		}
	}

	req.Stream = false
	bufResp := postChat(t, bufferSrv, req)
	var out openai.ChatResponse
	if err := json.NewDecoder(bufResp.Body).Decode(&out); err != nil {
		t.Fatalf("decode buffered: %v", err)
	}

	if streamed.String() != out.Choices[0].Message.Content {
		t.Errorf("streamed and buffered content differ:\nstream: %q\nbuffer: %q",
			streamed.String(), out.Choices[0].Message.Content)
	}
}

func TestServerRejectsInvalidRequests(t *testing.T) {
	srv := newTestServer(t, testConfig())

	tests := []struct {
		name string
		body string
	}{
		{"malformed json", `{"model":`},
		{"no model", `{"messages":[{"role":"user","content":"hi"}]}`},
		{"no messages", `{"model":"mock-7b","messages":[]}`},
		{"bad temperature", `{"model":"m","messages":[{"role":"user","content":"x"}],"temperature":9}`},
		{"zero max_tokens", `{"model":"m","messages":[{"role":"user","content":"x"}],"max_tokens":0}`},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			resp, err := srv.Client().Post(srv.URL+"/v1/chat/completions",
				"application/json", strings.NewReader(tc.body))
			if err != nil {
				t.Fatalf("POST: %v", err)
			}
			defer resp.Body.Close()

			if resp.StatusCode != http.StatusBadRequest {
				t.Errorf("status = %d, want 400", resp.StatusCode)
			}

			var env openai.ErrorEnvelope
			if err := json.NewDecoder(resp.Body).Decode(&env); err != nil {
				t.Fatalf("decode error body: %v", err)
			}
			if env.Error.Message == "" {
				t.Error("error envelope has no message")
			}
			if env.Error.Type != openai.ErrTypeInvalidRequest {
				t.Errorf("error type = %q", env.Error.Type)
			}
		})
	}
}

func TestServerStats(t *testing.T) {
	srv := newTestServer(t, testConfig())

	postChat(t, srv, openai.ChatRequest{
		Model:    "mock-7b",
		Messages: []openai.Message{{Role: openai.RoleUser, Content: strings.Repeat("warm ", 50)}},
	})

	resp, err := srv.Client().Get(srv.URL + "/stats")
	if err != nil {
		t.Fatalf("GET /stats: %v", err)
	}
	defer resp.Body.Close()

	var stats Stats
	if err := json.NewDecoder(resp.Body).Decode(&stats); err != nil {
		t.Fatalf("decode stats: %v", err)
	}

	if stats.Requests != 1 {
		t.Errorf("Requests = %d, want 1", stats.Requests)
	}
	if stats.PrefixCache.BlockSize == 0 {
		t.Error("prefix cache stats missing")
	}
	if stats.Scheduler.MaxBatch == 0 {
		t.Error("scheduler stats missing")
	}
}

func TestServerFaultInjectionErrors(t *testing.T) {
	srv := newTestServer(t, testConfig())

	setFaults(t, srv, FaultState{ErrorRate: 1.0})

	resp := postChat(t, srv, openai.ChatRequest{
		Model:    "mock-7b",
		Messages: []openai.Message{{Role: openai.RoleUser, Content: "hi"}},
	})
	if resp.StatusCode != http.StatusInternalServerError {
		t.Errorf("status = %d, want 500", resp.StatusCode)
	}

	var env openai.ErrorEnvelope
	if err := json.NewDecoder(resp.Body).Decode(&env); err != nil {
		t.Fatalf("decode: %v", err)
	}
	if env.Error.Type != openai.ErrTypeServer {
		t.Errorf("error type = %q, want %q", env.Error.Type, openai.ErrTypeServer)
	}
}

func TestServerFaultHangIsUnhealthy(t *testing.T) {
	srv := newTestServer(t, testConfig())
	setFaults(t, srv, FaultState{Hang: true})

	resp, err := srv.Client().Get(srv.URL + "/healthz")
	if err != nil {
		t.Fatalf("GET /healthz: %v", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusServiceUnavailable {
		t.Errorf("status = %d, want 503 when hanging", resp.StatusCode)
	}
}

func TestServerFaultHangBlocksUntilClientGivesUp(t *testing.T) {
	srv := newTestServer(t, testConfig())
	setFaults(t, srv, FaultState{Hang: true})

	client := &http.Client{Timeout: 150 * time.Millisecond}
	body, _ := json.Marshal(openai.ChatRequest{
		Model:    "mock-7b",
		Messages: []openai.Message{{Role: openai.RoleUser, Content: "hi"}},
	})

	start := time.Now()
	resp, err := client.Post(srv.URL+"/v1/chat/completions", "application/json", bytes.NewReader(body))
	if err == nil {
		resp.Body.Close()
		t.Fatal("hanging backend returned a response")
	}
	if elapsed := time.Since(start); elapsed > 2*time.Second {
		t.Errorf("client blocked %s; hang did not respect cancellation", elapsed)
	}
}

func TestServerFaultExtraLatency(t *testing.T) {
	srv := newTestServer(t, testConfig())
	setFaults(t, srv, FaultState{ExtraLatencyMS: 120})

	start := time.Now()
	postChat(t, srv, openai.ChatRequest{
		Model:    "mock-7b",
		Messages: []openai.Message{{Role: openai.RoleUser, Content: "hi"}},
	})
	if elapsed := time.Since(start); elapsed < 100*time.Millisecond {
		t.Errorf("request took %s, want >= ~120ms of injected latency", elapsed)
	}
}

func TestServerAdminReset(t *testing.T) {
	srv := newTestServer(t, testConfig())
	setFaults(t, srv, FaultState{ErrorRate: 1.0})

	resp, err := srv.Client().Post(srv.URL+"/admin/reset", "application/json", nil)
	if err != nil {
		t.Fatalf("POST /admin/reset: %v", err)
	}
	defer resp.Body.Close()

	// Faults cleared, so the next request should succeed.
	chat := postChat(t, srv, openai.ChatRequest{
		Model:    "mock-7b",
		Messages: []openai.Message{{Role: openai.RoleUser, Content: "hi"}},
	})
	if chat.StatusCode != http.StatusOK {
		t.Errorf("status after reset = %d, want 200", chat.StatusCode)
	}
}

func TestServerAdminCrashCallsExit(t *testing.T) {
	s := NewServer(NewEngine(testConfig()), discardLogger())

	exited := make(chan int, 1)
	s.SetExitFunc(func(code int) { exited <- code })

	srv := httptest.NewServer(s.Handler())
	defer srv.Close()

	resp, err := srv.Client().Post(srv.URL+"/admin/crash", "application/json", nil)
	if err != nil {
		t.Fatalf("POST /admin/crash: %v", err)
	}
	defer resp.Body.Close()

	select {
	case code := <-exited:
		if code != 1 {
			t.Errorf("exit code = %d, want 1", code)
		}
	case <-time.After(time.Second):
		t.Fatal("crash endpoint did not trigger exit")
	}
}

func TestServerGetFaults(t *testing.T) {
	srv := newTestServer(t, testConfig())
	setFaults(t, srv, FaultState{ErrorRate: 0.25, ExtraLatencyMS: 10})

	resp, err := srv.Client().Get(srv.URL + "/admin/faults")
	if err != nil {
		t.Fatalf("GET /admin/faults: %v", err)
	}
	defer resp.Body.Close()

	var state FaultState
	if err := json.NewDecoder(resp.Body).Decode(&state); err != nil {
		t.Fatalf("decode: %v", err)
	}
	if state.ErrorRate != 0.25 || state.ExtraLatencyMS != 10 {
		t.Errorf("fault state = %+v, want the values just set", state)
	}
}

func TestServerSetFaultsRejectsBadJSON(t *testing.T) {
	srv := newTestServer(t, testConfig())

	resp, err := srv.Client().Post(srv.URL+"/admin/faults", "application/json",
		strings.NewReader(`{"error_rate":`))
	if err != nil {
		t.Fatalf("POST: %v", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusBadRequest {
		t.Errorf("status = %d, want 400", resp.StatusCode)
	}
}

func TestServerBackendHeaderIdentifiesServer(t *testing.T) {
	cfg := testConfig()
	cfg.Name = "mock-42"
	srv := newTestServer(t, cfg)

	resp := postChat(t, srv, openai.ChatRequest{
		Model:    "mock-7b",
		Messages: []openai.Message{{Role: openai.RoleUser, Content: "hi"}},
	})
	if got := resp.Header.Get("X-Waypoint-Backend"); got != "mock-42" {
		t.Errorf("X-Waypoint-Backend = %q, want mock-42", got)
	}
}

func setFaults(t *testing.T, srv *httptest.Server, state FaultState) {
	t.Helper()
	body, err := json.Marshal(state)
	if err != nil {
		t.Fatalf("marshal faults: %v", err)
	}
	resp, err := srv.Client().Post(srv.URL+"/admin/faults", "application/json", bytes.NewReader(body))
	if err != nil {
		t.Fatalf("POST /admin/faults: %v", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("set faults status = %d", resp.StatusCode)
	}
}
