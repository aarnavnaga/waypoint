package api

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"io"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/aarnavnaga/waypoint/internal/admission"
	"github.com/aarnavnaga/waypoint/internal/backend"
	"github.com/aarnavnaga/waypoint/internal/config"
	"github.com/aarnavnaga/waypoint/internal/mocksim"
	"github.com/aarnavnaga/waypoint/internal/openai"
	"github.com/aarnavnaga/waypoint/internal/resilience"
	"github.com/aarnavnaga/waypoint/internal/router"
	"github.com/aarnavnaga/waypoint/internal/sse"
	"github.com/aarnavnaga/waypoint/internal/tokenizer"
)

func discardLogger() *slog.Logger {
	return slog.New(slog.NewTextHandler(io.Discard, &slog.HandlerOptions{Level: slog.LevelError + 1}))
}

// fixture is a gateway wired to real mock backends over loopback HTTP, so tests
// exercise the actual streaming path rather than a stubbed transport.
type fixture struct {
	gateway   *httptest.Server
	backends  []*httptest.Server
	engines   []*mocksim.Engine
	pool      *backend.Pool
	router    *router.Router
	admission *admission.Controller
	breakers  *Breakers
	budget    *resilience.RetryBudget
}

type fixtureOpts struct {
	backends    int
	apiKeys     []string
	maxConc     int
	maxQueue    int
	ttft        time.Duration
	maxAttempts int
	mutateMock  func(*mocksim.Config)
}

func newFixture(t *testing.T, opts fixtureOpts) *fixture {
	t.Helper()

	if opts.backends == 0 {
		opts.backends = 2
	}
	if opts.ttft == 0 {
		opts.ttft = 5 * time.Second
	}

	f := &fixture{}
	cfgBackends := make([]config.Backend, 0, opts.backends)

	for i := 0; i < opts.backends; i++ {
		mcfg := mocksim.DefaultConfig()
		mcfg.Name = "mock-" + string(rune('1'+i))
		mcfg.Seed = int64(i + 1)
		mcfg.Prefill.Base = mocksim.Duration{Duration: 0}
		mcfg.Prefill.PerToken = mocksim.Duration{Duration: time.Microsecond}
		mcfg.Decode.TokensPerSecond = 50000
		mcfg.Output.MinTokens, mcfg.Output.MaxTokens = 6, 6
		if opts.mutateMock != nil {
			opts.mutateMock(&mcfg)
		}

		engine := mocksim.NewEngine(mcfg)
		srv := httptest.NewServer(mocksim.NewServer(engine, discardLogger()).Handler())
		t.Cleanup(srv.Close)

		f.backends = append(f.backends, srv)
		f.engines = append(f.engines, engine)
		cfgBackends = append(cfgBackends, config.Backend{
			Name:              mcfg.Name,
			URL:               srv.URL,
			MaxInflightTokens: 100000,
			Models:            mcfg.Models,
		})
	}

	cfg := config.Default()
	cfg.Backends = cfgBackends
	cfg.Auth.APIKeys = opts.apiKeys
	cfg.Server.MaxConcurrent = opts.maxConc
	cfg.Server.MaxQueue = opts.maxQueue
	if cfg.Server.MaxQueue == 0 {
		cfg.Server.MaxQueue = 128
	}
	cfg.Resilience.TTFTTimeout = config.Duration{Duration: opts.ttft}

	pool, err := backend.NewPool(cfg.Backends)
	if err != nil {
		t.Fatalf("NewPool: %v", err)
	}
	f.pool = pool

	est := tokenizer.NewCharHeuristic()
	rt, err := router.New(cfg.Routing, est)
	if err != nil {
		t.Fatalf("router.New: %v", err)
	}
	f.router = rt

	admitter := admission.New(admission.Options{
		MaxConcurrent: cfg.Server.MaxConcurrent,
		MaxQueue:      cfg.Server.MaxQueue,
	})
	f.admission = admitter

	proxy := NewProxy(&http.Client{}, Timeouts{
		TTFT:    cfg.Resilience.TTFTTimeout.Duration,
		Request: cfg.Resilience.RequestTimeout.Duration,
	}, discardLogger())

	if opts.maxAttempts == 0 {
		opts.maxAttempts = 1 // most tests assert single-attempt behavior
	}
	breakers := NewBreakers(pool, cfg.Resilience.Breaker)
	budget := resilience.NewRetryBudget(resilience.RetryBudgetOptions{
		Percent: cfg.Resilience.RetryBudgetPct, MinRetriesPerSecond: 100,
	})
	f.breakers = breakers
	f.budget = budget

	dispatcher := NewDispatcher(DispatcherOptions{
		Pool: pool, Router: rt, Proxy: proxy,
		Breakers:    breakers,
		Budget:      budget,
		Backoff:     resilience.NewBackoff(time.Millisecond, 5*time.Millisecond),
		MaxAttempts: opts.maxAttempts,
		Logger:      discardLogger(),
	})

	srv := NewServer(Options{
		Config:     cfg,
		Pool:       pool,
		Router:     rt,
		Proxy:      proxy,
		Dispatcher: dispatcher,
		Breakers:   breakers,
		Budget:     budget,
		Admission:  admitter,
		Estimator:  est,
		Logger:     discardLogger(),
		Version:    "test",
	})

	f.gateway = httptest.NewServer(srv.Handler())
	t.Cleanup(f.gateway.Close)
	return f
}

func (f *fixture) post(t *testing.T, path string, body any, headers map[string]string) *http.Response {
	t.Helper()

	var reader io.Reader
	switch v := body.(type) {
	case nil:
	case string:
		reader = strings.NewReader(v)
	default:
		payload, err := json.Marshal(v)
		if err != nil {
			t.Fatalf("marshal body: %v", err)
		}
		reader = bytes.NewReader(payload)
	}

	req, err := http.NewRequest(http.MethodPost, f.gateway.URL+path, reader)
	if err != nil {
		t.Fatalf("new request: %v", err)
	}
	req.Header.Set("Content-Type", "application/json")
	for k, v := range headers {
		req.Header.Set(k, v)
	}

	resp, err := f.gateway.Client().Do(req)
	if err != nil {
		t.Fatalf("POST %s: %v", path, err)
	}
	t.Cleanup(func() { _ = resp.Body.Close() })
	return resp
}

func chatBody(content string, stream bool) openai.ChatRequest {
	return openai.ChatRequest{
		Model:    "mock-7b",
		Stream:   stream,
		Messages: []openai.Message{{Role: openai.RoleUser, Content: content}},
	}
}

// collectStream reads an SSE response and returns the assembled content plus
// the number of content-bearing chunks.
func collectStream(t *testing.T, resp *http.Response) (content string, chunks int, sawDone bool) {
	t.Helper()

	var sb strings.Builder
	dec := sse.NewDecoder(resp.Body)
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
			if c.Delta.Content != "" {
				sb.WriteString(c.Delta.Content)
				chunks++
			}
		}
	}
	return sb.String(), chunks, sawDone
}

func TestStreamingPassesThroughGateway(t *testing.T) {
	f := newFixture(t, fixtureOpts{})

	resp := f.post(t, "/v1/chat/completions", chatBody("hello gateway", true), nil)
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("status = %d, want 200", resp.StatusCode)
	}
	if ct := resp.Header.Get("Content-Type"); ct != "text/event-stream" {
		t.Errorf("Content-Type = %q, want text/event-stream", ct)
	}
	if resp.Header.Get("X-Waypoint-Backend") == "" {
		t.Error("response does not identify the serving backend")
	}

	content, chunks, sawDone := collectStream(t, resp)
	if !sawDone {
		t.Error("stream did not terminate with [DONE]")
	}
	if chunks != 6 {
		t.Errorf("got %d content chunks, want 6", chunks)
	}
	if content == "" {
		t.Error("stream produced no content")
	}
}

func TestNonStreamingPassesThroughGateway(t *testing.T) {
	f := newFixture(t, fixtureOpts{})

	resp := f.post(t, "/v1/chat/completions", chatBody("hello", false), nil)
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("status = %d, want 200", resp.StatusCode)
	}

	var out openai.ChatResponse
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
		t.Fatalf("decode: %v", err)
	}
	if len(out.Choices) != 1 || out.Choices[0].Message.Content == "" {
		t.Errorf("unexpected completion: %+v", out)
	}
	if out.Usage.CompletionTokens != 6 {
		t.Errorf("CompletionTokens = %d, want 6", out.Usage.CompletionTokens)
	}
}

// The gateway must not mangle responses: what the client receives has to match
// what the backend produced, token for token.
func TestStreamContentMatchesBackend(t *testing.T) {
	f := newFixture(t, fixtureOpts{backends: 1})

	gwResp := f.post(t, "/v1/chat/completions", chatBody("fidelity check", true), nil)
	gwContent, _, _ := collectStream(t, gwResp)

	// Ask the backend directly for the same prompt.
	payload, err := json.Marshal(chatBody("fidelity check", true))
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	directResp, err := http.Post(f.backends[0].URL+"/v1/chat/completions", "application/json", bytes.NewReader(payload))
	if err != nil {
		t.Fatalf("direct POST: %v", err)
	}
	defer directResp.Body.Close()
	directContent, _, _ := collectStream(t, directResp)

	if gwContent != directContent {
		t.Errorf("gateway altered the response:\nvia gateway: %q\ndirect:      %q",
			gwContent, directContent)
	}
}

func TestUnknownFieldsReachBackend(t *testing.T) {
	f := newFixture(t, fixtureOpts{backends: 1})

	// guided_json is not modeled by openai.ChatRequest; it must still survive
	// the round trip rather than being silently dropped by the proxy.
	body := `{"model":"mock-7b","stream":false,"messages":[{"role":"user","content":"hi"}],"guided_json":{"type":"object"}}`
	resp := f.post(t, "/v1/chat/completions", body, nil)

	if resp.StatusCode != http.StatusOK {
		t.Fatalf("status = %d, want 200", resp.StatusCode)
	}
}

func TestAuthenticationRequired(t *testing.T) {
	f := newFixture(t, fixtureOpts{apiKeys: []string{"secret-key"}})

	tests := []struct {
		name    string
		headers map[string]string
		want    int
	}{
		{"no key", nil, http.StatusUnauthorized},
		{"wrong key", map[string]string{"Authorization": "Bearer nope"}, http.StatusUnauthorized},
		{"malformed scheme", map[string]string{"Authorization": "secret-key"}, http.StatusUnauthorized},
		{"valid bearer", map[string]string{"Authorization": "Bearer secret-key"}, http.StatusOK},
		{"case-insensitive scheme", map[string]string{"Authorization": "bearer secret-key"}, http.StatusOK},
		{"x-api-key header", map[string]string{"X-API-Key": "secret-key"}, http.StatusOK},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			resp := f.post(t, "/v1/chat/completions", chatBody("hi", false), tc.headers)
			if resp.StatusCode != tc.want {
				t.Errorf("status = %d, want %d", resp.StatusCode, tc.want)
			}
			if tc.want == http.StatusUnauthorized {
				var env openai.ErrorEnvelope
				if err := json.NewDecoder(resp.Body).Decode(&env); err != nil {
					t.Fatalf("decode error: %v", err)
				}
				if env.Error.Type != openai.ErrTypeAuthentication {
					t.Errorf("error type = %q, want %q", env.Error.Type, openai.ErrTypeAuthentication)
				}
			}
		})
	}
}

func TestAuthDisabledWhenNoKeysConfigured(t *testing.T) {
	f := newFixture(t, fixtureOpts{})
	resp := f.post(t, "/v1/chat/completions", chatBody("hi", false), nil)
	if resp.StatusCode != http.StatusOK {
		t.Errorf("status = %d, want 200 when auth is disabled", resp.StatusCode)
	}
}

func TestInvalidRequestsRejected(t *testing.T) {
	f := newFixture(t, fixtureOpts{})

	tests := []struct {
		name string
		body string
		want int
	}{
		{"malformed json", `{"model":`, http.StatusBadRequest},
		{"missing model", `{"messages":[{"role":"user","content":"x"}]}`, http.StatusBadRequest},
		{"empty messages", `{"model":"mock-7b","messages":[]}`, http.StatusBadRequest},
		{"bad temperature", `{"model":"mock-7b","messages":[{"role":"user","content":"x"}],"temperature":5}`, http.StatusBadRequest},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			resp := f.post(t, "/v1/chat/completions", tc.body, nil)
			if resp.StatusCode != tc.want {
				t.Errorf("status = %d, want %d", resp.StatusCode, tc.want)
			}

			var env openai.ErrorEnvelope
			if err := json.NewDecoder(resp.Body).Decode(&env); err != nil {
				t.Fatalf("decode: %v", err)
			}
			if env.Error.Message == "" {
				t.Error("error envelope missing message")
			}
		})
	}
}

func TestUnknownModelReturns404(t *testing.T) {
	f := newFixture(t, fixtureOpts{})

	resp := f.post(t, "/v1/chat/completions", openai.ChatRequest{
		Model:    "gpt-9",
		Messages: []openai.Message{{Role: openai.RoleUser, Content: "hi"}},
	}, nil)

	if resp.StatusCode != http.StatusNotFound {
		t.Errorf("status = %d, want 404", resp.StatusCode)
	}
}

func TestNoHealthyBackendReturns503(t *testing.T) {
	f := newFixture(t, fixtureOpts{})
	for _, b := range f.pool.All() {
		b.SetHealthy(false)
	}

	resp := f.post(t, "/v1/chat/completions", chatBody("hi", false), nil)
	if resp.StatusCode != http.StatusServiceUnavailable {
		t.Errorf("status = %d, want 503", resp.StatusCode)
	}
	if resp.Header.Get("Retry-After") == "" {
		t.Error("503 should carry Retry-After")
	}
}

// A backend returning 500 must surface as 502 from the gateway: the client's
// request was fine, the upstream failed.
func TestBackendErrorBecomes502(t *testing.T) {
	f := newFixture(t, fixtureOpts{backends: 1})
	f.engines[0].Faults().Set(mocksim.FaultState{ErrorRate: 1.0})

	resp := f.post(t, "/v1/chat/completions", chatBody("hi", false), nil)
	if resp.StatusCode != http.StatusBadGateway {
		t.Errorf("status = %d, want 502", resp.StatusCode)
	}
}

func TestTTFTTimeoutReturns504(t *testing.T) {
	f := newFixture(t, fixtureOpts{backends: 1, ttft: 100 * time.Millisecond})
	f.engines[0].Faults().Set(mocksim.FaultState{Hang: true})

	start := time.Now()
	resp := f.post(t, "/v1/chat/completions", chatBody("hi", true), nil)
	elapsed := time.Since(start)

	if resp.StatusCode != http.StatusGatewayTimeout {
		t.Errorf("status = %d, want 504", resp.StatusCode)
	}
	if elapsed > 3*time.Second {
		t.Errorf("request took %s; TTFT deadline was not enforced", elapsed)
	}
}

func TestQueueFullSheds503(t *testing.T) {
	// One concurrent request, no queue: the second must be shed immediately.
	f := newFixture(t, fixtureOpts{
		backends: 1,
		maxConc:  1,
		maxQueue: 1,
		mutateMock: func(c *mocksim.Config) {
			c.Decode.TokensPerSecond = 20 // slow enough to hold the slot
			c.Output.MinTokens, c.Output.MaxTokens = 50, 50
		},
	})

	// Hold the slot and fill the queue with background requests. These own
	// their own response bodies so they cannot race with test cleanup.
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	var wg sync.WaitGroup
	occupy := func(prompt string) {
		defer wg.Done()
		payload, err := json.Marshal(chatBody(prompt, true))
		if err != nil {
			return
		}
		req, err := http.NewRequestWithContext(ctx, http.MethodPost,
			f.gateway.URL+"/v1/chat/completions", bytes.NewReader(payload))
		if err != nil {
			return
		}
		req.Header.Set("Content-Type", "application/json")

		resp, err := f.gateway.Client().Do(req)
		if err != nil {
			return
		}
		defer func() { _ = resp.Body.Close() }()
		_, _ = io.Copy(io.Discard, resp.Body)
	}

	wg.Add(1)
	go occupy("holds the slot")
	wg.Add(1)
	go occupy("fills the queue")

	// Wait until the controller reports the slot taken and the queue occupied.
	deadline := time.Now().Add(2 * time.Second)
	for time.Now().Before(deadline) {
		if f.admission.Stats().Active >= 1 && f.admission.QueueDepth() >= 1 {
			break
		}
		time.Sleep(5 * time.Millisecond)
	}

	resp := f.post(t, "/v1/chat/completions", chatBody("shed me", false), nil)
	if resp.StatusCode != http.StatusServiceUnavailable {
		t.Errorf("status = %d, want 503 when the queue is full", resp.StatusCode)
	}
	if resp.Header.Get("Retry-After") == "" {
		t.Error("shed response should carry Retry-After")
	}

	cancel()
	wg.Wait()
}

func TestModelsUnion(t *testing.T) {
	f := newFixture(t, fixtureOpts{backends: 2})
	f.pool.All()[0].SetModels([]string{"mock-7b", "extra"})
	f.pool.All()[1].SetModels([]string{"mock-7b"})

	resp, err := f.gateway.Client().Get(f.gateway.URL + "/v1/models")
	if err != nil {
		t.Fatalf("GET /v1/models: %v", err)
	}
	defer resp.Body.Close()

	var list openai.ModelList
	if err := json.NewDecoder(resp.Body).Decode(&list); err != nil {
		t.Fatalf("decode: %v", err)
	}
	if len(list.Data) != 2 {
		t.Fatalf("got %d models, want 2 (deduplicated union)", len(list.Data))
	}
	if list.Data[0].ID != "extra" || list.Data[1].ID != "mock-7b" {
		t.Errorf("models = %+v, want sorted [extra mock-7b]", list.Data)
	}
}

func TestHealthzAlwaysOK(t *testing.T) {
	f := newFixture(t, fixtureOpts{})
	for _, b := range f.pool.All() {
		b.SetHealthy(false)
	}

	resp, err := f.gateway.Client().Get(f.gateway.URL + "/healthz")
	if err != nil {
		t.Fatalf("GET /healthz: %v", err)
	}
	defer resp.Body.Close()

	// Liveness is not readiness: a gateway with no backends is still alive.
	if resp.StatusCode != http.StatusOK {
		t.Errorf("status = %d, want 200", resp.StatusCode)
	}
}

func TestReadyzReflectsBackendHealth(t *testing.T) {
	f := newFixture(t, fixtureOpts{})

	resp, err := f.gateway.Client().Get(f.gateway.URL + "/readyz")
	if err != nil {
		t.Fatalf("GET /readyz: %v", err)
	}
	if resp.StatusCode != http.StatusOK {
		t.Errorf("status = %d, want 200 when backends are healthy", resp.StatusCode)
	}
	resp.Body.Close()

	for _, b := range f.pool.All() {
		b.SetHealthy(false)
	}

	resp2, err := f.gateway.Client().Get(f.gateway.URL + "/readyz")
	if err != nil {
		t.Fatalf("GET /readyz: %v", err)
	}
	defer resp2.Body.Close()
	if resp2.StatusCode != http.StatusServiceUnavailable {
		t.Errorf("status = %d, want 503 with no healthy backends", resp2.StatusCode)
	}
}

func TestAdminBackends(t *testing.T) {
	f := newFixture(t, fixtureOpts{backends: 3})

	resp, err := f.gateway.Client().Get(f.gateway.URL + "/admin/backends")
	if err != nil {
		t.Fatalf("GET /admin/backends: %v", err)
	}
	defer resp.Body.Close()

	var body struct {
		Policy   string          `json:"policy"`
		Backends []backend.Stats `json:"backends"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&body); err != nil {
		t.Fatalf("decode: %v", err)
	}
	if len(body.Backends) != 3 {
		t.Errorf("got %d backends, want 3", len(body.Backends))
	}
	if body.Policy != config.PolicyRoundRobin {
		t.Errorf("policy = %q", body.Policy)
	}
}

func TestRoundRobinSpreadsAcrossBackends(t *testing.T) {
	f := newFixture(t, fixtureOpts{backends: 3})

	seen := map[string]int{}
	for i := 0; i < 9; i++ {
		resp := f.post(t, "/v1/chat/completions", chatBody("spread", false), nil)
		if resp.StatusCode != http.StatusOK {
			t.Fatalf("request %d: status %d", i, resp.StatusCode)
		}
		seen[resp.Header.Get("X-Waypoint-Backend")]++
	}

	if len(seen) != 3 {
		t.Errorf("round robin used %d backends, want 3: %v", len(seen), seen)
	}
	for name, count := range seen {
		if count != 3 {
			t.Errorf("backend %s served %d requests, want 3", name, count)
		}
	}
}

func TestSetRoutingPolicyAtRuntime(t *testing.T) {
	f := newFixture(t, fixtureOpts{})

	resp := f.post(t, "/admin/routing", map[string]string{"policy": "nonsense"}, nil)
	if resp.StatusCode != http.StatusBadRequest {
		t.Errorf("status = %d, want 400 for an unknown policy", resp.StatusCode)
	}

	ok := f.post(t, "/admin/routing", map[string]string{"policy": config.PolicyRoundRobin}, nil)
	if ok.StatusCode != http.StatusOK {
		t.Errorf("status = %d, want 200", ok.StatusCode)
	}
	if got := f.router.PolicyName(); got != config.PolicyRoundRobin {
		t.Errorf("policy = %q", got)
	}
}

func TestRequestBodyTooLarge(t *testing.T) {
	f := newFixture(t, fixtureOpts{})

	huge := strings.Repeat("x", maxRequestBytes+1024)
	resp := f.post(t, "/v1/chat/completions", chatBody(huge, false), nil)

	if resp.StatusCode != http.StatusRequestEntityTooLarge {
		t.Errorf("status = %d, want 413", resp.StatusCode)
	}
}

// Load must be released after a request completes, or routing permanently
// believes the fleet is busier than it is.
func TestInflightLoadReleasedAfterRequest(t *testing.T) {
	f := newFixture(t, fixtureOpts{backends: 1})

	for i := 0; i < 3; i++ {
		resp := f.post(t, "/v1/chat/completions", chatBody("accounting", false), nil)
		if resp.StatusCode != http.StatusOK {
			t.Fatalf("status = %d", resp.StatusCode)
		}
	}

	if got := f.pool.TotalInflightTokens(); got != 0 {
		t.Errorf("TotalInflightTokens = %d after all requests completed, want 0", got)
	}
	if got := f.pool.All()[0].InflightRequests(); got != 0 {
		t.Errorf("InflightRequests = %d, want 0", got)
	}
}

func TestClientDisconnectDoesNotLeakLoad(t *testing.T) {
	f := newFixture(t, fixtureOpts{
		backends: 1,
		mutateMock: func(c *mocksim.Config) {
			c.Decode.TokensPerSecond = 20
			c.Output.MinTokens, c.Output.MaxTokens = 100, 100
		},
	})

	payload, err := json.Marshal(chatBody("abandon me", true))
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	req, err := http.NewRequest(http.MethodPost, f.gateway.URL+"/v1/chat/completions", bytes.NewReader(payload))
	if err != nil {
		t.Fatalf("new request: %v", err)
	}
	req.Header.Set("Content-Type", "application/json")

	client := &http.Client{Timeout: 200 * time.Millisecond}
	resp, err := client.Do(req)
	if err == nil {
		// Read a little, then hang up mid-stream.
		buf := make([]byte, 64)
		_, _ = resp.Body.Read(buf)
		_ = resp.Body.Close()
	}

	deadline := time.Now().Add(3 * time.Second)
	for time.Now().Before(deadline) {
		if f.pool.TotalInflightTokens() == 0 {
			return
		}
		time.Sleep(20 * time.Millisecond)
	}
	t.Errorf("TotalInflightTokens = %d after client disconnect, want 0",
		f.pool.TotalInflightTokens())
}

func TestPrefixKeyStableAcrossTurns(t *testing.T) {
	system := strings.Repeat("You are a careful assistant. ", 10)

	turn1 := &openai.ChatRequest{Model: "m", Messages: []openai.Message{
		{Role: openai.RoleSystem, Content: system},
		{Role: openai.RoleUser, Content: "first question"},
	}}
	turn5 := &openai.ChatRequest{Model: "m", Messages: []openai.Message{
		{Role: openai.RoleSystem, Content: system},
		{Role: openai.RoleUser, Content: "first question"},
		{Role: openai.RoleAssistant, Content: "first answer"},
		{Role: openai.RoleUser, Content: "a completely different follow-up"},
	}}

	// With a prefix budget covering only the system prompt, later turns of the
	// same conversation must hash identically so they stay on one backend.
	if router.PrefixKey(turn1, 20) != router.PrefixKey(turn5, 20) {
		t.Error("prefix key changed across turns; conversations would not be sticky")
	}

	// A different system prompt must produce a different key.
	other := &openai.ChatRequest{Model: "m", Messages: []openai.Message{
		{Role: openai.RoleSystem, Content: "You are a rude assistant."},
		{Role: openai.RoleUser, Content: "first question"},
	}}
	if router.PrefixKey(turn1, 20) == router.PrefixKey(other, 20) {
		t.Error("different system prompts produced the same prefix key")
	}
}

func TestEstimateTokensIncludesOutputBudget(t *testing.T) {
	rt, err := router.New(config.Default().Routing, tokenizer.NewCharHeuristic())
	if err != nil {
		t.Fatalf("router.New: %v", err)
	}

	small := &openai.ChatRequest{Model: "m", Messages: []openai.Message{
		{Role: openai.RoleUser, Content: "hi"},
	}}
	big := &openai.ChatRequest{Model: "m", Messages: []openai.Message{
		{Role: openai.RoleUser, Content: "hi"},
	}}
	maxTok := 4000
	big.MaxTokens = &maxTok

	if rt.EstimateTokens(big) <= rt.EstimateTokens(small) {
		t.Error("a large max_tokens must raise the estimated cost")
	}
}

// bytesReader is a small helper so tests can post pre-marshaled payloads.
func bytesReader(b []byte) io.Reader { return bytes.NewReader(b) }
