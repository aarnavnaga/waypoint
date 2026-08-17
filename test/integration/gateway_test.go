//go:build integration

// Package integration exercises the gateway against real processes rather than
// in-process fakes: three mock backends and a gateway, each a separate HTTP
// server, communicating over loopback TCP.
//
// Run with: make integration
package integration

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"net"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/aarnavnaga/waypoint/internal/admission"
	"github.com/aarnavnaga/waypoint/internal/api"
	"github.com/aarnavnaga/waypoint/internal/backend"
	"github.com/aarnavnaga/waypoint/internal/config"
	"github.com/aarnavnaga/waypoint/internal/mocksim"
	"github.com/aarnavnaga/waypoint/internal/openai"
	"github.com/aarnavnaga/waypoint/internal/router"
	"github.com/aarnavnaga/waypoint/internal/sse"
	"github.com/aarnavnaga/waypoint/internal/tokenizer"
)

func quietLogger() *slog.Logger {
	return slog.New(slog.NewTextHandler(io.Discard, &slog.HandlerOptions{Level: slog.LevelError + 1}))
}

// stack is a full gateway + backend fleet on real TCP sockets.
type stack struct {
	gatewayURL string
	backends   []*httptest.Server
	engines    []*mocksim.Engine
	pool       *backend.Pool
	router     *router.Router
	client     *http.Client
}

func newStack(t *testing.T, backendCount int, mutate func(*mocksim.Config)) *stack {
	t.Helper()

	s := &stack{client: &http.Client{Timeout: 30 * time.Second}}
	cfgBackends := make([]config.Backend, 0, backendCount)

	for i := 0; i < backendCount; i++ {
		mcfg := mocksim.DefaultConfig()
		mcfg.Name = fmt.Sprintf("mock-%d", i+1)
		mcfg.Seed = int64(i + 1)
		mcfg.Prefill.Base = mocksim.Duration{Duration: time.Millisecond}
		mcfg.Prefill.PerToken = mocksim.Duration{Duration: 50 * time.Microsecond}
		mcfg.Decode.TokensPerSecond = 2000
		mcfg.Output.MinTokens, mcfg.Output.MaxTokens = 12, 12
		if mutate != nil {
			mutate(&mcfg)
		}

		engine := mocksim.NewEngine(mcfg)
		srv := httptest.NewServer(mocksim.NewServer(engine, quietLogger()).Handler())
		t.Cleanup(srv.Close)

		s.backends = append(s.backends, srv)
		s.engines = append(s.engines, engine)
		cfgBackends = append(cfgBackends, config.Backend{
			Name:              mcfg.Name,
			URL:               srv.URL,
			MaxInflightTokens: 200000,
		})
	}

	cfg := config.Default()
	cfg.Backends = cfgBackends
	cfg.Auth.APIKeys = []string{"integration-key"}
	cfg.Server.MaxConcurrent = 64
	cfg.Server.MaxQueue = 256
	cfg.Resilience.TTFTTimeout = config.Duration{Duration: 3 * time.Second}

	pool, err := backend.NewPool(cfg.Backends)
	if err != nil {
		t.Fatalf("NewPool: %v", err)
	}
	s.pool = pool

	est := tokenizer.NewCharHeuristic()
	rt, err := router.New(cfg.Routing, est)
	if err != nil {
		t.Fatalf("router.New: %v", err)
	}
	s.router = rt

	// Discover models the way the real binary does.
	disc := backend.NewDiscoverer(pool, backend.DiscovererOptions{
		Client: &http.Client{Timeout: 2 * time.Second},
		Logger: quietLogger(),
	})
	disc.RefreshAll(context.Background())

	srv := api.NewServer(api.Options{
		Config:    cfg,
		Pool:      pool,
		Router:    rt,
		Proxy:     api.NewProxy(&http.Client{}, api.Timeouts{TTFT: 3 * time.Second}, quietLogger()),
		Admission: admission.New(admission.Options{MaxConcurrent: 64, MaxQueue: 256}),
		Estimator: est,
		Logger:    quietLogger(),
		Version:   "integration",
	})

	// Bind a real listener so the gateway is reached over TCP, not in-process.
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("listen: %v", err)
	}
	httpSrv := &http.Server{Handler: srv.Handler(), ReadHeaderTimeout: 5 * time.Second}
	go func() { _ = httpSrv.Serve(ln) }()
	t.Cleanup(func() {
		ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
		defer cancel()
		_ = httpSrv.Shutdown(ctx)
	})

	s.gatewayURL = "http://" + ln.Addr().String()
	waitReady(t, s)
	return s
}

func waitReady(t *testing.T, s *stack) {
	t.Helper()
	deadline := time.Now().Add(5 * time.Second)
	for time.Now().Before(deadline) {
		resp, err := s.client.Get(s.gatewayURL + "/readyz")
		if err == nil {
			code := resp.StatusCode
			_ = resp.Body.Close()
			if code == http.StatusOK {
				return
			}
		}
		time.Sleep(20 * time.Millisecond)
	}
	t.Fatal("gateway never became ready")
}

func (s *stack) chat(t *testing.T, body any) *http.Response {
	t.Helper()

	payload, err := json.Marshal(body)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	req, err := http.NewRequest(http.MethodPost, s.gatewayURL+"/v1/chat/completions", bytes.NewReader(payload))
	if err != nil {
		t.Fatalf("new request: %v", err)
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer integration-key")

	resp, err := s.client.Do(req)
	if err != nil {
		t.Fatalf("chat request: %v", err)
	}
	t.Cleanup(func() { _ = resp.Body.Close() })
	return resp
}

func readStream(t *testing.T, resp *http.Response) (content string, chunks int, done bool) {
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
			done = true
			continue
		}
		var chunk openai.ChatChunk
		if err := json.Unmarshal([]byte(ev.Data), &chunk); err != nil {
			t.Fatalf("decode chunk: %v", err)
		}
		for _, c := range chunk.Choices {
			if c.Delta.Content != "" {
				sb.WriteString(c.Delta.Content)
				chunks++
			}
		}
	}
	return sb.String(), chunks, done
}

func streamReq(content string) openai.ChatRequest {
	return openai.ChatRequest{
		Model:    "mock-7b",
		Stream:   true,
		Messages: []openai.Message{{Role: openai.RoleUser, Content: content}},
	}
}

func TestEndToEndStreaming(t *testing.T) {
	s := newStack(t, 3, nil)

	resp := s.chat(t, streamReq("hello over tcp"))
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("status = %d, want 200", resp.StatusCode)
	}

	content, chunks, done := readStream(t, resp)
	if !done {
		t.Error("stream did not end with [DONE]")
	}
	if chunks != 12 {
		t.Errorf("got %d content chunks, want 12", chunks)
	}
	if content == "" {
		t.Error("no content streamed")
	}
}

// Tokens must reach the client incrementally. If the gateway buffered the whole
// response, the first chunk would arrive only after generation completed.
func TestStreamingIsIncremental(t *testing.T) {
	s := newStack(t, 1, func(c *mocksim.Config) {
		c.Decode.TokensPerSecond = 20 // 50ms per token
		c.Output.MinTokens, c.Output.MaxTokens = 20, 20
	})

	start := time.Now()
	resp := s.chat(t, streamReq("stream incrementally"))
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("status = %d", resp.StatusCode)
	}

	// Read the stream to completion, recording when the first content token
	// arrived. Stopping at the first token would make "total" meaningless.
	dec := sse.NewDecoder(resp.Body)
	var firstContentAt time.Duration
	tokens := 0
	for {
		ev, err := dec.Next()
		if err != nil {
			break
		}
		if ev.IsDone() {
			break
		}
		var chunk openai.ChatChunk
		if json.Unmarshal([]byte(ev.Data), &chunk) != nil {
			continue
		}
		if len(chunk.Choices) > 0 && chunk.Choices[0].Delta.Content != "" {
			if firstContentAt == 0 {
				firstContentAt = time.Since(start)
			}
			tokens++
		}
	}
	total := time.Since(start)

	if firstContentAt == 0 {
		t.Fatal("never received a content chunk")
	}
	if tokens != 20 {
		t.Errorf("received %d tokens, want 20", tokens)
	}
	// 20 tokens at 50ms each is ~1s total. If the gateway buffered the whole
	// response, the first token would land near the end instead of near the
	// start.
	if firstContentAt > total/2 {
		t.Errorf("first token at %s of %s total: response appears buffered, not streamed",
			firstContentAt, total)
	}
	t.Logf("first token at %s, full stream %s (%d tokens)", firstContentAt, total, tokens)
}

func TestFailoverCandidatesExcludeDeadBackend(t *testing.T) {
	s := newStack(t, 3, nil)

	// Take one backend out of rotation the way the health checker will in M4.
	dead, _ := s.pool.Get("mock-2")
	dead.SetHealthy(false)

	seen := map[string]int{}
	for i := 0; i < 12; i++ {
		resp := s.chat(t, openai.ChatRequest{
			Model:    "mock-7b",
			Messages: []openai.Message{{Role: openai.RoleUser, Content: "route me"}},
		})
		if resp.StatusCode != http.StatusOK {
			t.Fatalf("request %d: status %d", i, resp.StatusCode)
		}
		seen[resp.Header.Get("X-Waypoint-Backend")]++
	}

	if seen["mock-2"] != 0 {
		t.Errorf("unhealthy backend served %d requests", seen["mock-2"])
	}
	if len(seen) != 2 {
		t.Errorf("traffic spread over %d backends, want 2: %v", len(seen), seen)
	}
}

func TestModelsDiscoveredFromBackends(t *testing.T) {
	s := newStack(t, 2, nil)

	req, err := http.NewRequest(http.MethodGet, s.gatewayURL+"/v1/models", nil)
	if err != nil {
		t.Fatalf("new request: %v", err)
	}
	req.Header.Set("Authorization", "Bearer integration-key")

	resp, err := s.client.Do(req)
	if err != nil {
		t.Fatalf("GET /v1/models: %v", err)
	}
	defer resp.Body.Close()

	var list openai.ModelList
	if err := json.NewDecoder(resp.Body).Decode(&list); err != nil {
		t.Fatalf("decode: %v", err)
	}
	if len(list.Data) != 1 || list.Data[0].ID != "mock-7b" {
		t.Errorf("models = %+v, want [mock-7b] discovered from backends", list.Data)
	}
}

func TestConcurrentStreamsAreIsolated(t *testing.T) {
	s := newStack(t, 3, nil)

	const n = 24
	var wg sync.WaitGroup
	results := make([]string, n)
	statuses := make([]int, n)

	for i := 0; i < n; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()

			payload, err := json.Marshal(streamReq(fmt.Sprintf("request number %d", i)))
			if err != nil {
				return
			}
			req, err := http.NewRequest(http.MethodPost,
				s.gatewayURL+"/v1/chat/completions", bytes.NewReader(payload))
			if err != nil {
				return
			}
			req.Header.Set("Content-Type", "application/json")
			req.Header.Set("Authorization", "Bearer integration-key")

			resp, err := s.client.Do(req)
			if err != nil {
				return
			}
			defer func() { _ = resp.Body.Close() }()
			statuses[i] = resp.StatusCode

			var sb strings.Builder
			dec := sse.NewDecoder(resp.Body)
			for {
				ev, err := dec.Next()
				if err != nil || ev.IsDone() {
					break
				}
				var chunk openai.ChatChunk
				if json.Unmarshal([]byte(ev.Data), &chunk) != nil {
					continue
				}
				for _, c := range chunk.Choices {
					sb.WriteString(c.Delta.Content)
				}
			}
			results[i] = sb.String()
		}(i)
	}
	wg.Wait()

	for i := 0; i < n; i++ {
		if statuses[i] != http.StatusOK {
			t.Errorf("request %d: status %d", i, statuses[i])
		}
		if results[i] == "" {
			t.Errorf("request %d produced no content", i)
		}
	}

	// Different prompts must produce different responses: identical output
	// across streams would mean responses were crossed between clients.
	if results[0] == results[1] {
		t.Error("two different prompts produced identical output; streams may be crossed")
	}

	if got := s.pool.TotalInflightTokens(); got != 0 {
		t.Errorf("TotalInflightTokens = %d after all requests finished, want 0", got)
	}
}

// Prefix cache reuse must be visible end to end: the same prompt sent twice to
// the same backend should be served largely from its KV cache.
func TestPrefixCacheWarmsThroughGateway(t *testing.T) {
	s := newStack(t, 1, func(c *mocksim.Config) {
		c.Prefill.PerToken = mocksim.Duration{Duration: 200 * time.Microsecond}
	})

	prompt := strings.Repeat("You are a careful assistant answering questions. ", 40)

	for i := 0; i < 3; i++ {
		resp := s.chat(t, streamReq(prompt))
		if resp.StatusCode != http.StatusOK {
			t.Fatalf("request %d: status %d", i, resp.StatusCode)
		}
		readStream(t, resp)
	}

	stats := s.engines[0].Stats()
	if stats.PrefixCache.HitRate == 0 {
		t.Error("repeated prompts produced no prefix-cache hits through the gateway")
	}
	if stats.CachedToks == 0 {
		t.Error("no prompt tokens were served from cache")
	}
	t.Logf("prefix cache hit rate after 3 identical prompts: %.2f", stats.PrefixCache.HitRate)
}

func TestBackendErrorSurfacesAs502(t *testing.T) {
	s := newStack(t, 1, nil)
	s.engines[0].Faults().Set(mocksim.FaultState{ErrorRate: 1.0})

	resp := s.chat(t, openai.ChatRequest{
		Model:    "mock-7b",
		Messages: []openai.Message{{Role: openai.RoleUser, Content: "will fail"}},
	})
	if resp.StatusCode != http.StatusBadGateway {
		t.Errorf("status = %d, want 502", resp.StatusCode)
	}

	var env openai.ErrorEnvelope
	if err := json.NewDecoder(resp.Body).Decode(&env); err != nil {
		t.Fatalf("decode error body: %v", err)
	}
	if env.Error.Type != openai.ErrTypeServer {
		t.Errorf("error type = %q", env.Error.Type)
	}
}

func TestRuntimePolicySwitch(t *testing.T) {
	s := newStack(t, 3, nil)

	body, err := json.Marshal(map[string]string{"policy": config.PolicyRoundRobin})
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	resp, err := s.client.Post(s.gatewayURL+"/admin/routing", "application/json", bytes.NewReader(body))
	if err != nil {
		t.Fatalf("POST /admin/routing: %v", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		t.Fatalf("status = %d, want 200", resp.StatusCode)
	}
	if got := s.router.PolicyName(); got != config.PolicyRoundRobin {
		t.Errorf("policy = %q", got)
	}
}

func TestMain(m *testing.M) {
	os.Exit(m.Run())
}
