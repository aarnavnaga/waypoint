//go:build integration

package integration

import (
	"encoding/json"
	"fmt"
	"net"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/aarnavnaga/waypoint/internal/admission"
	"github.com/aarnavnaga/waypoint/internal/api"
	"github.com/aarnavnaga/waypoint/internal/backend"
	"github.com/aarnavnaga/waypoint/internal/cache"
	"github.com/aarnavnaga/waypoint/internal/config"
	"github.com/aarnavnaga/waypoint/internal/mocksim"
	"github.com/aarnavnaga/waypoint/internal/openai"
	"github.com/aarnavnaga/waypoint/internal/router"
	"github.com/aarnavnaga/waypoint/internal/tokenizer"
)

// cacheStack is a gateway with both cache layers wired to real Redis.
type cacheStack struct {
	url      string
	client   *http.Client
	apiKey   string
	engine   *mocksim.Engine
	cache    *cache.Cache
	semantic *cache.SemanticCache
}

func newCacheStack(t *testing.T, semanticThreshold float64) *cacheStack {
	t.Helper()

	rdb := requireRedis(t)
	keyPrefix := fmt.Sprintf("cachetest-%d", time.Now().UnixNano())

	mcfg := mocksim.DefaultConfig()
	mcfg.Name = "mock-1"
	mcfg.Prefill.Base = mocksim.Duration{Duration: 20 * time.Millisecond}
	mcfg.Prefill.PerToken = mocksim.Duration{Duration: 50 * time.Microsecond}
	mcfg.Decode.TokensPerSecond = 500
	mcfg.Output.MinTokens, mcfg.Output.MaxTokens = 10, 10

	engine := mocksim.NewEngine(mcfg)
	backendSrv := httptest.NewServer(mocksim.NewServer(engine, quietLogger()).Handler())
	t.Cleanup(backendSrv.Close)

	cfg := config.Default()
	cfg.Backends = []config.Backend{{Name: "mock-1", URL: backendSrv.URL, MaxInflightTokens: 200000}}
	cfg.Auth.APIKeys = []string{"cache-key"}
	cfg.RateLimit.Enabled = false
	cfg.Redis.KeyPrefix = keyPrefix

	pool, err := backend.NewPool(cfg.Backends)
	if err != nil {
		t.Fatalf("NewPool: %v", err)
	}
	est := tokenizer.NewCharHeuristic()
	rt, err := router.New(cfg.Routing, est)
	if err != nil {
		t.Fatalf("router.New: %v", err)
	}

	exact := cache.NewExactCache(rdb, cache.ExactOptions{
		TTL: time.Minute, KeyPrefix: keyPrefix,
	})

	// The deterministic embedder keeps these tests independent of the Python
	// sidecar while still exercising the real Redis vector-search path.
	semantic := cache.NewSemanticCache(rdb, cache.SemanticOptions{
		Embedder:  cache.StaticEmbedder{Dims: 64},
		IndexName: "idx_" + keyPrefix,
		KeyPrefix: keyPrefix,
		Threshold: semanticThreshold,
		TopK:      3,
		TTL:       time.Minute,
	})
	if err := semantic.EnsureIndex(t.Context()); err != nil {
		t.Skipf("vector index unavailable (Redis Stack required): %v", err)
	}
	t.Cleanup(func() { _ = semantic.DropIndex(t.Context()) })

	responseCache := cache.New(cache.Options{
		Exact: exact, Semantic: semantic, Logger: quietLogger(),
	})

	proxy := api.NewProxy(&http.Client{}, api.Timeouts{TTFT: 5 * time.Second}, quietLogger())
	srv := api.NewServer(api.Options{
		Config: cfg, Pool: pool, Router: rt, Proxy: proxy,
		Dispatcher: api.NewDispatcher(api.DispatcherOptions{
			Pool: pool, Router: rt, Proxy: proxy, MaxAttempts: 1, Logger: quietLogger(),
		}),
		Admission: admission.New(admission.Options{MaxConcurrent: 64, MaxQueue: 256}),
		Cache:     responseCache,
		Estimator: est,
		Logger:    quietLogger(),
		Version:   "cache-test",
	})

	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("listen: %v", err)
	}
	httpSrv := &http.Server{Handler: srv.Handler(), ReadHeaderTimeout: 5 * time.Second}
	go func() { _ = httpSrv.Serve(ln) }()
	t.Cleanup(func() { _ = httpSrv.Close() })

	return &cacheStack{
		url:      "http://" + ln.Addr().String(),
		client:   &http.Client{Timeout: 20 * time.Second},
		apiKey:   "cache-key",
		engine:   engine,
		cache:    responseCache,
		semantic: semantic,
	}
}

// ask sends a greedy (cacheable) request and returns status, headers and body.
func (cs *cacheStack) ask(t *testing.T, prompt string, stream bool, headers map[string]string) (*http.Response, string) {
	t.Helper()

	zero := 0.0
	payload, err := json.Marshal(openai.ChatRequest{
		Model:       "mock-7b",
		Stream:      stream,
		Temperature: &zero,
		Messages:    []openai.Message{{Role: openai.RoleUser, Content: prompt}},
	})
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}

	req, err := http.NewRequest(http.MethodPost, cs.url+"/v1/chat/completions", bytesReaderOf(payload))
	if err != nil {
		t.Fatalf("new request: %v", err)
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+cs.apiKey)
	for k, v := range headers {
		req.Header.Set(k, v)
	}

	resp, err := cs.client.Do(req)
	if err != nil {
		t.Fatalf("request: %v", err)
	}
	t.Cleanup(func() { _ = resp.Body.Close() })

	if !stream {
		var out openai.ChatResponse
		if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
			return resp, ""
		}
		if len(out.Choices) == 0 {
			return resp, ""
		}
		return resp, out.Choices[0].Message.Content
	}

	content, _, _ := readStream(t, resp)
	return resp, content
}

// waitForCacheWrite polls until the asynchronous cache store lands.
func (cs *cacheStack) waitForCacheWrite(t *testing.T, minStores uint64) {
	t.Helper()
	deadline := time.Now().Add(3 * time.Second)
	for time.Now().Before(deadline) {
		if cs.cache.Stats().Stores >= minStores {
			return
		}
		time.Sleep(10 * time.Millisecond)
	}
	t.Fatalf("cache write never completed (stats: %+v)", cs.cache.Stats())
}

func TestExactCacheHitThroughGateway(t *testing.T) {
	cs := newCacheStack(t, 0.99)

	resp1, body1 := cs.ask(t, "what is a KV cache", false, nil)
	if resp1.StatusCode != http.StatusOK {
		t.Fatalf("status = %d", resp1.StatusCode)
	}
	if got := resp1.Header.Get("X-Waypoint-Cache"); got != cache.SourceMiss {
		t.Errorf("first request cache header = %q, want miss", got)
	}
	cs.waitForCacheWrite(t, 1)

	requestsBefore := cs.engine.Stats().Requests

	resp2, body2 := cs.ask(t, "what is a KV cache", false, nil)
	if got := resp2.Header.Get("X-Waypoint-Cache"); got != cache.SourceExact {
		t.Errorf("second request cache header = %q, want exact", got)
	}
	if body1 != body2 {
		t.Errorf("cached body differs from the original:\n%q\n%q", body1, body2)
	}

	// A cache hit must not reach the backend at all.
	if after := cs.engine.Stats().Requests; after != requestsBefore {
		t.Errorf("backend served %d extra requests on a cache hit", after-requestsBefore)
	}
}

// A cached response must be measurably faster, which is the entire point.
func TestCacheHitIsFaster(t *testing.T) {
	cs := newCacheStack(t, 0.99)

	start := time.Now()
	cs.ask(t, "measure my latency please", false, nil)
	live := time.Since(start)
	cs.waitForCacheWrite(t, 1)

	start = time.Now()
	resp, _ := cs.ask(t, "measure my latency please", false, nil)
	cached := time.Since(start)

	if resp.Header.Get("X-Waypoint-Cache") != cache.SourceExact {
		t.Fatal("second request was not a cache hit")
	}
	t.Logf("live=%s cached=%s (%.1fx faster)", live, cached, float64(live)/float64(cached))

	if cached >= live {
		t.Errorf("cached response (%s) was not faster than live (%s)", cached, live)
	}
}

// A cached streaming response must replay with the original chunk boundaries, so
// a client that renders incrementally cannot tell the difference.
func TestCachedStreamReplaysChunks(t *testing.T) {
	cs := newCacheStack(t, 0.99)

	resp1, live := cs.ask(t, "stream and cache me", true, nil)
	if resp1.StatusCode != http.StatusOK {
		t.Fatalf("status = %d", resp1.StatusCode)
	}
	cs.waitForCacheWrite(t, 1)

	resp2 := cs.askStreamCounting(t, "stream and cache me")
	if resp2.source != cache.SourceExact {
		t.Errorf("cache header = %q, want exact", resp2.source)
	}
	if resp2.content != live {
		t.Errorf("replayed content differs:\nlive:   %q\ncached: %q", live, resp2.content)
	}
	// 10 output tokens were generated, so the replay should produce 10 chunks.
	if resp2.chunks != 10 {
		t.Errorf("replayed %d chunks, want 10: chunk boundaries were lost", resp2.chunks)
	}
}

type streamResult struct {
	content string
	chunks  int
	source  string
}

func (cs *cacheStack) askStreamCounting(t *testing.T, prompt string) streamResult {
	t.Helper()

	zero := 0.0
	payload, err := json.Marshal(openai.ChatRequest{
		Model:       "mock-7b",
		Stream:      true,
		Temperature: &zero,
		Messages:    []openai.Message{{Role: openai.RoleUser, Content: prompt}},
	})
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}

	req, _ := http.NewRequest(http.MethodPost, cs.url+"/v1/chat/completions", bytesReaderOf(payload))
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+cs.apiKey)

	resp, err := cs.client.Do(req)
	if err != nil {
		t.Fatalf("request: %v", err)
	}
	defer func() { _ = resp.Body.Close() }()

	content, chunks, _ := readStream(t, resp)
	return streamResult{content: content, chunks: chunks, source: resp.Header.Get("X-Waypoint-Cache")}
}

// A response cached from a streaming request must be servable to a
// non-streaming one and vice versa: the content is identical, only framing
// differs.
func TestCacheSharedBetweenStreamingAndBuffered(t *testing.T) {
	cs := newCacheStack(t, 0.99)

	_, streamed := cs.ask(t, "framing independence", true, nil)
	cs.waitForCacheWrite(t, 1)

	resp, buffered := cs.ask(t, "framing independence", false, nil)
	if got := resp.Header.Get("X-Waypoint-Cache"); got != cache.SourceExact {
		t.Errorf("cache header = %q, want exact: streamed and buffered forms should share an entry", got)
	}
	if streamed != buffered {
		t.Errorf("content differs across framing:\nstream: %q\nbuffer: %q", streamed, buffered)
	}
}

func TestCacheBypassHeader(t *testing.T) {
	cs := newCacheStack(t, 0.99)

	cs.ask(t, "bypass me", false, nil)
	cs.waitForCacheWrite(t, 1)

	before := cs.engine.Stats().Requests
	resp, _ := cs.ask(t, "bypass me", false, map[string]string{"X-Waypoint-Cache": "bypass"})

	if got := resp.Header.Get("X-Waypoint-Cache"); got != cache.SourceBypass {
		t.Errorf("cache header = %q, want bypass", got)
	}
	if after := cs.engine.Stats().Requests; after <= before {
		t.Error("bypass request did not reach the backend")
	}
}

// Sampled requests must never be cached: serving a cached answer would remove
// the variation the caller explicitly asked for.
func TestSampledRequestsAreNotCached(t *testing.T) {
	cs := newCacheStack(t, 0.99)

	hot := 0.8
	send := func() *http.Response {
		payload, err := json.Marshal(openai.ChatRequest{
			Model:       "mock-7b",
			Temperature: &hot,
			Messages:    []openai.Message{{Role: openai.RoleUser, Content: "sample me"}},
		})
		if err != nil {
			t.Fatalf("marshal: %v", err)
		}
		req, _ := http.NewRequest(http.MethodPost, cs.url+"/v1/chat/completions", bytesReaderOf(payload))
		req.Header.Set("Content-Type", "application/json")
		req.Header.Set("Authorization", "Bearer "+cs.apiKey)

		resp, err := cs.client.Do(req)
		if err != nil {
			t.Fatalf("request: %v", err)
		}
		t.Cleanup(func() { _ = resp.Body.Close() })
		return resp
	}

	send()
	before := cs.engine.Stats().Requests
	resp := send()

	if got := resp.Header.Get("X-Waypoint-Cache"); got == cache.SourceExact {
		t.Error("a temperature>0 request was served from cache")
	}
	if after := cs.engine.Stats().Requests; after <= before {
		t.Error("the second sampled request did not reach the backend")
	}
}

func TestSemanticCacheHitThroughGateway(t *testing.T) {
	// A permissive threshold so the bag-of-words embedder produces a hit on a
	// reworded question.
	cs := newCacheStack(t, 0.6)

	cs.ask(t, "how does prefix caching work in inference engines", false, nil)
	cs.waitForCacheWrite(t, 1)

	before := cs.engine.Stats().Requests
	resp, body := cs.ask(t, "how does prefix caching work", false, nil)

	source := resp.Header.Get("X-Waypoint-Cache")
	t.Logf("reworded question served from: %s (similarity header %q)",
		source, resp.Header.Get("X-Waypoint-Cache-Similarity"))

	if source != cache.SourceSemantic {
		t.Errorf("cache header = %q, want semantic", source)
	}
	if body == "" {
		t.Error("semantic hit returned empty content")
	}
	if after := cs.engine.Stats().Requests; after != before {
		t.Error("a semantic cache hit still reached the backend")
	}
	if resp.Header.Get("X-Waypoint-Cache-Similarity") == "" {
		t.Error("semantic hit did not report its similarity score")
	}
}

// Raising the threshold must turn borderline hits back into misses, which is
// the knob the M8 quality experiment sweeps.
func TestSemanticThresholdControlsHitRate(t *testing.T) {
	cs := newCacheStack(t, 0.6)

	cs.ask(t, "how does prefix caching work in inference engines", false, nil)
	cs.waitForCacheWrite(t, 1)

	resp, _ := cs.ask(t, "how does prefix caching work", false, nil)
	if resp.Header.Get("X-Waypoint-Cache") != cache.SourceSemantic {
		t.Fatal("expected a semantic hit at a 0.6 threshold")
	}

	cs.semantic.SetThreshold(0.999)

	resp2, _ := cs.ask(t, "how does prefix caching work differently", false, nil)
	if got := resp2.Header.Get("X-Waypoint-Cache"); got == cache.SourceSemantic {
		t.Errorf("still a semantic hit at a 0.999 threshold (header %q)", got)
	}
}

func TestCacheStatsExposedInAdmin(t *testing.T) {
	cs := newCacheStack(t, 0.99)

	cs.ask(t, "stats please", false, nil)
	cs.waitForCacheWrite(t, 1)
	cs.ask(t, "stats please", false, nil)

	resp, err := cs.client.Get(cs.url + "/admin/backends")
	if err != nil {
		t.Fatalf("GET /admin/backends: %v", err)
	}
	defer func() { _ = resp.Body.Close() }()

	var body struct {
		Cache cache.Stats `json:"cache"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&body); err != nil {
		t.Fatalf("decode: %v", err)
	}

	if body.Cache.ExactHits < 1 {
		t.Errorf("ExactHits = %d, want >= 1", body.Cache.ExactHits)
	}
	if body.Cache.HitRate <= 0 {
		t.Errorf("HitRate = %v, want > 0", body.Cache.HitRate)
	}
}

// Different conversations whose final question is identical must not share a
// cached answer.
func TestCacheSeparatesConversationContexts(t *testing.T) {
	cs := newCacheStack(t, 0.5)

	askWithSystem := func(system, question string) *http.Response {
		zero := 0.0
		payload, err := json.Marshal(openai.ChatRequest{
			Model:       "mock-7b",
			Temperature: &zero,
			Messages: []openai.Message{
				{Role: openai.RoleSystem, Content: system},
				{Role: openai.RoleUser, Content: question},
			},
		})
		if err != nil {
			t.Fatalf("marshal: %v", err)
		}
		req, _ := http.NewRequest(http.MethodPost, cs.url+"/v1/chat/completions", bytesReaderOf(payload))
		req.Header.Set("Content-Type", "application/json")
		req.Header.Set("Authorization", "Bearer "+cs.apiKey)

		resp, err := cs.client.Do(req)
		if err != nil {
			t.Fatalf("request: %v", err)
		}
		t.Cleanup(func() { _ = resp.Body.Close() })
		return resp
	}

	askWithSystem("you advise on tax law", "what should I do?")
	cs.waitForCacheWrite(t, 1)

	resp := askWithSystem("you advise on surgery", "what should I do?")
	if got := resp.Header.Get("X-Waypoint-Cache"); got == cache.SourceExact || got == cache.SourceSemantic {
		t.Errorf("identical questions in different conversations shared a cache entry (%q)", got)
	}
}
