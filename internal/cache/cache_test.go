package cache

import (
	"context"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/redis/go-redis/v9"

	"github.com/aarnavnaga/waypoint/internal/openai"
)

func quietLogger() *slog.Logger {
	return slog.New(slog.NewTextHandler(io.Discard, &slog.HandlerOptions{Level: slog.LevelError + 1}))
}

func redisAddr() string {
	if v := os.Getenv("WAYPOINT_REDIS_ADDR"); v != "" {
		return v
	}
	return "localhost:6379"
}

func newRedis(t *testing.T) *redis.Client {
	t.Helper()

	client := redis.NewClient(&redis.Options{
		Addr:        redisAddr(),
		DialTimeout: time.Second,
		ReadTimeout: 2 * time.Second,
	})

	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()
	if err := client.Ping(ctx).Err(); err != nil {
		_ = client.Close()
		t.Skipf("Redis not reachable at %s (%v); start it with:\n"+
			"  docker run -d --name waypoint-redis -p 6379:6379 redis/redis-stack-server:7.4.0-v1",
			redisAddr(), err)
	}

	t.Cleanup(func() { _ = client.Close() })
	return client
}

func prefix(t *testing.T) string {
	t.Helper()
	return fmt.Sprintf("ctest-%d", time.Now().UnixNano())
}

func chatReq(content string) *openai.ChatRequest {
	zero := 0.0
	return &openai.ChatRequest{
		Model:       "mock-7b",
		Temperature: &zero,
		Messages:    []openai.Message{{Role: openai.RoleUser, Content: content}},
	}
}

func sampleEntry(content string) *Entry {
	return &Entry{
		Model:            "mock-7b",
		Content:          content,
		Chunks:           strings.SplitAfter(content, " "),
		PromptTokens:     10,
		CompletionTokens: 5,
		Backend:          "mock-1",
		CreatedAt:        time.Now().UTC(),
	}
}

// ---------- keys and normalization ----------

func TestExactKeyIsStable(t *testing.T) {
	req := chatReq("what is a KV cache?")
	first := ExactKey(req)

	for i := 0; i < 20; i++ {
		if got := ExactKey(chatReq("what is a KV cache?")); got != first {
			t.Fatalf("ExactKey is not stable: %q vs %q", got, first)
		}
	}
}

func TestExactKeyDistinguishesContent(t *testing.T) {
	a := ExactKey(chatReq("question one"))
	b := ExactKey(chatReq("question two"))
	if a == b {
		t.Error("different prompts produced the same cache key")
	}
}

func TestExactKeyDistinguishesParams(t *testing.T) {
	base := chatReq("same question")
	withMax := chatReq("same question")
	n := 100
	withMax.MaxTokens = &n

	if ExactKey(base) == ExactKey(withMax) {
		t.Error("max_tokens did not affect the cache key")
	}

	withSeed := chatReq("same question")
	seed := 42
	withSeed.Seed = &seed
	if ExactKey(base) == ExactKey(withSeed) {
		t.Error("seed did not affect the cache key")
	}
}

func TestExactKeyDistinguishesModel(t *testing.T) {
	a := chatReq("same question")
	b := chatReq("same question")
	b.Model = "other-model"

	if ExactKey(a) == ExactKey(b) {
		t.Error("model did not affect the cache key")
	}
}

// Streaming and non-streaming forms of the same question must share an entry:
// the content is identical, only the framing differs.
func TestExactKeyIgnoresStream(t *testing.T) {
	a := chatReq("same question")
	b := chatReq("same question")
	b.Stream = true

	if ExactKey(a) != ExactKey(b) {
		t.Error("stream flag changed the cache key; streamed and buffered forms would not share an entry")
	}
}

func TestExactKeyNormalizesWhitespace(t *testing.T) {
	a := chatReq("  what is a KV cache?  ")
	b := chatReq("what is a KV cache?")

	if ExactKey(a) != ExactKey(b) {
		t.Error("surrounding whitespace produced a different key")
	}
}

// Normalization must stay conservative: a wrong cache hit is far worse than a
// missed one, so differences a model would answer differently must be preserved.
func TestExactKeyPreservesMeaningfulDifferences(t *testing.T) {
	pairs := [][2]string{
		{"What is a KV cache?", "what is a kv cache?"}, // case
		{"Explain X, briefly", "Explain X briefly"},    // punctuation
		{"a b", "a  b"}, // internal spacing
	}
	for _, pair := range pairs {
		if ExactKey(chatReq(pair[0])) == ExactKey(chatReq(pair[1])) {
			t.Errorf("over-normalized: %q and %q share a key", pair[0], pair[1])
		}
	}
}

// Caching a sampled response would silently remove the variation the caller
// asked for, so only greedy decoding is cacheable.
func TestCacheableRequiresZeroTemperature(t *testing.T) {
	zero, hot := 0.0, 0.7

	cold := chatReq("q")
	cold.Temperature = &zero
	if !Cacheable(cold) {
		t.Error("temperature 0 should be cacheable")
	}

	warm := chatReq("q")
	warm.Temperature = &hot
	if Cacheable(warm) {
		t.Error("temperature 0.7 must not be cacheable")
	}

	// An unset temperature defaults to 1.0 and is therefore not cacheable.
	unset := &openai.ChatRequest{
		Model:    "m",
		Messages: []openai.Message{{Role: openai.RoleUser, Content: "q"}},
	}
	if Cacheable(unset) {
		t.Error("an unspecified temperature defaults to 1.0 and must not be cacheable")
	}
}

func TestCacheableRejectsEmptyMessages(t *testing.T) {
	zero := 0.0
	if Cacheable(&openai.ChatRequest{Model: "m", Temperature: &zero}) {
		t.Error("a request with no messages must not be cacheable")
	}
}

func TestSemanticTextUsesFinalUserMessage(t *testing.T) {
	req := &openai.ChatRequest{
		Model: "m",
		Messages: []openai.Message{
			{Role: openai.RoleSystem, Content: "you are helpful"},
			{Role: openai.RoleUser, Content: "first question"},
			{Role: openai.RoleAssistant, Content: "first answer"},
			{Role: openai.RoleUser, Content: "the actual question"},
		},
	}

	text, ctxHash, ok := SemanticText(req)
	if !ok {
		t.Fatal("SemanticText reported failure on a valid conversation")
	}
	if text != "the actual question" {
		t.Errorf("text = %q, want the final user message", text)
	}
	if ctxHash == "" {
		t.Error("context hash is empty")
	}
}

// Identical final questions in different conversations must not collide.
func TestSemanticTextContextHashSeparatesConversations(t *testing.T) {
	makeReq := func(system string) *openai.ChatRequest {
		return &openai.ChatRequest{
			Model: "m",
			Messages: []openai.Message{
				{Role: openai.RoleSystem, Content: system},
				{Role: openai.RoleUser, Content: "what should I do?"},
			},
		}
	}

	_, hashA, _ := SemanticText(makeReq("you advise on taxes"))
	_, hashB, _ := SemanticText(makeReq("you advise on surgery"))

	if hashA == hashB {
		t.Error("different conversation contexts produced the same hash")
	}
}

func TestSemanticTextRejectsNoUserMessage(t *testing.T) {
	req := &openai.ChatRequest{
		Model:    "m",
		Messages: []openai.Message{{Role: openai.RoleSystem, Content: "only a system prompt"}},
	}
	if _, _, ok := SemanticText(req); ok {
		t.Error("SemanticText accepted a conversation with no user message")
	}
}

func TestParamsFingerprintDistinguishesSampling(t *testing.T) {
	base := chatReq("q")
	other := chatReq("q")
	p := 0.5
	other.TopP = &p

	if ParamsFingerprint(base) == ParamsFingerprint(other) {
		t.Error("top_p did not change the params fingerprint")
	}
}

func TestBypassRequested(t *testing.T) {
	tests := map[string]bool{
		"bypass":   true,
		"BYPASS":   true,
		" bypass ": true,
		"":         false,
		"refresh":  false,
	}
	for input, want := range tests {
		if got := BypassRequested(input); got != want {
			t.Errorf("BypassRequested(%q) = %v, want %v", input, got, want)
		}
	}
}

// ---------- exact cache ----------

func TestExactCacheRoundTrip(t *testing.T) {
	client := newRedis(t)
	c := NewExactCache(client, ExactOptions{TTL: time.Minute, KeyPrefix: prefix(t)})

	ctx := context.Background()
	key := ExactKey(chatReq("round trip"))
	entry := sampleEntry("the cached answer")

	if _, err := c.Get(ctx, key); !errors.Is(err, ErrMiss) {
		t.Fatalf("cold lookup error = %v, want ErrMiss", err)
	}

	if err := c.Set(ctx, key, entry); err != nil {
		t.Fatalf("Set: %v", err)
	}

	got, err := c.Get(ctx, key)
	if err != nil {
		t.Fatalf("Get: %v", err)
	}
	if got.Content != entry.Content {
		t.Errorf("Content = %q, want %q", got.Content, entry.Content)
	}
	if got.Source != SourceExact {
		t.Errorf("Source = %q, want %q", got.Source, SourceExact)
	}
	if len(got.Chunks) != len(entry.Chunks) {
		t.Errorf("chunk boundaries lost: got %d, want %d", len(got.Chunks), len(entry.Chunks))
	}
}

func TestExactCacheTTL(t *testing.T) {
	client := newRedis(t)
	c := NewExactCache(client, ExactOptions{TTL: 300 * time.Millisecond, KeyPrefix: prefix(t)})

	ctx := context.Background()
	key := "ttl-test"

	if err := c.Set(ctx, key, sampleEntry("expires soon")); err != nil {
		t.Fatalf("Set: %v", err)
	}
	if _, err := c.Get(ctx, key); err != nil {
		t.Fatalf("Get before expiry: %v", err)
	}

	time.Sleep(500 * time.Millisecond)

	if _, err := c.Get(ctx, key); !errors.Is(err, ErrMiss) {
		t.Errorf("entry survived its TTL (err = %v)", err)
	}
}

func TestExactCacheDelete(t *testing.T) {
	client := newRedis(t)
	c := NewExactCache(client, ExactOptions{TTL: time.Minute, KeyPrefix: prefix(t)})
	ctx := context.Background()

	if err := c.Set(ctx, "k", sampleEntry("answer")); err != nil {
		t.Fatalf("Set: %v", err)
	}
	if err := c.Delete(ctx, "k"); err != nil {
		t.Fatalf("Delete: %v", err)
	}
	if _, err := c.Get(ctx, "k"); !errors.Is(err, ErrMiss) {
		t.Error("entry survived deletion")
	}
}

// A corrupt entry must read as a miss, not an error: the request can still be
// served live and the bad key will expire.
func TestExactCacheCorruptEntryIsAMiss(t *testing.T) {
	client := newRedis(t)
	p := prefix(t)
	c := NewExactCache(client, ExactOptions{TTL: time.Minute, KeyPrefix: p})
	ctx := context.Background()

	if err := client.Set(ctx, p+":cache:exact:bad", "not json", time.Minute).Err(); err != nil {
		t.Fatalf("seed corrupt entry: %v", err)
	}

	if _, err := c.Get(ctx, "bad"); !errors.Is(err, ErrMiss) {
		t.Errorf("corrupt entry returned %v, want ErrMiss", err)
	}
}

// ---------- semantic cache ----------

func newSemanticCache(t *testing.T, threshold float64) (*SemanticCache, func()) {
	t.Helper()

	client := newRedis(t)
	p := prefix(t)
	c := NewSemanticCache(client, SemanticOptions{
		Embedder:  StaticEmbedder{Dims: 64},
		IndexName: "idx_" + p,
		KeyPrefix: p,
		Threshold: threshold,
		TopK:      3,
		TTL:       time.Minute,
	})

	ctx := context.Background()
	if err := c.EnsureIndex(ctx); err != nil {
		t.Skipf("vector index unavailable (Redis Stack required): %v", err)
	}
	return c, func() { _ = c.DropIndex(context.Background()) }
}

func TestSemanticCacheExactTextHits(t *testing.T) {
	c, cleanup := newSemanticCache(t, 0.9)
	defer cleanup()

	ctx := context.Background()
	const text = "how does prefix caching work"

	if err := c.Store(ctx, text, "mock-7b", "p1", "ctx1", sampleEntry("prefix answer")); err != nil {
		t.Fatalf("Store: %v", err)
	}

	got, err := c.Lookup(ctx, text, "mock-7b", "p1", "ctx1")
	if err != nil {
		t.Fatalf("Lookup: %v", err)
	}
	if got.Content != "prefix answer" {
		t.Errorf("Content = %q", got.Content)
	}
	if got.Source != SourceSemantic {
		t.Errorf("Source = %q, want semantic", got.Source)
	}
	if got.Similarity < 0.99 {
		t.Errorf("Similarity = %.3f for identical text, want ~1.0", got.Similarity)
	}
}

func TestSemanticCacheSimilarTextHits(t *testing.T) {
	c, cleanup := newSemanticCache(t, 0.6)
	defer cleanup()

	ctx := context.Background()

	if err := c.Store(ctx, "how does prefix caching work in vllm",
		"mock-7b", "p1", "ctx1", sampleEntry("prefix answer")); err != nil {
		t.Fatalf("Store: %v", err)
	}

	// Heavy word overlap, so the static embedder rates these as similar.
	got, err := c.Lookup(ctx, "how does prefix caching work", "mock-7b", "p1", "ctx1")
	if err != nil {
		t.Fatalf("Lookup: %v", err)
	}
	if got.Content != "prefix answer" {
		t.Errorf("Content = %q", got.Content)
	}
	t.Logf("similarity for a near-duplicate question: %.3f", got.Similarity)
}

func TestSemanticCacheRejectsBelowThreshold(t *testing.T) {
	c, cleanup := newSemanticCache(t, 0.99)
	defer cleanup()

	ctx := context.Background()
	if err := c.Store(ctx, "how does prefix caching work",
		"mock-7b", "p1", "ctx1", sampleEntry("prefix answer")); err != nil {
		t.Fatalf("Store: %v", err)
	}

	_, err := c.Lookup(ctx, "what is the airspeed velocity of a swallow",
		"mock-7b", "p1", "ctx1")
	if !errors.Is(err, ErrMiss) {
		t.Errorf("unrelated question returned %v, want ErrMiss", err)
	}
}

// Similarity alone is not a valid hit: the model and sampling parameters must
// match exactly, or a cached greedy answer could be served for a different
// decoding configuration.
func TestSemanticCacheRequiresMatchingParams(t *testing.T) {
	c, cleanup := newSemanticCache(t, 0.5)
	defer cleanup()

	ctx := context.Background()
	const text = "identical question text"

	if err := c.Store(ctx, text, "mock-7b", "params-A", "ctx1", sampleEntry("answer")); err != nil {
		t.Fatalf("Store: %v", err)
	}

	if _, err := c.Lookup(ctx, text, "mock-7b", "params-B", "ctx1"); !errors.Is(err, ErrMiss) {
		t.Error("a hit was returned across different generation parameters")
	}
	if _, err := c.Lookup(ctx, text, "other-model", "params-A", "ctx1"); !errors.Is(err, ErrMiss) {
		t.Error("a hit was returned across different models")
	}
	if _, err := c.Lookup(ctx, text, "mock-7b", "params-A", "ctx2"); !errors.Is(err, ErrMiss) {
		t.Error("a hit was returned across different conversation contexts")
	}
}

func TestSemanticCacheThresholdIsAdjustable(t *testing.T) {
	c, cleanup := newSemanticCache(t, 0.99)
	defer cleanup()

	ctx := context.Background()
	if err := c.Store(ctx, "how does prefix caching work in vllm engines",
		"mock-7b", "p1", "ctx1", sampleEntry("answer")); err != nil {
		t.Fatalf("Store: %v", err)
	}

	query := "how does prefix caching work"
	if _, err := c.Lookup(ctx, query, "mock-7b", "p1", "ctx1"); !errors.Is(err, ErrMiss) {
		t.Fatal("expected a miss at a 0.99 threshold")
	}

	c.SetThreshold(0.5)
	if _, err := c.Lookup(ctx, query, "mock-7b", "p1", "ctx1"); err != nil {
		t.Errorf("expected a hit after lowering the threshold, got %v", err)
	}

	// Out-of-range values are ignored rather than silently disabling the cache.
	c.SetThreshold(5)
	if got := c.Threshold(); got != 0.5 {
		t.Errorf("Threshold = %v after an invalid SetThreshold, want 0.5", got)
	}
}

// ---------- coordinator ----------

func TestCacheLookupPrefersExact(t *testing.T) {
	client := newRedis(t)
	p := prefix(t)

	exact := NewExactCache(client, ExactOptions{TTL: time.Minute, KeyPrefix: p})
	semantic, cleanup := newSemanticCache(t, 0.5)
	defer cleanup()

	c := New(Options{Exact: exact, Semantic: semantic, Logger: quietLogger()})
	ctx := context.Background()
	req := chatReq("which layer answers")

	c.Store(ctx, req, sampleEntry("from cache"))

	got, err := c.Lookup(ctx, req)
	if err != nil {
		t.Fatalf("Lookup: %v", err)
	}
	if got.Source != SourceExact {
		t.Errorf("Source = %q, want exact (it is cheaper and safer than semantic)", got.Source)
	}
}

func TestCacheFallsBackToSemantic(t *testing.T) {
	client := newRedis(t)
	p := prefix(t)

	exact := NewExactCache(client, ExactOptions{TTL: time.Minute, KeyPrefix: p})
	semantic, cleanup := newSemanticCache(t, 0.5)
	defer cleanup()

	c := New(Options{Exact: exact, Semantic: semantic, Logger: quietLogger()})
	ctx := context.Background()

	c.Store(ctx, chatReq("how does prefix caching work in engines"), sampleEntry("semantic answer"))

	// A different wording misses exact but should hit semantic.
	got, err := c.Lookup(ctx, chatReq("how does prefix caching work"))
	if err != nil {
		t.Fatalf("Lookup: %v", err)
	}
	if got.Source != SourceSemantic {
		t.Errorf("Source = %q, want semantic", got.Source)
	}
}

func TestCacheSkipsUncacheableRequests(t *testing.T) {
	client := newRedis(t)
	exact := NewExactCache(client, ExactOptions{TTL: time.Minute, KeyPrefix: prefix(t)})
	c := New(Options{Exact: exact, Logger: quietLogger()})

	hot := 0.8
	req := chatReq("sampled question")
	req.Temperature = &hot

	ctx := context.Background()
	c.Store(ctx, req, sampleEntry("should not be stored"))

	if _, err := c.Lookup(ctx, req); !errors.Is(err, ErrMiss) {
		t.Error("a sampled request was served from cache")
	}
}

func TestCacheSkipsEmptyContent(t *testing.T) {
	client := newRedis(t)
	exact := NewExactCache(client, ExactOptions{TTL: time.Minute, KeyPrefix: prefix(t)})
	c := New(Options{Exact: exact, Logger: quietLogger()})

	ctx := context.Background()
	req := chatReq("empty answer")
	c.Store(ctx, req, &Entry{Model: "m", Content: ""})

	if _, err := c.Lookup(ctx, req); !errors.Is(err, ErrMiss) {
		t.Error("an empty completion was cached")
	}
}

// Redis being unavailable must degrade to a miss, never an error that fails the
// request: a cache that can take down the service it accelerates is worse than
// no cache.
func TestCacheDegradesWhenRedisIsDown(t *testing.T) {
	dead := redis.NewClient(&redis.Options{Addr: "127.0.0.1:1", DialTimeout: 50 * time.Millisecond})
	defer func() { _ = dead.Close() }()

	exact := NewExactCache(dead, ExactOptions{TTL: time.Minute, Timeout: 100 * time.Millisecond})
	c := New(Options{Exact: exact, Logger: quietLogger()})

	ctx := context.Background()
	req := chatReq("unreachable")

	if _, err := c.Lookup(ctx, req); !errors.Is(err, ErrMiss) {
		t.Errorf("Lookup returned %v, want ErrMiss when Redis is down", err)
	}
	// Storing must not panic or block either.
	c.Store(ctx, req, sampleEntry("answer"))

	if s := c.Stats(); s.Errors == 0 {
		t.Error("cache errors were not counted")
	}
}

func TestCacheDisabled(t *testing.T) {
	c := New(Options{Logger: quietLogger()})
	if c.Enabled() {
		t.Error("a cache with no layers reports enabled")
	}
	if _, err := c.Lookup(context.Background(), chatReq("q")); !errors.Is(err, ErrMiss) {
		t.Error("a disabled cache returned a hit")
	}
}

func TestCacheStats(t *testing.T) {
	client := newRedis(t)
	exact := NewExactCache(client, ExactOptions{TTL: time.Minute, KeyPrefix: prefix(t)})
	c := New(Options{Exact: exact, Logger: quietLogger()})

	ctx := context.Background()
	req := chatReq("stats question")

	_, _ = c.Lookup(ctx, req) // miss
	c.Store(ctx, req, sampleEntry("answer"))
	_, _ = c.Lookup(ctx, req) // hit

	s := c.Stats()
	if s.ExactHits != 1 {
		t.Errorf("ExactHits = %d, want 1", s.ExactHits)
	}
	if s.ExactMisses != 1 {
		t.Errorf("ExactMisses = %d, want 1", s.ExactMisses)
	}
	if s.HitRate != 0.5 {
		t.Errorf("HitRate = %v, want 0.5", s.HitRate)
	}

	c.ResetStats()
	if got := c.Stats().ExactHits; got != 0 {
		t.Errorf("ExactHits = %d after reset", got)
	}
}

// ---------- embeddings ----------

func TestStaticEmbedderSimilarity(t *testing.T) {
	e := StaticEmbedder{Dims: 64}
	ctx := context.Background()

	a, _ := e.Embed(ctx, "how does prefix caching work")
	b, _ := e.Embed(ctx, "how does prefix caching work")
	cVec, _ := e.Embed(ctx, "what is the capital of france")

	if sim := CosineSimilarity(a, b); sim < 0.999 {
		t.Errorf("identical text similarity = %.4f, want ~1.0", sim)
	}
	if sim := CosineSimilarity(a, cVec); sim > 0.5 {
		t.Errorf("unrelated text similarity = %.4f, want low", sim)
	}
}

func TestCosineSimilarityEdgeCases(t *testing.T) {
	if got := CosineSimilarity(nil, nil); got != 0 {
		t.Errorf("CosineSimilarity(nil,nil) = %v, want 0", got)
	}
	if got := CosineSimilarity([]float32{1, 0}, []float32{1, 0, 0}); got != 0 {
		t.Errorf("mismatched lengths = %v, want 0", got)
	}
	if got := CosineSimilarity([]float32{0, 0}, []float32{0, 0}); got != 0 {
		t.Errorf("zero vectors = %v, want 0", got)
	}
}

func TestNormalizeProducesUnitVector(t *testing.T) {
	vec := normalize([]float32{3, 4})
	if got := CosineSimilarity(vec, []float32{3, 4}); got < 0.999 {
		t.Errorf("normalize changed direction (similarity %.4f)", got)
	}

	var sum float64
	for _, v := range vec {
		sum += float64(v) * float64(v)
	}
	if sum < 0.999 || sum > 1.001 {
		t.Errorf("normalized magnitude = %.4f, want 1.0", sum)
	}
}

func TestVectorToBytesLength(t *testing.T) {
	if got := len(vectorToBytes([]float32{1, 2, 3})); got != 12 {
		t.Errorf("vectorToBytes length = %d, want 12 (3 x float32)", got)
	}
}

func TestEmbedderErrorsOnBadSidecar(t *testing.T) {
	e := NewHTTPEmbedder(EmbedderOptions{
		BaseURL: "http://127.0.0.1:1", Dimensions: 384, Timeout: 100 * time.Millisecond,
	})
	if _, err := e.Embed(context.Background(), "text"); err == nil {
		t.Error("Embed succeeded against an unreachable sidecar")
	}
	if err := e.Ready(context.Background()); err == nil {
		t.Error("Ready succeeded against an unreachable sidecar")
	}
}

func TestEntryEncodeDecode(t *testing.T) {
	entry := sampleEntry("round trip content")

	data, err := entry.Encode()
	if err != nil {
		t.Fatalf("Encode: %v", err)
	}
	got, err := DecodeEntry(data)
	if err != nil {
		t.Fatalf("DecodeEntry: %v", err)
	}

	if got.Content != entry.Content || got.Backend != entry.Backend {
		t.Errorf("round trip mismatch: %+v", got)
	}
	if u := got.Usage(); u.TotalTokens != 15 {
		t.Errorf("TotalTokens = %d, want 15", u.TotalTokens)
	}
}

func TestDecodeEntryRejectsGarbage(t *testing.T) {
	if _, err := DecodeEntry([]byte("not json")); err == nil {
		t.Error("DecodeEntry accepted garbage")
	}
}
