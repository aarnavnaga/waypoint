package mocksim

import (
	"context"
	"errors"
	"strings"
	"testing"
	"time"

	"github.com/aarnavnaga/waypoint/internal/openai"
)

func testConfig() Config {
	cfg := DefaultConfig()
	// Keep simulated time small so unit tests stay fast.
	cfg.Prefill.Base = Duration{0}
	cfg.Prefill.PerToken = Duration{time.Microsecond}
	cfg.Decode.TokensPerSecond = 100000
	cfg.Output.MinTokens = 8
	cfg.Output.MaxTokens = 8
	return cfg
}

func chatReq(content string) *openai.ChatRequest {
	return &openai.ChatRequest{
		Model:    "mock-7b",
		Messages: []openai.Message{{Role: openai.RoleUser, Content: content}},
	}
}

func TestEnginePlanCountsPromptTokens(t *testing.T) {
	e := NewEngine(testConfig())
	plan := e.Plan(chatReq("hello world this is a prompt"))

	if plan.PromptTokens == 0 {
		t.Fatal("PromptTokens = 0, want > 0")
	}
	if plan.OutputTokens != 8 {
		t.Errorf("OutputTokens = %d, want 8", plan.OutputTokens)
	}
	if len(plan.Text) != plan.OutputTokens {
		t.Errorf("len(Text) = %d, want %d", len(plan.Text), plan.OutputTokens)
	}
}

// The headline simulated behavior: a repeated prompt skips prefill.
func TestEnginePrefixCacheReducesPrefill(t *testing.T) {
	cfg := testConfig()
	cfg.Prefill.PerToken = Duration{time.Millisecond}
	e := NewEngine(cfg)

	long := strings.Repeat("the quick brown fox jumps over the lazy dog ", 20)

	cold := e.Plan(chatReq(long))
	warm := e.Plan(chatReq(long))

	if cold.CachedTokens != 0 {
		t.Errorf("cold CachedTokens = %d, want 0", cold.CachedTokens)
	}
	if warm.CachedTokens == 0 {
		t.Fatal("warm request got no cache benefit")
	}
	if warm.PrefillDuration >= cold.PrefillDuration {
		t.Errorf("warm prefill %s not faster than cold %s",
			warm.PrefillDuration, cold.PrefillDuration)
	}
	if warm.UncachedTokens() >= cold.UncachedTokens() {
		t.Errorf("warm uncached %d not below cold %d",
			warm.UncachedTokens(), cold.UncachedTokens())
	}
}

func TestEngineSharedPrefixAcrossDifferentQuestions(t *testing.T) {
	e := NewEngine(testConfig())
	system := strings.Repeat("You are a careful assistant. ", 10)

	first := &openai.ChatRequest{Model: "m", Messages: []openai.Message{
		{Role: openai.RoleSystem, Content: system},
		{Role: openai.RoleUser, Content: "what is a KV cache?"},
	}}
	second := &openai.ChatRequest{Model: "m", Messages: []openai.Message{
		{Role: openai.RoleSystem, Content: system},
		{Role: openai.RoleUser, Content: "what is continuous batching?"},
	}}

	e.Plan(first)
	warm := e.Plan(second)

	if warm.CachedTokens == 0 {
		t.Error("shared system prompt produced no prefix reuse")
	}
}

func TestEngineDeterministicOutput(t *testing.T) {
	e1 := NewEngine(testConfig())
	e2 := NewEngine(testConfig())

	p1 := e1.Plan(chatReq("explain consistent hashing"))
	p2 := e2.Plan(chatReq("explain consistent hashing"))

	if strings.Join(p1.Text, "") != strings.Join(p2.Text, "") {
		t.Errorf("identical requests produced different text:\n%q\n%q",
			strings.Join(p1.Text, ""), strings.Join(p2.Text, ""))
	}
}

func TestEngineDifferentSeedsDifferentText(t *testing.T) {
	c1, c2 := testConfig(), testConfig()
	c1.Seed, c2.Seed = 1, 2

	t1 := strings.Join(NewEngine(c1).Plan(chatReq("same prompt")).Text, "")
	t2 := strings.Join(NewEngine(c2).Plan(chatReq("same prompt")).Text, "")

	if t1 == t2 {
		t.Error("different server seeds produced identical text")
	}
}

func TestEngineRespectsMaxTokens(t *testing.T) {
	cfg := testConfig()
	cfg.Output.MinTokens, cfg.Output.MaxTokens = 100, 200
	e := NewEngine(cfg)

	limit := 5
	req := chatReq("short answer please")
	req.MaxTokens = &limit

	if plan := e.Plan(req); plan.OutputTokens != 5 {
		t.Errorf("OutputTokens = %d, want 5 (max_tokens cap)", plan.OutputTokens)
	}
}

func TestEngineExecuteEmitsAllTokens(t *testing.T) {
	e := NewEngine(testConfig())
	plan := e.Plan(chatReq("hello"))

	var got []string
	err := e.Execute(context.Background(), plan, func(_ int, frag string) error {
		got = append(got, frag)
		return nil
	})
	if err != nil {
		t.Fatalf("Execute: %v", err)
	}
	if len(got) != plan.OutputTokens {
		t.Errorf("emitted %d tokens, want %d", len(got), plan.OutputTokens)
	}
}

func TestEngineExecuteStopsOnEmitError(t *testing.T) {
	e := NewEngine(testConfig())
	plan := e.Plan(chatReq("hello"))

	sentinel := errors.New("client gone")
	count := 0
	err := e.Execute(context.Background(), plan, func(_ int, _ string) error {
		count++
		if count == 3 {
			return sentinel
		}
		return nil
	})

	if !errors.Is(err, sentinel) {
		t.Errorf("Execute error = %v, want sentinel", err)
	}
	if count != 3 {
		t.Errorf("emitted %d tokens after error, want 3", count)
	}
}

func TestEngineExecuteHonorsContextCancellation(t *testing.T) {
	cfg := testConfig()
	cfg.Decode.TokensPerSecond = 20 // 50ms per token
	cfg.Output.MinTokens, cfg.Output.MaxTokens = 100, 100
	e := NewEngine(cfg)

	plan := e.Plan(chatReq("long answer"))
	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Millisecond)
	defer cancel()

	start := time.Now()
	err := e.Execute(ctx, plan, func(_ int, _ string) error { return nil })
	elapsed := time.Since(start)

	if err == nil {
		t.Error("Execute returned nil despite cancellation")
	}
	if elapsed > time.Second {
		t.Errorf("Execute took %s; it ignored cancellation", elapsed)
	}
}

func TestEngineExecuteReleasesSlot(t *testing.T) {
	cfg := testConfig()
	cfg.Batching.MaxBatch = 1
	e := NewEngine(cfg)

	plan := e.Plan(chatReq("hi"))
	for i := 0; i < 3; i++ {
		if err := e.Execute(context.Background(), plan, func(int, string) error { return nil }); err != nil {
			t.Fatalf("Execute %d: %v", i, err)
		}
	}
	if got := e.sched.BatchSize(); got != 0 {
		t.Errorf("BatchSize = %d after sequential executes, want 0", got)
	}
}

func TestEngineStatsAccumulate(t *testing.T) {
	e := NewEngine(testConfig())
	e.Plan(chatReq("first request"))
	e.Plan(chatReq("second request"))

	s := e.Stats()
	if s.Requests != 2 {
		t.Errorf("Requests = %d, want 2", s.Requests)
	}
	if s.PromptToks == 0 {
		t.Error("PromptToks = 0")
	}
	if s.OutputToks != 16 {
		t.Errorf("OutputToks = %d, want 16", s.OutputToks)
	}
}

func TestEngineReset(t *testing.T) {
	e := NewEngine(testConfig())
	long := strings.Repeat("reset me ", 30)
	e.Plan(chatReq(long))
	e.Reset()

	if s := e.Stats(); s.Requests != 0 {
		t.Errorf("Requests = %d after Reset, want 0", s.Requests)
	}
	if plan := e.Plan(chatReq(long)); plan.CachedTokens != 0 {
		t.Errorf("CachedTokens = %d after Reset, want 0", plan.CachedTokens)
	}
}

func TestFaultsDecodeMultiplier(t *testing.T) {
	f := NewFaults(FaultState{}, 1)
	if got := f.DecodeMultiplier(); got != 1.0 {
		t.Errorf("default DecodeMultiplier = %v, want 1.0", got)
	}

	f.Set(FaultState{SlowTokens: 3})
	if got := f.DecodeMultiplier(); got != 3 {
		t.Errorf("DecodeMultiplier = %v, want 3", got)
	}
}

func TestFaultsErrorRateBounds(t *testing.T) {
	f := NewFaults(FaultState{}, 1)

	f.Set(FaultState{ErrorRate: -5})
	if f.ShouldError() {
		t.Error("negative error rate produced an error")
	}

	f.Set(FaultState{ErrorRate: 99})
	if !f.ShouldError() {
		t.Error("error rate above 1 should always error")
	}

	f.Clear()
	if f.ShouldError() {
		t.Error("Clear did not disable error injection")
	}
}

func TestFaultsErrorRateIsApproximatelyHonored(t *testing.T) {
	f := NewFaults(FaultState{ErrorRate: 0.3}, 42)

	errorCount := 0
	const trials = 2000
	for i := 0; i < trials; i++ {
		if f.ShouldError() {
			errorCount++
		}
	}

	rate := float64(errorCount) / trials
	if rate < 0.25 || rate > 0.35 {
		t.Errorf("observed error rate %.3f, want ~0.30", rate)
	}
}

func TestConfigValidate(t *testing.T) {
	tests := []struct {
		name    string
		mutate  func(*Config)
		wantErr bool
	}{
		{"default valid", func(*Config) {}, false},
		{"no name", func(c *Config) { c.Name = "" }, true},
		{"no models", func(c *Config) { c.Models = nil }, true},
		{"zero tps", func(c *Config) { c.Decode.TokensPerSecond = 0 }, true},
		{"negative alpha", func(c *Config) { c.Decode.Alpha = -1 }, true},
		{"zero batch", func(c *Config) { c.Batching.MaxBatch = 0 }, true},
		{"zero block size", func(c *Config) { c.PrefixCache.BlockSize = 0 }, true},
		{"max below min tokens", func(c *Config) { c.Output.MaxTokens = 1; c.Output.MinTokens = 10 }, true},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			cfg := DefaultConfig()
			tc.mutate(&cfg)
			err := cfg.Validate()
			if tc.wantErr != (err != nil) {
				t.Errorf("Validate() error = %v, wantErr %v", err, tc.wantErr)
			}
		})
	}
}

func TestDecodeInterval(t *testing.T) {
	if got := (Decode{TokensPerSecond: 100}).Interval(); got != 10*time.Millisecond {
		t.Errorf("Interval = %s, want 10ms", got)
	}
	if got := (Decode{TokensPerSecond: 0}).Interval(); got != 0 {
		t.Errorf("Interval for zero tps = %s, want 0", got)
	}
}

// TTFT must include prefill, otherwise prefix-cache savings are invisible to any
// client-side measurement and the routing benchmark measures nothing.
func TestAdmitBlocksForPrefillBeforeDecoding(t *testing.T) {
	cfg := testConfig()
	cfg.Prefill.Base = Duration{80 * time.Millisecond}
	e := NewEngine(cfg)

	plan := e.Plan(chatReq("hi"))

	start := time.Now()
	release, err := e.Admit(context.Background(), plan)
	elapsed := time.Since(start)
	if err != nil {
		t.Fatalf("Admit: %v", err)
	}
	defer release()

	if elapsed < 70*time.Millisecond {
		t.Errorf("Admit returned after %s, want >= ~80ms of prefill", elapsed)
	}
}

func TestAdmitReleasesSlotOnCancellation(t *testing.T) {
	cfg := testConfig()
	cfg.Prefill.Base = Duration{time.Second}
	cfg.Batching.MaxBatch = 1
	e := NewEngine(cfg)

	plan := e.Plan(chatReq("hi"))

	ctx, cancel := context.WithTimeout(context.Background(), 20*time.Millisecond)
	defer cancel()

	if _, err := e.Admit(ctx, plan); err == nil {
		t.Fatal("Admit returned nil error despite cancellation")
	}
	// If the slot leaked, the batch would still be occupied.
	if got := e.sched.BatchSize(); got != 0 {
		t.Errorf("BatchSize = %d after cancelled Admit, want 0 (slot leak)", got)
	}
}

func TestDecodeRequiresNoPrefillWait(t *testing.T) {
	cfg := testConfig()
	cfg.Prefill.Base = Duration{500 * time.Millisecond}
	e := NewEngine(cfg)

	plan := e.Plan(chatReq("hi"))
	plan.PrefillDuration = 0 // already paid by Admit in the real path

	start := time.Now()
	if err := e.Decode(context.Background(), plan, func(int, string) error { return nil }); err != nil {
		t.Fatalf("Decode: %v", err)
	}
	if elapsed := time.Since(start); elapsed > 200*time.Millisecond {
		t.Errorf("Decode took %s; it should not re-pay prefill", elapsed)
	}
}
