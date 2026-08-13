package tokenizer

import (
	"math"
	"strings"
	"testing"
)

func TestSplitIsDeterministic(t *testing.T) {
	text := "You are a helpful assistant. Answer concisely!\nUse examples."
	first := Split(text)
	for i := 0; i < 50; i++ {
		if got := Split(text); !equal(got, first) {
			t.Fatalf("Split is not deterministic: run %d gave %q, first gave %q", i, got, first)
		}
	}
}

func TestSplitRules(t *testing.T) {
	tests := []struct {
		name string
		in   string
		want []string
	}{
		{"empty", "", nil},
		{"single word", "hello", []string{"hello"}},
		{"leading space attaches", "hello world", []string{"hello", " world"}},
		{"punctuation is separate", "hi!", []string{"hi", "!"}},
		{"newline is its own token", "a\nb", []string{"a", "\n", "b"}},
		{"long word is chopped", "antidisestablishment", []string{"antidi", "sestab", "lishme", "nt"}},
		{"long word keeps leading space on first piece", " antidisestab",
			[]string{" antidi", "sestab"}},
		{"digits are word runes", "gpt4 turbo", []string{"gpt4", " turbo"}},
		{"double space", "a  b", []string{"a", " ", " b"}},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got := Split(tc.in)
			if !equal(got, tc.want) {
				t.Errorf("Split(%q) = %q, want %q", tc.in, got, tc.want)
			}
		})
	}
}

func TestSplitRoundTrips(t *testing.T) {
	// Concatenating tokens must reproduce the input exactly, otherwise prefix
	// blocks would not correspond to real prompt content.
	inputs := []string{
		"You are a helpful assistant.",
		"Multi\nline\ttext with  spacing",
		"unicode: café — naïve 日本語",
		"antidisestablishmentarianism and supercalifragilistic",
	}
	for _, in := range inputs {
		if got := strings.Join(Split(in), ""); got != in {
			t.Errorf("round trip failed:\n got %q\nwant %q", got, in)
		}
	}
}

func TestCharHeuristicCount(t *testing.T) {
	h := NewCharHeuristic()

	if got := h.Count(""); got != 0 {
		t.Errorf("Count(\"\") = %d, want 0", got)
	}
	if got := h.Count("a"); got != 1 {
		t.Errorf("Count(\"a\") = %d, want 1 (never round down to zero)", got)
	}
	if got := h.Count(strings.Repeat("a", 400)); got != 100 {
		t.Errorf("Count(400 bytes) = %d, want 100", got)
	}
}

func TestCharHeuristicZeroRatioFallsBack(t *testing.T) {
	h := CharHeuristic{BytesPerToken: 0}
	if got := h.Count(strings.Repeat("a", 400)); got != 100 {
		t.Errorf("Count with zero ratio = %d, want 100 via default ratio", got)
	}
}

// TestHeuristicAgreesWithSplit measures the error the fast path accepts.
// This is the quantitative basis for DESIGN.md D5 — if this bound regresses,
// routing and admission decisions are being made on worse information.
func TestHeuristicAgreesWithSplit(t *testing.T) {
	corpus := []string{
		"You are a helpful assistant that answers questions about distributed systems.",
		"Explain how a KV cache works in a transformer, and why prefix reuse matters.",
		"Summarize the following incident report in three bullet points for an exec.",
		"def handler(req): return {'ok': True, 'latency_ms': compute(req)}",
		strings.Repeat("The quick brown fox jumps over the lazy dog. ", 20),
	}

	h := NewCharHeuristic()
	var totalHeuristic, totalSplit int
	for _, text := range corpus {
		totalHeuristic += h.Count(text)
		totalSplit += len(Split(text))
	}

	relErr := math.Abs(float64(totalHeuristic-totalSplit)) / float64(totalSplit)
	t.Logf("heuristic=%d split=%d relative error=%.1f%%", totalHeuristic, totalSplit, relErr*100)

	if relErr > 0.35 {
		t.Errorf("heuristic drifted %.1f%% from Split, above the 35%% bound documented in DESIGN.md D5", relErr*100)
	}
}

func TestEstimatorsImplementInterface(t *testing.T) {
	var _ Estimator = NewCharHeuristic()
	var _ Estimator = SplitEstimator{}

	if NewCharHeuristic().Name() != "char_heuristic" {
		t.Error("unexpected CharHeuristic name")
	}
	if (SplitEstimator{}).Name() != "split" {
		t.Error("unexpected SplitEstimator name")
	}
	if got := (SplitEstimator{}).Count("hello world"); got != 2 {
		t.Errorf("SplitEstimator.Count = %d, want 2", got)
	}
}

func BenchmarkCharHeuristic(b *testing.B) {
	text := strings.Repeat("The quick brown fox jumps over the lazy dog. ", 50)
	h := NewCharHeuristic()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = h.Count(text)
	}
}

func BenchmarkSplit(b *testing.B) {
	text := strings.Repeat("The quick brown fox jumps over the lazy dog. ", 50)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = Split(text)
	}
}

func equal(a, b []string) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}
