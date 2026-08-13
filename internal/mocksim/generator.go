package mocksim

import (
	"encoding/binary"
	"hash/fnv"
	"math/rand"
	"strings"

	"github.com/aarnavnaga/waypoint/internal/openai"
)

// vocab is a fixed word list. Output text only needs to be plausible and
// *stable*; the benchmarks care about token counts and timing, and the semantic
// cache experiments need identical prompts to produce identical answers.
var vocab = []string{
	"the", "system", "request", "token", "cache", "latency", "throughput", "batch",
	"server", "routing", "policy", "prefix", "decode", "prefill", "queue", "replica",
	"gateway", "model", "stream", "response", "context", "window", "memory", "kernel",
	"attention", "sequence", "scheduler", "budget", "breaker", "retry", "timeout",
	"metric", "trace", "span", "bucket", "limit", "hash", "ring", "node", "weight",
	"because", "therefore", "however", "which", "means", "that", "consider", "given",
	"under", "load", "across", "between", "within", "without", "before", "after",
}

// Generator produces deterministic completion text.
//
// Determinism matters for three reasons: benchmark runs are comparable across
// policies, the semantic-cache quality experiment can compare a cached answer to
// the answer that *would* have been produced, and a flaky test cannot be blamed
// on changing output.
type Generator struct {
	serverSeed uint64
}

// NewGenerator returns a Generator whose output varies by server seed, so two
// backends produce distinguishable text for the same prompt.
func NewGenerator(serverSeed uint64) *Generator {
	return &Generator{serverSeed: serverSeed}
}

// PromptKey renders a request into the canonical string used for seeding and
// for prefix tokenization.
//
// Field order is fixed so the same logical request always produces the same key.
func PromptKey(req *openai.ChatRequest) string {
	var b strings.Builder
	b.Grow(128)
	b.WriteString(req.Model)
	b.WriteByte('\n')
	for _, m := range req.Messages {
		b.WriteString(m.Role)
		b.WriteByte(':')
		b.WriteString(m.Content)
		b.WriteByte('\n')
	}
	return b.String()
}

// seedFor derives a stable 64-bit seed from a key, the server seed, and any
// explicit seed in the request.
func (g *Generator) seedFor(key string, explicit *int) int64 {
	h := fnv.New64a()
	_, _ = h.Write([]byte(key))

	var scratch [8]byte
	binary.LittleEndian.PutUint64(scratch[:], g.serverSeed)
	_, _ = h.Write(scratch[:])

	if explicit != nil {
		binary.LittleEndian.PutUint64(scratch[:], uint64(*explicit))
		_, _ = h.Write(scratch[:])
	}
	// Mask the sign bit: rand.NewSource expects a non-negative seed to behave
	// predictably across platforms.
	return int64(h.Sum64() & 0x7fffffffffffffff)
}

// OutputLength picks a deterministic completion length within [minTokens,
// maxTokens], honoring an explicit max_tokens request cap.
func (g *Generator) OutputLength(req *openai.ChatRequest, minTokens, maxTokens int) int {
	if minTokens < 1 {
		minTokens = 1
	}
	if maxTokens < minTokens {
		maxTokens = minTokens
	}

	rng := rand.New(rand.NewSource(g.seedFor(PromptKey(req), req.Seed))) //nolint:gosec // simulation, not crypto
	n := minTokens
	if span := maxTokens - minTokens; span > 0 {
		n += rng.Intn(span + 1)
	}

	if req.MaxTokens != nil && *req.MaxTokens < n {
		n = *req.MaxTokens
	}
	if n < 1 {
		n = 1
	}
	return n
}

// Generate returns n deterministic text fragments. Each fragment is one emitted
// "token" and already carries its leading space where appropriate, so
// concatenating them yields readable prose.
func (g *Generator) Generate(req *openai.ChatRequest, n int) []string {
	if n < 1 {
		return nil
	}

	rng := rand.New(rand.NewSource(g.seedFor(PromptKey(req), req.Seed))) //nolint:gosec // simulation, not crypto
	out := make([]string, 0, n)

	for i := 0; i < n; i++ {
		word := vocab[rng.Intn(len(vocab))]
		switch {
		case i == 0:
			// Capitalize the opening word.
			out = append(out, strings.ToUpper(word[:1])+word[1:])
		case i == n-1:
			out = append(out, " "+word+".")
		case rng.Intn(11) == 0:
			out = append(out, " "+word+",")
		default:
			out = append(out, " "+word)
		}
	}
	return out
}
