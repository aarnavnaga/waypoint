// Package cache implements Waypoint's exact-match and semantic response caches.
package cache

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"strings"
	"time"

	"github.com/aarnavnaga/waypoint/internal/openai"
)

// Entry is a cached completion.
//
// Chunks preserve the original token boundaries so a cached response can be
// replayed with the same streaming granularity the client would have seen live.
// Storing only the joined string would make replays arrive as one giant chunk,
// which is observably different for any client that renders incrementally.
type Entry struct {
	Model            string    `json:"model"`
	Content          string    `json:"content"`
	Chunks           []string  `json:"chunks"`
	PromptTokens     int       `json:"prompt_tokens"`
	CompletionTokens int       `json:"completion_tokens"`
	Backend          string    `json:"backend"`
	CreatedAt        time.Time `json:"created_at"`
	// Source records which cache served this entry, for metrics and headers.
	Source string `json:"-"`
	// Similarity is set for semantic hits.
	Similarity float64 `json:"-"`
}

// Cache source labels, also used as X-Waypoint-Cache response values.
const (
	SourceExact    = "exact"
	SourceSemantic = "semantic"
	SourceMiss     = "miss"
	SourceBypass   = "bypass"
	SourceDisabled = "disabled"
)

// Encode serializes an entry for storage.
func (e *Entry) Encode() ([]byte, error) { return json.Marshal(e) }

// DecodeEntry parses a stored entry.
func DecodeEntry(data []byte) (*Entry, error) {
	var e Entry
	if err := json.Unmarshal(data, &e); err != nil {
		return nil, fmt.Errorf("cache: decode entry: %w", err)
	}
	return &e, nil
}

// Usage reconstructs token accounting for a response.
func (e *Entry) Usage() openai.Usage {
	return openai.Usage{
		PromptTokens:     e.PromptTokens,
		CompletionTokens: e.CompletionTokens,
		TotalTokens:      e.PromptTokens + e.CompletionTokens,
	}
}

// keyPayload is the canonical form hashed for the exact-match key.
//
// Field order is fixed by the struct definition and Go's encoder is
// deterministic for structs, so the same logical request always produces the
// same bytes. A map would not be safe here — Go randomizes map iteration order
// for *ranging*, and while encoding/json sorts map keys, relying on that is
// fragile compared to an explicit struct.
type keyPayload struct {
	Model       string           `json:"model"`
	Messages    []openai.Message `json:"messages"`
	Temperature float64          `json:"temperature"`
	TopP        *float64         `json:"top_p,omitempty"`
	MaxTokens   *int             `json:"max_tokens,omitempty"`
	Stop        []string         `json:"stop,omitempty"`
	Seed        *int             `json:"seed,omitempty"`
}

// ExactKey returns the cache key for a request: SHA-256 over the normalized
// request fields that affect the answer.
//
// `stream` is deliberately excluded. The same question asked with and without
// streaming must share one cache entry — the response content is identical, only
// its framing differs, and the stored chunks let either framing be reproduced.
func ExactKey(req *openai.ChatRequest) string {
	payload := keyPayload{
		Model:       req.Model,
		Messages:    normalizeMessages(req.Messages),
		Temperature: req.TemperatureOrDefault(),
		TopP:        req.TopP,
		MaxTokens:   req.MaxTokens,
		Stop:        req.Stop,
		Seed:        req.Seed,
	}

	data, err := json.Marshal(payload)
	if err != nil {
		// Marshaling these types cannot fail; fall back to something unique so
		// a hypothetical failure causes a cache miss rather than a collision.
		return "unhashable-" + req.Model
	}

	sum := sha256.Sum256(data)
	return hex.EncodeToString(sum[:])
}

// normalizeMessages trims incidental whitespace so that requests differing only
// in trailing spaces share a cache entry.
//
// Normalization is kept conservative on purpose: anything more aggressive
// (lowercasing, collapsing internal whitespace, stripping punctuation) risks
// conflating prompts that a model would answer differently, and a wrong cache
// hit is far worse than a missed one.
func normalizeMessages(msgs []openai.Message) []openai.Message {
	out := make([]openai.Message, len(msgs))
	for i, m := range msgs {
		out[i] = openai.Message{
			Role:    strings.TrimSpace(m.Role),
			Content: strings.TrimSpace(m.Content),
			Name:    m.Name,
		}
	}
	return out
}

// ParamsFingerprint identifies the generation parameters that must match for a
// *semantic* hit to be valid.
//
// Unlike the exact key this excludes message content — that is what the
// embedding compares — but two prompts with similar meaning are only
// interchangeable if they would be decoded the same way. A cached greedy answer
// is not a valid response to a request with different sampling settings.
func ParamsFingerprint(req *openai.ChatRequest) string {
	var b strings.Builder
	fmt.Fprintf(&b, "m=%s;t=%.4f", req.Model, req.TemperatureOrDefault())
	if req.TopP != nil {
		fmt.Fprintf(&b, ";p=%.4f", *req.TopP)
	}
	if req.MaxTokens != nil {
		fmt.Fprintf(&b, ";n=%d", *req.MaxTokens)
	}
	if len(req.Stop) > 0 {
		fmt.Fprintf(&b, ";s=%s", strings.Join(req.Stop, "\x00"))
	}

	sum := sha256.Sum256([]byte(b.String()))
	return hex.EncodeToString(sum[:16])
}

// Cacheable reports whether a request may be served from, or written to, cache.
//
// Only temperature 0 is cacheable. At any higher temperature the model is
// sampling, so two identical requests are *supposed* to produce different
// answers; serving a cached one silently removes the variation the caller asked
// for. This is a correctness rule, not an optimization.
func Cacheable(req *openai.ChatRequest) bool {
	if req.TemperatureOrDefault() != 0 {
		return false
	}
	if len(req.Messages) == 0 {
		return false
	}
	return true
}

// SemanticText builds the text embedded for similarity search.
//
// Only the final user message is embedded, because that is the actual question;
// embedding the whole conversation would let a long shared history dominate the
// vector and make unrelated questions look similar. The preceding context is
// folded in as a hash instead, so two different conversations can never collide
// even if their last messages are identical.
func SemanticText(req *openai.ChatRequest) (text, contextHash string, ok bool) {
	lastUser := -1
	for i := len(req.Messages) - 1; i >= 0; i-- {
		if req.Messages[i].Role == openai.RoleUser {
			lastUser = i
			break
		}
	}
	if lastUser < 0 {
		return "", "", false
	}

	text = strings.TrimSpace(req.Messages[lastUser].Content)
	if text == "" {
		return "", "", false
	}

	h := sha256.New()
	for _, m := range req.Messages[:lastUser] {
		h.Write([]byte(m.Role))
		h.Write([]byte{0})
		h.Write([]byte(strings.TrimSpace(m.Content)))
		h.Write([]byte{0})
	}
	return text, hex.EncodeToString(h.Sum(nil)[:16]), true
}

// BypassRequested reports whether the caller asked to skip the cache via the
// X-Waypoint-Cache header.
func BypassRequested(headerValue string) bool {
	return strings.EqualFold(strings.TrimSpace(headerValue), "bypass")
}
