package cache

import (
	"bytes"
	"context"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"math"
	"net/http"
	"time"
)

// Embedder turns text into a dense vector.
type Embedder interface {
	Embed(ctx context.Context, text string) ([]float32, error)
	Dimensions() int
}

// HTTPEmbedder calls the Python sidecar running sentence-transformers.
//
// The model runs out-of-process for two reasons: there is no good pure-Go
// sentence-transformers runtime, and embedding is CPU-bound work that should not
// share a process with a latency-sensitive proxy, where it would contend for the
// same scheduler and GC.
type HTTPEmbedder struct {
	baseURL string
	client  *http.Client
	dims    int
	timeout time.Duration
}

// EmbedderOptions configures an HTTPEmbedder.
type EmbedderOptions struct {
	BaseURL    string
	Dimensions int
	Timeout    time.Duration
	Client     *http.Client
}

// NewHTTPEmbedder builds an embedder client.
func NewHTTPEmbedder(opts EmbedderOptions) *HTTPEmbedder {
	if opts.Dimensions <= 0 {
		opts.Dimensions = 384 // all-MiniLM-L6-v2
	}
	if opts.Timeout <= 0 {
		opts.Timeout = 500 * time.Millisecond
	}
	client := opts.Client
	if client == nil {
		client = &http.Client{Timeout: opts.Timeout}
	}
	return &HTTPEmbedder{
		baseURL: opts.BaseURL,
		client:  client,
		dims:    opts.Dimensions,
		timeout: opts.Timeout,
	}
}

// Dimensions reports the embedding width.
func (e *HTTPEmbedder) Dimensions() int { return e.dims }

type embedRequest struct {
	Texts []string `json:"texts"`
}

type embedResponse struct {
	Embeddings [][]float32 `json:"embeddings"`
	Model      string      `json:"model"`
	Dimensions int         `json:"dimensions"`
}

// Embed returns the vector for text.
//
// The timeout is deliberately tight. Embedding sits in front of a cache lookup,
// so a slow embedder makes every *miss* slower than having no cache at all —
// the lookup cost is pure overhead when it fails to hit.
func (e *HTTPEmbedder) Embed(ctx context.Context, text string) ([]float32, error) {
	callCtx, cancel := context.WithTimeout(ctx, e.timeout)
	defer cancel()

	payload, err := json.Marshal(embedRequest{Texts: []string{text}})
	if err != nil {
		return nil, err
	}

	req, err := http.NewRequestWithContext(callCtx, http.MethodPost,
		e.baseURL+"/embed", bytes.NewReader(payload))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := e.client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("embed: %w", err)
	}
	defer func() { _ = resp.Body.Close() }()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("embed: sidecar returned %d", resp.StatusCode)
	}

	var out embedResponse
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
		return nil, fmt.Errorf("embed: decode: %w", err)
	}
	if len(out.Embeddings) == 0 {
		return nil, fmt.Errorf("embed: sidecar returned no embeddings")
	}

	vec := out.Embeddings[0]
	if len(vec) != e.dims {
		return nil, fmt.Errorf("embed: got %d dimensions, want %d", len(vec), e.dims)
	}
	return vec, nil
}

// Ready reports whether the sidecar is reachable and loaded.
func (e *HTTPEmbedder) Ready(ctx context.Context) error {
	callCtx, cancel := context.WithTimeout(ctx, 2*time.Second)
	defer cancel()

	req, err := http.NewRequestWithContext(callCtx, http.MethodGet, e.baseURL+"/healthz", http.NoBody)
	if err != nil {
		return err
	}
	resp, err := e.client.Do(req)
	if err != nil {
		return fmt.Errorf("embedder: %w", err)
	}
	defer func() { _ = resp.Body.Close() }()

	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("embedder: status %d", resp.StatusCode)
	}
	return nil
}

// vectorToBytes encodes a float32 slice as little-endian bytes, the layout
// Redis expects for a FLOAT32 vector field.
func vectorToBytes(vec []float32) []byte {
	buf := make([]byte, 4*len(vec))
	for i, v := range vec {
		binary.LittleEndian.PutUint32(buf[i*4:], math.Float32bits(v))
	}
	return buf
}

// normalize scales a vector to unit length.
//
// Cosine similarity is the dot product of unit vectors, so normalizing once at
// write and query time lets the index do less work and makes similarity scores
// directly comparable across entries.
func normalize(vec []float32) []float32 {
	var sum float64
	for _, v := range vec {
		sum += float64(v) * float64(v)
	}
	if sum == 0 {
		return vec
	}

	norm := float32(math.Sqrt(sum))
	out := make([]float32, len(vec))
	for i, v := range vec {
		out[i] = v / norm
	}
	return out
}

// CosineSimilarity returns the cosine similarity of two vectors, used by the M8
// cache-quality experiment to score a cached answer against a fresh one.
func CosineSimilarity(a, b []float32) float64 {
	if len(a) != len(b) || len(a) == 0 {
		return 0
	}

	var dot, normA, normB float64
	for i := range a {
		dot += float64(a[i]) * float64(b[i])
		normA += float64(a[i]) * float64(a[i])
		normB += float64(b[i]) * float64(b[i])
	}
	if normA == 0 || normB == 0 {
		return 0
	}
	return dot / (math.Sqrt(normA) * math.Sqrt(normB))
}

// StaticEmbedder is a deterministic embedder for tests, producing vectors whose
// similarity reflects token overlap. It avoids needing the Python sidecar for
// unit tests while still exercising the real vector-search path.
type StaticEmbedder struct {
	Dims int
}

// Dimensions implements Embedder.
func (s StaticEmbedder) Dimensions() int {
	if s.Dims <= 0 {
		return 64
	}
	return s.Dims
}

// Embed implements Embedder with a bag-of-words hashing trick: each word
// increments one dimension, so texts sharing words produce similar vectors.
func (s StaticEmbedder) Embed(_ context.Context, text string) ([]float32, error) {
	dims := s.Dimensions()
	vec := make([]float32, dims)

	for _, word := range splitWords(text) {
		h := fnv32(word)
		vec[h%uint32(dims)] += 1
	}
	return normalize(vec), nil
}

func splitWords(s string) []string {
	var words []string
	start := -1
	for i, r := range s {
		isWord := (r >= 'a' && r <= 'z') || (r >= 'A' && r <= 'Z') || (r >= '0' && r <= '9')
		switch {
		case isWord && start < 0:
			start = i
		case !isWord && start >= 0:
			words = append(words, lower(s[start:i]))
			start = -1
		}
	}
	if start >= 0 {
		words = append(words, lower(s[start:]))
	}
	return words
}

func lower(s string) string {
	out := []byte(s)
	for i, b := range out {
		if b >= 'A' && b <= 'Z' {
			out[i] = b + 32
		}
	}
	return string(out)
}

func fnv32(s string) uint32 {
	const (
		offset = 2166136261
		prime  = 16777619
	)
	h := uint32(offset)
	for i := 0; i < len(s); i++ {
		h ^= uint32(s[i])
		h *= prime
	}
	return h
}
