package cache

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"strconv"
	"strings"
	"time"

	"github.com/redis/go-redis/v9"
)

// RedisDoer is the Redis surface the semantic cache needs: ordinary commands
// plus Do, for the RediSearch commands go-redis has no typed method for.
type RedisDoer interface {
	redis.Cmdable
	Do(ctx context.Context, args ...any) *redis.Cmd
}

// SemanticCache answers a request from a *similar* previous request.
//
// This is the risky half of caching, and the risk is asymmetric: a missed hit
// costs latency, a wrong hit returns an answer to a question the user did not
// ask. Every guard here exists because of that asymmetry — temperature must be
// 0, model and sampling parameters must match exactly, the conversation context
// must match exactly, and only the final question is compared by similarity.
type SemanticCache struct {
	client    RedisDoer
	embedder  Embedder
	index     string
	keyPrefix string
	threshold float64
	topK      int
	ttl       time.Duration
	timeout   time.Duration
}

// SemanticOptions configures a SemanticCache.
type SemanticOptions struct {
	Embedder  Embedder
	IndexName string
	KeyPrefix string
	Threshold float64
	TopK      int
	TTL       time.Duration
	Timeout   time.Duration
}

// NewSemanticCache builds a SemanticCache.
func NewSemanticCache(client RedisDoer, opts SemanticOptions) *SemanticCache {
	if opts.IndexName == "" {
		opts.IndexName = "waypoint_semantic"
	}
	if opts.KeyPrefix == "" {
		opts.KeyPrefix = "waypoint"
	}
	if opts.Threshold <= 0 {
		opts.Threshold = 0.92
	}
	if opts.TopK < 1 {
		opts.TopK = 3
	}
	if opts.TTL <= 0 {
		opts.TTL = time.Hour
	}
	if opts.Timeout <= 0 {
		opts.Timeout = time.Second
	}
	return &SemanticCache{
		client:    client,
		embedder:  opts.Embedder,
		index:     opts.IndexName,
		keyPrefix: opts.KeyPrefix,
		threshold: opts.Threshold,
		topK:      opts.TopK,
		ttl:       opts.TTL,
		timeout:   opts.Timeout,
	}
}

// docPrefix is the Redis key prefix the index watches.
func (c *SemanticCache) docPrefix() string {
	return fmt.Sprintf("%s:cache:sem:", c.keyPrefix)
}

// EnsureIndex creates the vector index if it does not exist.
//
// FLAT (exhaustive) rather than HNSW: a response cache holds thousands of
// entries, not millions, and at that size an exact scan is both fast and free of
// HNSW's recall/parameter tuning. HNSW would be the right call an order of
// magnitude higher.
func (c *SemanticCache) EnsureIndex(ctx context.Context) error {
	callCtx, cancel := context.WithTimeout(ctx, 5*time.Second)
	defer cancel()

	// FT.INFO succeeds only if the index already exists.
	if err := c.client.Do(callCtx, "FT.INFO", c.index).Err(); err == nil {
		return nil
	}

	dims := c.embedder.Dimensions()
	err := c.client.Do(callCtx,
		"FT.CREATE", c.index,
		"ON", "HASH",
		"PREFIX", "1", c.docPrefix(),
		"SCHEMA",
		"model", "TAG",
		"params", "TAG",
		"ctx", "TAG",
		"embedding", "VECTOR", "FLAT", "6",
		"TYPE", "FLOAT32",
		"DIM", strconv.Itoa(dims),
		"DISTANCE_METRIC", "COSINE",
	).Err()

	if err != nil && !strings.Contains(strings.ToLower(err.Error()), "index already exists") {
		return fmt.Errorf("cache: create vector index: %w", err)
	}
	return nil
}

// Lookup finds the most similar cached entry above the threshold.
func (c *SemanticCache) Lookup(ctx context.Context, text, model, params, contextHash string) (*Entry, error) {
	vec, err := c.embedder.Embed(ctx, text)
	if err != nil {
		return nil, fmt.Errorf("cache: embed query: %w", err)
	}

	callCtx, cancel := context.WithTimeout(ctx, c.timeout)
	defer cancel()

	// Filter on exact-match fields first, then rank the survivors by vector
	// distance. Similarity alone is not sufficient for a valid hit.
	filter := fmt.Sprintf("(@model:{%s} @params:{%s} @ctx:{%s})",
		escapeTag(model), escapeTag(params), escapeTag(contextHash))
	query := fmt.Sprintf("%s=>[KNN %d @embedding $vec AS score]", filter, c.topK)

	raw, err := c.client.Do(callCtx,
		"FT.SEARCH", c.index, query,
		"PARAMS", "2", "vec", vectorToBytes(normalize(vec)),
		"SORTBY", "score",
		"RETURN", "2", "payload", "score",
		"DIALECT", "2",
	).Result()
	if err != nil {
		if isIndexMissing(err) {
			return nil, ErrMiss
		}
		return nil, fmt.Errorf("cache: vector search: %w", err)
	}

	entry, similarity, err := parseSearchReply(raw)
	if err != nil {
		return nil, err
	}
	if entry == nil {
		return nil, ErrMiss
	}

	if similarity < c.threshold {
		return nil, ErrMiss
	}

	entry.Source = SourceSemantic
	entry.Similarity = similarity
	return entry, nil
}

// Store indexes an entry for future similarity lookups.
func (c *SemanticCache) Store(ctx context.Context, text, model, params, contextHash string, entry *Entry) error {
	vec, err := c.embedder.Embed(ctx, text)
	if err != nil {
		return fmt.Errorf("cache: embed for store: %w", err)
	}

	payload, err := entry.Encode()
	if err != nil {
		return err
	}

	callCtx, cancel := context.WithTimeout(ctx, c.timeout)
	defer cancel()

	// Key on the text itself so re-answering the same question overwrites
	// rather than accumulating near-duplicate vectors.
	sum := sha256.Sum256([]byte(model + "\x00" + params + "\x00" + contextHash + "\x00" + text))
	key := c.docPrefix() + hex.EncodeToString(sum[:])

	if err := c.client.HSet(callCtx, key,
		"model", model,
		"params", params,
		"ctx", contextHash,
		"embedding", vectorToBytes(normalize(vec)),
		"payload", payload,
	).Err(); err != nil {
		return fmt.Errorf("cache: store semantic entry: %w", err)
	}

	// Entries expire so the cache reflects recent behavior and cannot grow
	// without bound.
	return c.client.Expire(callCtx, key, c.ttl).Err()
}

// Threshold reports the configured similarity threshold.
func (c *SemanticCache) Threshold() float64 { return c.threshold }

// SetThreshold adjusts the threshold at runtime, which the M8 experiment uses to
// sweep the hit-rate/quality tradeoff without restarting the stack.
func (c *SemanticCache) SetThreshold(t float64) {
	if t > 0 && t <= 1 {
		c.threshold = t
	}
}

// DropIndex removes the index and its documents, used between benchmark runs.
func (c *SemanticCache) DropIndex(ctx context.Context) error {
	callCtx, cancel := context.WithTimeout(ctx, 5*time.Second)
	defer cancel()

	err := c.client.Do(callCtx, "FT.DROPINDEX", c.index, "DD").Err()
	if err != nil && !isIndexMissing(err) {
		return err
	}
	return nil
}

// parseSearchReply extracts the best entry and its similarity from an
// FT.SEARCH reply.
//
// Redis returns COSINE *distance* (0 = identical), so similarity is 1 - distance.
func parseSearchReply(raw any) (*Entry, float64, error) {
	// RESP3 returns a map; RESP2 returns a flat array. Handle both.
	switch reply := raw.(type) {
	case map[any]any:
		return parseSearchMap(reply)
	case []any:
		return parseSearchArray(reply)
	default:
		return nil, 0, fmt.Errorf("cache: unexpected FT.SEARCH reply type %T", raw)
	}
}

func parseSearchMap(reply map[any]any) (*Entry, float64, error) {
	results, ok := reply["results"].([]any)
	if !ok || len(results) == 0 {
		return nil, 0, nil
	}

	first, ok := results[0].(map[any]any)
	if !ok {
		return nil, 0, nil
	}
	attrs, ok := first["extra_attributes"].(map[any]any)
	if !ok {
		return nil, 0, nil
	}

	payload, _ := attrs["payload"].(string)
	scoreStr, _ := attrs["score"].(string)
	return buildEntry(payload, scoreStr)
}

func parseSearchArray(reply []any) (*Entry, float64, error) {
	// [count, key1, [field, value, ...], key2, [...], ...]
	if len(reply) < 3 {
		return nil, 0, nil
	}

	fields, ok := reply[2].([]any)
	if !ok {
		return nil, 0, nil
	}

	var payload, scoreStr string
	for i := 0; i+1 < len(fields); i += 2 {
		name, _ := fields[i].(string)
		value, _ := fields[i+1].(string)
		switch name {
		case "payload":
			payload = value
		case "score":
			scoreStr = value
		}
	}
	return buildEntry(payload, scoreStr)
}

func buildEntry(payload, scoreStr string) (*Entry, float64, error) {
	if payload == "" {
		return nil, 0, nil
	}

	entry, err := DecodeEntry([]byte(payload))
	if err != nil {
		return nil, 0, nil // treat corruption as a miss
	}

	distance, err := strconv.ParseFloat(scoreStr, 64)
	if err != nil {
		return nil, 0, nil
	}
	return entry, 1 - distance, nil
}

// escapeTag escapes characters that are special inside a Redis TAG filter.
func escapeTag(s string) string {
	replacer := strings.NewReplacer(
		"-", "\\-", "@", "\\@", ":", "\\:", "{", "\\{", "}", "\\}",
		"|", "\\|", " ", "\\ ", ".", "\\.", "=", "\\=",
	)
	return replacer.Replace(s)
}

func isIndexMissing(err error) bool {
	if err == nil {
		return false
	}
	if errors.Is(err, redis.Nil) {
		return true
	}
	msg := strings.ToLower(err.Error())
	return strings.Contains(msg, "no such index") || strings.Contains(msg, "unknown index")
}
