package mocksim

import (
	"container/list"
	"encoding/binary"
	"hash/fnv"
	"sync"
)

// PrefixCache simulates a model server's block-level KV prefix cache.
//
// Real engines (vLLM, SGLang) store attention key/value tensors in fixed-size
// blocks of tokens and key each block by a hash that *chains* the hashes of all
// preceding blocks. Chaining is the detail that makes prefix caching correct: a
// block may only be reused if the entire token prefix leading up to it is
// identical, because the KV values for a token depend on every token before it.
//
// A trailing partial block is never cached, matching real engines, which can
// only commit a block once it is full.
type PrefixCache struct {
	mu        sync.Mutex
	blockSize int
	capacity  int // in blocks
	order     *list.List
	items     map[uint64]*list.Element

	// Cumulative stats, guarded by mu.
	blockLookups uint64
	blockHits    uint64
	requests     uint64
	evictions    uint64
}

// NewPrefixCache returns a cache holding at most capacity blocks of blockSize
// tokens each. A capacity of 0 disables caching entirely.
func NewPrefixCache(blockSize, capacity int) *PrefixCache {
	if blockSize < 1 {
		blockSize = 16
	}
	if capacity < 0 {
		capacity = 0
	}
	return &PrefixCache{
		blockSize: blockSize,
		capacity:  capacity,
		order:     list.New(),
		items:     make(map[uint64]*list.Element, capacity),
	}
}

// BlockSize returns the number of tokens per block.
func (c *PrefixCache) BlockSize() int { return c.blockSize }

// fnvOffset64 is the FNV-1a 64-bit offset basis, used to seed the hash chain.
const fnvOffset64 = uint64(14695981039346656037)

// BlockHashes converts a token sequence into one chained hash per *full* block.
//
// hash[i] = FNV1a(hash[i-1] || tokens of block i), so block i only matches when
// blocks 0..i all match.
func BlockHashes(tokens []string, blockSize int) []uint64 {
	if blockSize < 1 || len(tokens) < blockSize {
		return nil
	}

	numBlocks := len(tokens) / blockSize
	hashes := make([]uint64, 0, numBlocks)

	prev := fnvOffset64
	var scratch [8]byte

	for b := 0; b < numBlocks; b++ {
		h := fnv.New64a()
		binary.LittleEndian.PutUint64(scratch[:], prev)
		_, _ = h.Write(scratch[:])

		for _, tok := range tokens[b*blockSize : (b+1)*blockSize] {
			_, _ = h.Write([]byte(tok))
			// A separator prevents ["ab","c"] and ["a","bc"] colliding.
			_, _ = h.Write([]byte{0})
		}

		prev = h.Sum64()
		hashes = append(hashes, prev)
	}
	return hashes
}

// Lookup reports how many leading blocks are already resident, stopping at the
// first miss. Hit blocks are refreshed in LRU order.
func (c *PrefixCache) Lookup(hashes []uint64) int {
	c.mu.Lock()
	defer c.mu.Unlock()

	c.requests++
	c.blockLookups += uint64(len(hashes))

	hit := 0
	for _, h := range hashes {
		elem, ok := c.items[h]
		if !ok {
			break
		}
		c.order.MoveToFront(elem)
		hit++
	}
	c.blockHits += uint64(hit)
	return hit
}

// Admit inserts every block hash, evicting least-recently-used blocks as needed.
func (c *PrefixCache) Admit(hashes []uint64) {
	if c.capacity == 0 {
		return
	}

	c.mu.Lock()
	defer c.mu.Unlock()

	for _, h := range hashes {
		if elem, ok := c.items[h]; ok {
			c.order.MoveToFront(elem)
			continue
		}
		c.items[h] = c.order.PushFront(h)

		for c.order.Len() > c.capacity {
			oldest := c.order.Back()
			if oldest == nil {
				break
			}
			c.order.Remove(oldest)
			key, ok := oldest.Value.(uint64)
			if ok {
				delete(c.items, key)
			}
			c.evictions++
		}
	}
}

// CachedTokens reports how many prompt tokens a lookup of hashes would serve
// from cache, and admits the blocks for future requests. It is the single call
// the request path needs.
func (c *PrefixCache) CachedTokens(tokens []string) int {
	hashes := BlockHashes(tokens, c.blockSize)
	hitBlocks := c.Lookup(hashes)
	c.Admit(hashes)
	return hitBlocks * c.blockSize
}

// PrefixCacheStats is a point-in-time view of cache behavior.
type PrefixCacheStats struct {
	BlockSize    int     `json:"block_size"`
	Capacity     int     `json:"capacity_blocks"`
	Resident     int     `json:"resident_blocks"`
	Requests     uint64  `json:"requests"`
	BlockLookups uint64  `json:"block_lookups"`
	BlockHits    uint64  `json:"block_hits"`
	HitRate      float64 `json:"hit_rate"`
	Evictions    uint64  `json:"evictions"`
}

// Stats returns a snapshot of cache counters.
func (c *PrefixCache) Stats() PrefixCacheStats {
	c.mu.Lock()
	defer c.mu.Unlock()

	var rate float64
	if c.blockLookups > 0 {
		rate = float64(c.blockHits) / float64(c.blockLookups)
	}
	return PrefixCacheStats{
		BlockSize:    c.blockSize,
		Capacity:     c.capacity,
		Resident:     c.order.Len(),
		Requests:     c.requests,
		BlockLookups: c.blockLookups,
		BlockHits:    c.blockHits,
		HitRate:      rate,
		Evictions:    c.evictions,
	}
}

// Reset clears cache contents and counters. Used between benchmark runs so one
// experiment cannot warm the cache for the next.
func (c *PrefixCache) Reset() {
	c.mu.Lock()
	defer c.mu.Unlock()

	c.order.Init()
	c.items = make(map[uint64]*list.Element, c.capacity)
	c.blockLookups, c.blockHits, c.requests, c.evictions = 0, 0, 0, 0
}
