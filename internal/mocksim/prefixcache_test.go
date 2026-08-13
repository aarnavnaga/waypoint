package mocksim

import (
	"fmt"
	"sync"
	"testing"
)

func tokens(n int) []string {
	out := make([]string, n)
	for i := range out {
		out[i] = fmt.Sprintf("t%d", i)
	}
	return out
}

func TestBlockHashesOnlyCoversFullBlocks(t *testing.T) {
	tests := []struct {
		name      string
		tokens    int
		blockSize int
		want      int
	}{
		{"exactly one block", 4, 4, 1},
		{"partial block is dropped", 7, 4, 1},
		{"two full blocks", 8, 4, 2},
		{"fewer tokens than a block", 3, 4, 0},
		{"empty", 0, 4, 0},
		{"invalid block size", 8, 0, 0},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got := BlockHashes(tokens(tc.tokens), tc.blockSize)
			if len(got) != tc.want {
				t.Errorf("BlockHashes(%d tokens, bs=%d) returned %d blocks, want %d",
					tc.tokens, tc.blockSize, len(got), tc.want)
			}
		})
	}
}

// Chaining is the property that makes prefix reuse correct: identical blocks at
// different positions, or after a different prefix, must hash differently.
func TestBlockHashesAreChained(t *testing.T) {
	base := []string{"you", "are", "a", "helpful", "assistant", ".", "answer", "well"}
	diverged := append([]string{"you", "are", "a", "terrible"}, base[4:]...)

	h1 := BlockHashes(base, 4)
	h2 := BlockHashes(diverged, 4)

	if len(h1) != 2 || len(h2) != 2 {
		t.Fatalf("expected 2 blocks each, got %d and %d", len(h1), len(h2))
	}
	if h1[0] == h2[0] {
		t.Error("different first blocks hashed identically")
	}
	// Second blocks have identical *content* but different predecessors.
	if h1[1] == h2[1] {
		t.Error("second block hash ignored its predecessor; chaining is broken")
	}
}

func TestBlockHashesSharedPrefixMatches(t *testing.T) {
	shared := []string{"sys", "prompt", "shared", "across", "many", "users", "ok", "go"}
	a := append(append([]string{}, shared...), "alice", "asks", "about", "dogs")
	b := append(append([]string{}, shared...), "bob", "asks", "about", "cats")

	ha, hb := BlockHashes(a, 4), BlockHashes(b, 4)

	if ha[0] != hb[0] || ha[1] != hb[1] {
		t.Error("shared prefix blocks should hash identically")
	}
	if ha[2] == hb[2] {
		t.Error("diverging blocks should hash differently")
	}
}

func TestBlockHashSeparatorPreventsCollision(t *testing.T) {
	// Without a separator byte these two token sequences would concatenate to
	// the same bytes and collide.
	h1 := BlockHashes([]string{"ab", "c", "x", "y"}, 4)
	h2 := BlockHashes([]string{"a", "bc", "x", "y"}, 4)
	if h1[0] == h2[0] {
		t.Error("token boundary collision: separator is not working")
	}
}

func TestPrefixCacheHitsAndMisses(t *testing.T) {
	c := NewPrefixCache(4, 100)
	toks := tokens(16) // 4 blocks

	if got := c.CachedTokens(toks); got != 0 {
		t.Errorf("cold cache served %d tokens, want 0", got)
	}
	if got := c.CachedTokens(toks); got != 16 {
		t.Errorf("warm cache served %d tokens, want 16", got)
	}
}

func TestPrefixCachePartialPrefixReuse(t *testing.T) {
	c := NewPrefixCache(4, 100)

	first := append(tokens(8), "alpha", "beta", "gamma", "delta")
	c.CachedTokens(first)

	// Same first 8 tokens, different continuation: 2 blocks should hit.
	second := append(tokens(8), "epsilon", "zeta", "eta", "theta")
	if got := c.CachedTokens(second); got != 8 {
		t.Errorf("partial prefix served %d tokens, want 8", got)
	}
}

func TestPrefixCacheStopsAtFirstMiss(t *testing.T) {
	c := NewPrefixCache(2, 100)

	// Warm blocks for a sequence, then query one whose *second* block differs
	// but whose third block content matches the warm one. Because hashing is
	// chained, nothing past the divergence may be reused.
	warm := []string{"a", "b", "c", "d", "e", "f"}
	c.CachedTokens(warm)

	probe := []string{"a", "b", "X", "Y", "e", "f"}
	if got := c.CachedTokens(probe); got != 2 {
		t.Errorf("served %d tokens past a divergence, want exactly 2", got)
	}
}

func TestPrefixCacheLRUEviction(t *testing.T) {
	c := NewPrefixCache(4, 2) // only 2 blocks fit

	seqA := tokens(8)                            // blocks A0,A1
	seqB := append([]string{"zz"}, tokens(7)...) // different chain
	_ = c.CachedTokens(seqA)                     // admit A0,A1
	_ = c.CachedTokens(seqB)                     // admits B0,B1, evicting A
	if got := c.CachedTokens(seqA); got != 0 {
		t.Errorf("evicted sequence served %d tokens, want 0", got)
	}

	stats := c.Stats()
	if stats.Evictions == 0 {
		t.Error("expected evictions to be counted")
	}
	if stats.Resident > 2 {
		t.Errorf("resident blocks = %d, exceeds capacity 2", stats.Resident)
	}
}

func TestPrefixCacheLRUKeepsRecentlyUsed(t *testing.T) {
	c := NewPrefixCache(2, 3)

	hot := []string{"h1", "h2"}
	cold := []string{"c1", "c2"}

	c.CachedTokens(hot)
	c.CachedTokens(cold)
	c.CachedTokens(hot) // refresh hot so cold is the LRU victim

	filler := []string{"f1", "f2", "f3", "f4"} // 2 blocks, forces one eviction
	c.CachedTokens(filler)

	if got := c.CachedTokens(hot); got != 2 {
		t.Errorf("recently used sequence was evicted (served %d tokens, want 2)", got)
	}
}

func TestPrefixCacheZeroCapacityDisables(t *testing.T) {
	c := NewPrefixCache(4, 0)
	toks := tokens(8)

	c.CachedTokens(toks)
	if got := c.CachedTokens(toks); got != 0 {
		t.Errorf("zero-capacity cache served %d tokens, want 0", got)
	}
}

func TestPrefixCacheStats(t *testing.T) {
	c := NewPrefixCache(4, 100)
	toks := tokens(16)

	c.CachedTokens(toks) // 4 lookups, 0 hits
	c.CachedTokens(toks) // 4 lookups, 4 hits

	s := c.Stats()
	if s.Requests != 2 {
		t.Errorf("Requests = %d, want 2", s.Requests)
	}
	if s.BlockLookups != 8 {
		t.Errorf("BlockLookups = %d, want 8", s.BlockLookups)
	}
	if s.BlockHits != 4 {
		t.Errorf("BlockHits = %d, want 4", s.BlockHits)
	}
	if s.HitRate != 0.5 {
		t.Errorf("HitRate = %v, want 0.5", s.HitRate)
	}
}

func TestPrefixCacheReset(t *testing.T) {
	c := NewPrefixCache(4, 100)
	toks := tokens(8)
	c.CachedTokens(toks)
	c.Reset()

	if got := c.CachedTokens(toks); got != 0 {
		t.Errorf("after Reset cache served %d tokens, want 0", got)
	}
	if s := c.Stats(); s.Requests != 1 || s.BlockHits != 0 {
		t.Errorf("after Reset stats not cleared: %+v", s)
	}
}

// The cache is hit by every concurrent request on a backend, so it must be
// race-free under -race.
func TestPrefixCacheConcurrentAccess(t *testing.T) {
	c := NewPrefixCache(4, 64)

	var wg sync.WaitGroup
	for g := 0; g < 16; g++ {
		wg.Add(1)
		go func(g int) {
			defer wg.Done()
			for i := 0; i < 100; i++ {
				seq := append([]string{fmt.Sprintf("u%d", g%4)}, tokens(15)...)
				c.CachedTokens(seq)
				_ = c.Stats()
			}
		}(g)
	}
	wg.Wait()

	if s := c.Stats(); s.Resident > 64 {
		t.Errorf("resident blocks = %d, exceeds capacity under concurrency", s.Resident)
	}
}
