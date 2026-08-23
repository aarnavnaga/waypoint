package router

import (
	"context"
	"fmt"
	"math"
	"sync"
	"testing"

	"github.com/aarnavnaga/waypoint/internal/backend"
	"github.com/aarnavnaga/waypoint/internal/config"
	"github.com/aarnavnaga/waypoint/internal/openai"
	"github.com/aarnavnaga/waypoint/internal/tokenizer"
)

func pool(t *testing.T, n int) []*backend.Backend {
	t.Helper()

	cfgs := make([]config.Backend, 0, n)
	for i := 0; i < n; i++ {
		cfgs = append(cfgs, config.Backend{
			Name: fmt.Sprintf("mock-%d", i+1),
			URL:  fmt.Sprintf("http://mock-%d:9001", i+1),
		})
	}
	p, err := backend.NewPool(cfgs)
	if err != nil {
		t.Fatalf("NewPool: %v", err)
	}
	return p.All()
}

func req(prefix string, tokens int64) *Request {
	return &Request{Model: "mock-7b", PrefixKey: prefix, EstimatedTokens: tokens}
}

func allPolicies() []Policy {
	return []Policy{
		NewRoundRobin(),
		NewLeastLoaded(),
		NewPrefixAffinity(150, 1.25),
		NewPrefixAffinityLoadAware(150, 1.25),
	}
}

func TestAllPoliciesRejectEmptyBackendSet(t *testing.T) {
	for _, p := range allPolicies() {
		t.Run(p.Name(), func(t *testing.T) {
			if _, err := p.Pick(context.Background(), req("k", 10), nil); err == nil {
				t.Error("Pick accepted an empty backend set")
			}
		})
	}
}

func TestAllPoliciesReturnAMemberOfTheSet(t *testing.T) {
	backends := pool(t, 4)

	for _, p := range allPolicies() {
		t.Run(p.Name(), func(t *testing.T) {
			for i := 0; i < 50; i++ {
				got, err := p.Pick(context.Background(), req(fmt.Sprintf("key-%d", i), 100), backends)
				if err != nil {
					t.Fatalf("Pick: %v", err)
				}
				found := false
				for _, b := range backends {
					if b == got {
						found = true
						break
					}
				}
				if !found {
					t.Fatalf("Pick returned %q, which is not in the eligible set", got.Name)
				}
			}
		})
	}
}

func TestAllPoliciesHandleSingleBackend(t *testing.T) {
	backends := pool(t, 1)

	for _, p := range allPolicies() {
		t.Run(p.Name(), func(t *testing.T) {
			got, err := p.Pick(context.Background(), req("k", 10), backends)
			if err != nil {
				t.Fatalf("Pick: %v", err)
			}
			if got != backends[0] {
				t.Errorf("Pick returned %q, want the only backend", got.Name)
			}
		})
	}
}

func TestRoundRobinCyclesInOrder(t *testing.T) {
	backends := pool(t, 3)
	p := NewRoundRobin()

	var got []string
	for i := 0; i < 6; i++ {
		b, err := p.Pick(context.Background(), req("k", 10), backends)
		if err != nil {
			t.Fatalf("Pick: %v", err)
		}
		got = append(got, b.Name)
	}

	want := []string{"mock-1", "mock-2", "mock-3", "mock-1", "mock-2", "mock-3"}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("sequence = %v, want %v", got, want)
		}
	}
}

// Round robin is deliberately blind to load: this documents the baseline's
// weakness that every other policy exists to fix.
func TestRoundRobinIgnoresLoad(t *testing.T) {
	backends := pool(t, 2)
	rel := backends[0].Reserve(1_000_000)
	defer rel()

	p := NewRoundRobin()
	saw := map[string]int{}
	for i := 0; i < 10; i++ {
		b, _ := p.Pick(context.Background(), req("k", 10), backends)
		saw[b.Name]++
	}

	if saw["mock-1"] != 5 {
		t.Errorf("round robin sent %d requests to the saturated backend, want 5 (it ignores load by design)",
			saw["mock-1"])
	}
}

func TestLeastLoadedPicksLowestTokenCount(t *testing.T) {
	backends := pool(t, 3)
	r1 := backends[0].Reserve(500)
	r2 := backends[1].Reserve(100)
	r3 := backends[2].Reserve(900)
	defer func() { r1(); r2(); r3() }()

	p := NewLeastLoaded()
	got, err := p.Pick(context.Background(), req("k", 10), backends)
	if err != nil {
		t.Fatalf("Pick: %v", err)
	}
	if got.Name != "mock-2" {
		t.Errorf("Pick = %q, want mock-2 (lowest load)", got.Name)
	}
}

// Load is measured in tokens, not requests: one huge request outweighs several
// small ones.
func TestLeastLoadedCountsTokensNotRequests(t *testing.T) {
	backends := pool(t, 2)

	// mock-1: one large request. mock-2: five small ones.
	big := backends[0].Reserve(32000)
	defer big()
	for i := 0; i < 5; i++ {
		defer backends[1].Reserve(100)()
	}

	got, err := NewLeastLoaded().Pick(context.Background(), req("k", 10), backends)
	if err != nil {
		t.Fatalf("Pick: %v", err)
	}
	if got.Name != "mock-2" {
		t.Errorf("Pick = %q, want mock-2: 5x100 tokens is lighter than 1x32000", got.Name)
	}
}

func TestLeastLoadedDeterministicOnTies(t *testing.T) {
	backends := pool(t, 3)
	p := NewLeastLoaded()

	first, _ := p.Pick(context.Background(), req("k", 10), backends)
	for i := 0; i < 20; i++ {
		got, _ := p.Pick(context.Background(), req("k", 10), backends)
		if got != first {
			t.Fatal("tie-breaking is not deterministic")
		}
	}
}

// The core affinity property: the same prefix must route to the same backend.
func TestPrefixAffinityIsStable(t *testing.T) {
	backends := pool(t, 5)
	p := NewPrefixAffinity(150, 1.25)

	first, err := p.Pick(context.Background(), req("system-prompt-alpha", 100), backends)
	if err != nil {
		t.Fatalf("Pick: %v", err)
	}
	for i := 0; i < 100; i++ {
		got, _ := p.Pick(context.Background(), req("system-prompt-alpha", 100), backends)
		if got != first {
			t.Fatalf("iteration %d routed to %q, want stable %q", i, got.Name, first.Name)
		}
	}
}

func TestPrefixAffinityDistributesDistinctKeys(t *testing.T) {
	backends := pool(t, 4)
	p := NewPrefixAffinity(150, 1.25)

	seen := map[string]int{}
	for i := 0; i < 400; i++ {
		b, _ := p.Pick(context.Background(), req(fmt.Sprintf("prompt-%d", i), 10), backends)
		seen[b.Name]++
	}

	if len(seen) != 4 {
		t.Errorf("only %d of 4 backends received traffic: %v", len(seen), seen)
	}
	// Virtual nodes should keep shares within a reasonable band of the 100 mean.
	for name, count := range seen {
		if count < 40 || count > 190 {
			t.Errorf("backend %s got %d of 400 keys; distribution is too skewed (%v)",
				name, count, seen)
		}
	}
}

// Consistent hashing's reason for existing: losing one backend must not remap
// every key, or the whole fleet's prefix caches are invalidated at once.
func TestPrefixAffinityMinimizesRemappingOnBackendLoss(t *testing.T) {
	backends := pool(t, 5)
	p := NewPrefixAffinity(150, 1.25)

	const keys = 1000
	before := make([]string, keys)
	for i := 0; i < keys; i++ {
		b, _ := p.Pick(context.Background(), req(fmt.Sprintf("key-%d", i), 1), backends)
		before[i] = b.Name
	}

	// Remove one backend from the eligible set.
	reduced := backends[:4]

	remapped := 0
	for i := 0; i < keys; i++ {
		b, _ := p.Pick(context.Background(), req(fmt.Sprintf("key-%d", i), 1), reduced)
		if b.Name != before[i] {
			remapped++
		}
	}

	// Ideal is 1/5 = 20%. Modulo hashing would remap ~80%.
	fraction := float64(remapped) / keys
	t.Logf("removing 1 of 5 backends remapped %.1f%% of keys", fraction*100)
	if fraction > 0.35 {
		t.Errorf("remapped %.1f%% of keys, want <=35%% (consistent hashing should remap ~1/N)",
			fraction*100)
	}
}

// The bounded-load guarantee required by SPEC §6.3: under a maximally skewed key
// distribution (every request shares one prefix), no backend may exceed
// c x average load.
func TestPrefixAffinityBoundedLoadUnderSkew(t *testing.T) {
	const (
		backendCount = 4
		requests     = 200
		tokensEach   = 100
		c            = 1.25
	)

	backends := pool(t, backendCount)
	p := NewPrefixAffinity(150, c)

	releases := make([]func(), 0, requests)
	defer func() {
		for _, rel := range releases {
			rel()
		}
	}()

	for i := 0; i < requests; i++ {
		// Every request has the SAME prefix: plain consistent hashing would
		// send all of them to one backend.
		b, err := p.Pick(context.Background(), req("one-very-popular-system-prompt", tokensEach), backends)
		if err != nil {
			t.Fatalf("Pick: %v", err)
		}
		releases = append(releases, b.Reserve(tokensEach))
	}

	total := int64(0)
	loads := map[string]int64{}
	for _, b := range backends {
		loads[b.Name] = b.InflightTokens()
		total += b.InflightTokens()
	}
	avg := float64(total) / float64(backendCount)

	t.Logf("loads under a single hot prefix: %v (avg %.0f, ceiling %.0f)", loads, avg, c*avg)

	for name, load := range loads {
		if float64(load) > math.Ceil(c*avg)+float64(tokensEach) {
			t.Errorf("backend %s carries %d tokens, above the %.0f ceiling: bounded load is not enforced",
				name, load, c*avg)
		}
	}

	// Sanity: a bounded-load policy must actually spread a hot key, otherwise
	// the test above would pass trivially with everything on one backend.
	used := 0
	for _, load := range loads {
		if load > 0 {
			used++
		}
	}
	if used < 2 {
		t.Errorf("a single hot prefix landed on %d backend(s); bounded load did not spill", used)
	}
}

// With no load anywhere, affinity should be pure: nothing to balance.
func TestPrefixAffinityPureWhenIdle(t *testing.T) {
	backends := pool(t, 3)
	p := NewPrefixAffinity(150, 1.25)

	first, _ := p.Pick(context.Background(), req("key", 0), backends)
	for i := 0; i < 20; i++ {
		got, _ := p.Pick(context.Background(), req("key", 0), backends)
		if got != first {
			t.Fatal("idle pool should give pure affinity")
		}
	}
}

func TestPrefixAffinityLoadAwareKeepsAffinityWhenBalanced(t *testing.T) {
	backends := pool(t, 3)
	p := NewPrefixAffinityLoadAware(150, 1.25)

	// Even load across the fleet: affinity should be preserved.
	for _, b := range backends {
		defer b.Reserve(1000)()
	}

	first, _ := p.Pick(context.Background(), req("sticky-key", 10), backends)
	for i := 0; i < 20; i++ {
		got, _ := p.Pick(context.Background(), req("sticky-key", 10), backends)
		if got != first {
			t.Fatalf("affinity abandoned despite a balanced pool (iteration %d)", i)
		}
	}
}

func TestPrefixAffinityLoadAwareFallsBackWhenPreferredIsHot(t *testing.T) {
	backends := pool(t, 3)
	p := NewPrefixAffinityLoadAware(150, 1.25)

	const key = "hot-prefix"
	preferred, _ := p.Pick(context.Background(), req(key, 10), backends)

	// Make the preferred backend far hotter than the pool average.
	rel := preferred.Reserve(100_000)
	defer rel()

	got, err := p.Pick(context.Background(), req(key, 10), backends)
	if err != nil {
		t.Fatalf("Pick: %v", err)
	}
	if got == preferred {
		t.Error("policy kept affinity to a backend far above the load ceiling")
	}

	// It should have chosen the least loaded, which is one of the idle peers.
	if got.InflightTokens() != 0 {
		t.Errorf("fallback chose %q with %d tokens, want an idle backend",
			got.Name, got.InflightTokens())
	}
}

func TestLoadFactorClampedToOne(t *testing.T) {
	// A factor below 1 would mean "allow less than average", which is
	// unsatisfiable for every backend simultaneously.
	backends := pool(t, 3)

	for _, p := range []Policy{
		NewPrefixAffinity(150, 0.1),
		NewPrefixAffinityLoadAware(150, 0.1),
	} {
		t.Run(p.Name(), func(t *testing.T) {
			for _, b := range backends {
				defer b.Reserve(100)()
			}
			if _, err := p.Pick(context.Background(), req("k", 10), backends); err != nil {
				t.Errorf("Pick failed with a clamped load factor: %v", err)
			}
		})
	}
}

func TestRingCacheRebuildsWhenBackendSetChanges(t *testing.T) {
	backends := pool(t, 4)
	p := NewPrefixAffinity(150, 1.25)

	// Warm the cache with the full set.
	if _, err := p.Pick(context.Background(), req("k", 1), backends); err != nil {
		t.Fatalf("Pick: %v", err)
	}

	// A smaller set must not return a backend that is no longer eligible.
	reduced := backends[:2]
	for i := 0; i < 100; i++ {
		got, err := p.Pick(context.Background(), req(fmt.Sprintf("k%d", i), 1), reduced)
		if err != nil {
			t.Fatalf("Pick: %v", err)
		}
		if got.Name != "mock-1" && got.Name != "mock-2" {
			t.Fatalf("Pick returned %q, which is outside the reduced eligible set", got.Name)
		}
	}
}

func TestHashRingWalkVisitsEveryBackendOnce(t *testing.T) {
	ring := newHashRing([]string{"a", "b", "c", "d"}, 50)

	var visited []int
	ring.walk(hashString("some-key"), func(idx int) bool {
		visited = append(visited, idx)
		return true // keep walking
	})

	if len(visited) != 4 {
		t.Fatalf("walk visited %d backends, want 4", len(visited))
	}
	seen := map[int]bool{}
	for _, idx := range visited {
		if seen[idx] {
			t.Errorf("walk visited backend %d twice", idx)
		}
		seen[idx] = true
	}
}

func TestHashRingWalkStopsEarly(t *testing.T) {
	ring := newHashRing([]string{"a", "b", "c"}, 10)

	count := 0
	ring.walk(hashString("k"), func(int) bool {
		count++
		return false
	})
	if count != 1 {
		t.Errorf("walk continued after the callback returned false (%d visits)", count)
	}
}

func TestHashRingEmpty(t *testing.T) {
	ring := newHashRing(nil, 10)
	called := false
	ring.walk(42, func(int) bool { called = true; return true })
	if called {
		t.Error("empty ring invoked the callback")
	}
}

func TestRouterSetPolicy(t *testing.T) {
	cfg := config.Default().Routing
	r, err := New(cfg, tokenizer.NewCharHeuristic())
	if err != nil {
		t.Fatalf("New: %v", err)
	}

	for _, name := range config.ValidPolicies {
		if err := r.SetPolicy(name); err != nil {
			t.Errorf("SetPolicy(%q): %v", name, err)
		}
		if got := r.PolicyName(); got != name {
			t.Errorf("PolicyName = %q, want %q", got, name)
		}
	}

	if err := r.SetPolicy("does-not-exist"); err == nil {
		t.Error("SetPolicy accepted an unknown policy")
	}
	// A failed switch must leave the previous policy in place.
	if got := r.PolicyName(); got != config.ValidPolicies[len(config.ValidPolicies)-1] {
		t.Errorf("PolicyName = %q after a failed switch; the active policy changed", got)
	}
}

func TestRouterRejectsUnknownPolicyAtConstruction(t *testing.T) {
	cfg := config.Default().Routing
	cfg.Policy = "nope"
	if _, err := New(cfg, tokenizer.NewCharHeuristic()); err == nil {
		t.Error("New accepted an unknown policy")
	}
}

func TestBuildRequestPopulatesRoutingView(t *testing.T) {
	r, err := New(config.Default().Routing, tokenizer.NewCharHeuristic())
	if err != nil {
		t.Fatalf("New: %v", err)
	}

	got := r.BuildRequest(&openai.ChatRequest{
		Model:    "mock-7b",
		Messages: []openai.Message{{Role: openai.RoleUser, Content: "hello world"}},
	})

	if got.Model != "mock-7b" {
		t.Errorf("Model = %q", got.Model)
	}
	if got.PrefixKey == "" {
		t.Error("PrefixKey is empty")
	}
	if got.EstimatedTokens <= 0 {
		t.Errorf("EstimatedTokens = %d, want > 0", got.EstimatedTokens)
	}
}

func TestPrefixKeyIncludesModel(t *testing.T) {
	msgs := []openai.Message{{Role: openai.RoleUser, Content: "identical text"}}
	a := PrefixKey(&openai.ChatRequest{Model: "model-a", Messages: msgs}, 100)
	b := PrefixKey(&openai.ChatRequest{Model: "model-b", Messages: msgs}, 100)

	if a == b {
		t.Error("identical text on different models produced the same prefix key")
	}
}

func TestPrefixKeyBoundedByTokenBudget(t *testing.T) {
	long := ""
	for i := 0; i < 500; i++ {
		long += fmt.Sprintf("word%d ", i)
	}

	base := &openai.ChatRequest{Model: "m", Messages: []openai.Message{{Role: openai.RoleUser, Content: long}}}
	extended := &openai.ChatRequest{Model: "m", Messages: []openai.Message{
		{Role: openai.RoleUser, Content: long + " and a completely different tail"},
	}}

	// With a small budget, the diverging tail is outside the key.
	if PrefixKey(base, 10) != PrefixKey(extended, 10) {
		t.Error("prefix key included content beyond the token budget")
	}
}

// Policies are hit by every concurrent request; selection must be race-free.
func TestPoliciesAreConcurrencySafe(t *testing.T) {
	backends := pool(t, 4)

	for _, p := range allPolicies() {
		t.Run(p.Name(), func(t *testing.T) {
			var wg sync.WaitGroup
			for g := 0; g < 16; g++ {
				wg.Add(1)
				go func(g int) {
					defer wg.Done()
					for i := 0; i < 100; i++ {
						b, err := p.Pick(context.Background(),
							req(fmt.Sprintf("key-%d", (g+i)%7), 100), backends)
						if err != nil {
							t.Errorf("Pick: %v", err)
							return
						}
						rel := b.Reserve(100)
						rel()
					}
				}(g)
			}
			wg.Wait()
		})
	}
}

func BenchmarkPrefixAffinityPick(b *testing.B) {
	backends := make([]*backend.Backend, 0, 8)
	for i := 0; i < 8; i++ {
		bk, err := backend.New(config.Backend{
			Name: fmt.Sprintf("mock-%d", i),
			URL:  fmt.Sprintf("http://mock-%d:9001", i),
		})
		if err != nil {
			b.Fatalf("New: %v", err)
		}
		backends = append(backends, bk)
	}

	p := NewPrefixAffinity(150, 1.25)
	r := req("a-fairly-typical-system-prompt-prefix-key", 500)
	ctx := context.Background()

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := p.Pick(ctx, r, backends); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkLeastLoadedPick(b *testing.B) {
	backends := make([]*backend.Backend, 0, 8)
	for i := 0; i < 8; i++ {
		bk, _ := backend.New(config.Backend{
			Name: fmt.Sprintf("mock-%d", i),
			URL:  fmt.Sprintf("http://mock-%d:9001", i),
		})
		backends = append(backends, bk)
	}

	p := NewLeastLoaded()
	r := req("k", 500)
	ctx := context.Background()

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := p.Pick(ctx, r, backends); err != nil {
			b.Fatal(err)
		}
	}
}

// Ring balance is a property of the hash function, not of the policy. This test
// exists because plain FNV-1a silently produced a ring where one backend of four
// owned 40% of the keyspace and another received no keys at all — the policy
// looked correct, and only the distribution gave it away.
func TestHashRingArcsAreBalanced(t *testing.T) {
	names := []string{"mock-1", "mock-2", "mock-3", "mock-4"}
	ring := newHashRing(names, 150)

	arcs := make([]float64, len(names))
	var prev uint64
	for _, p := range ring.points {
		arcs[p.backend] += float64(p.hash - prev)
		prev = p.hash
	}

	var total float64
	for _, a := range arcs {
		total += a
	}

	shares := make([]float64, len(arcs))
	for i, a := range arcs {
		shares[i] = a / total * 100
	}
	t.Logf("arc shares: %.1f%%", shares)

	const ideal = 100.0 / 4
	for i, share := range shares {
		if share < ideal*0.6 || share > ideal*1.5 {
			t.Errorf("backend %s owns %.1f%% of the ring, want ~%.1f%%: hash avalanche is poor",
				names[i], share, ideal)
		}
	}
}

func TestMix64Avalanche(t *testing.T) {
	// Adjacent inputs must produce uncorrelated outputs, which is precisely
	// what raw FNV-1a fails to do for ring keys.
	var lowBitDiffs int
	for i := uint64(0); i < 1000; i++ {
		a, b := mix64(i), mix64(i+1)
		diff := a ^ b
		// A good mixer flips roughly half of all 64 bits.
		ones := 0
		for d := diff; d != 0; d &= d - 1 {
			ones++
		}
		if ones < 16 || ones > 48 {
			lowBitDiffs++
		}
	}
	if lowBitDiffs > 50 {
		t.Errorf("%d of 1000 adjacent inputs produced poorly mixed outputs", lowBitDiffs)
	}
}

func TestHashRingKeyDistribution(t *testing.T) {
	backends := pool(t, 4)
	p := NewPrefixAffinity(150, 1.25)

	counts := map[string]int{}
	const keys = 2000
	for i := 0; i < keys; i++ {
		b, err := p.Pick(context.Background(), req(fmt.Sprintf("system-prompt-%d", i), 0), backends)
		if err != nil {
			t.Fatalf("Pick: %v", err)
		}
		counts[b.Name]++
	}

	t.Logf("key distribution over %d keys: %v", keys, counts)
	if len(counts) != 4 {
		t.Fatalf("only %d of 4 backends received keys: %v", len(counts), counts)
	}

	ideal := float64(keys) / 4
	for name, got := range counts {
		if float64(got) < ideal*0.6 || float64(got) > ideal*1.5 {
			t.Errorf("backend %s received %d keys, want ~%.0f", name, got, ideal)
		}
	}
}
