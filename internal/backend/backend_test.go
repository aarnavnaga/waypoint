package backend

import (
	"sync"
	"testing"

	"github.com/aarnavnaga/waypoint/internal/config"
)

func testPool(t *testing.T, names ...string) *Pool {
	t.Helper()
	cfgs := make([]config.Backend, 0, len(names))
	for i, n := range names {
		cfgs = append(cfgs, config.Backend{
			Name:              n,
			URL:               "http://" + n + ":9001",
			MaxInflightTokens: 1000 * (i + 1),
		})
	}
	p, err := NewPool(cfgs)
	if err != nil {
		t.Fatalf("NewPool: %v", err)
	}
	return p
}

func TestNewRejectsBadURL(t *testing.T) {
	tests := []struct {
		name string
		url  string
	}{
		{"relative", "mock-1:9001"},
		{"no scheme", "//mock-1:9001"},
		{"empty", ""},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			if _, err := New(config.Backend{Name: "b", URL: tc.url}); err == nil {
				t.Errorf("New accepted url %q", tc.url)
			}
		})
	}
}

func TestNewPoolRejectsDuplicates(t *testing.T) {
	_, err := NewPool([]config.Backend{
		{Name: "a", URL: "http://a:1"},
		{Name: "a", URL: "http://b:1"},
	})
	if err == nil {
		t.Error("NewPool accepted duplicate names")
	}
}

func TestNewPoolRejectsEmpty(t *testing.T) {
	if _, err := NewPool(nil); err == nil {
		t.Error("NewPool accepted an empty backend list")
	}
}

func TestReserveTracksTokenLoad(t *testing.T) {
	p := testPool(t, "a")
	b, _ := p.Get("a")

	rel1 := b.Reserve(500)
	if got := b.InflightTokens(); got != 500 {
		t.Errorf("InflightTokens = %d, want 500", got)
	}

	rel2 := b.Reserve(300)
	if got := b.InflightTokens(); got != 800 {
		t.Errorf("InflightTokens = %d, want 800", got)
	}
	if got := b.InflightRequests(); got != 2 {
		t.Errorf("InflightRequests = %d, want 2", got)
	}

	rel1()
	if got := b.InflightTokens(); got != 300 {
		t.Errorf("InflightTokens after release = %d, want 300", got)
	}

	rel2()
	if got := b.InflightTokens(); got != 0 {
		t.Errorf("InflightTokens after both releases = %d, want 0", got)
	}
}

// A double release would drive load negative and permanently skew routing.
func TestReserveReleaseIsIdempotent(t *testing.T) {
	p := testPool(t, "a")
	b, _ := p.Get("a")

	rel := b.Reserve(100)
	rel()
	rel()
	rel()

	if got := b.InflightTokens(); got != 0 {
		t.Errorf("InflightTokens = %d after repeated release, want 0", got)
	}
	if got := b.InflightRequests(); got != 0 {
		t.Errorf("InflightRequests = %d after repeated release, want 0", got)
	}
}

func TestReserveNegativeTokensClamped(t *testing.T) {
	p := testPool(t, "a")
	b, _ := p.Get("a")

	rel := b.Reserve(-50)
	defer rel()
	if got := b.InflightTokens(); got != 0 {
		t.Errorf("InflightTokens = %d, want 0 for a negative reservation", got)
	}
}

func TestHasCapacity(t *testing.T) {
	p := testPool(t, "a")
	b, _ := p.Get("a") // MaxInflightTokens = 1000

	if !b.HasCapacity() {
		t.Error("idle backend reported no capacity")
	}

	rel := b.Reserve(1000)
	defer rel()
	if b.HasCapacity() {
		t.Error("saturated backend reported capacity")
	}
}

func TestUnboundedCapacity(t *testing.T) {
	b, err := New(config.Backend{Name: "a", URL: "http://a:1", MaxInflightTokens: 0})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	rel := b.Reserve(1_000_000)
	defer rel()

	if !b.HasCapacity() {
		t.Error("backend with no token ceiling reported no capacity")
	}
}

func TestSetHealthyReportsChange(t *testing.T) {
	p := testPool(t, "a")
	b, _ := p.Get("a")

	if !b.Healthy() {
		t.Error("new backend should start healthy")
	}
	if changed := b.SetHealthy(true); changed {
		t.Error("SetHealthy(true) on a healthy backend reported a change")
	}
	if changed := b.SetHealthy(false); !changed {
		t.Error("SetHealthy(false) did not report a change")
	}
	if b.Healthy() {
		t.Error("backend still healthy after SetHealthy(false)")
	}
}

func TestAvailableRespectsBreaker(t *testing.T) {
	p := testPool(t, "a")
	b, _ := p.Get("a")

	if !b.Available() {
		t.Error("healthy backend with no breaker should be available")
	}
	if got := b.BreakerState(); got != "none" {
		t.Errorf("BreakerState = %q, want none", got)
	}

	open := true
	b.SetBreaker(func() bool { return !open }, func() string { return "open" })

	if b.Available() {
		t.Error("backend with an open breaker reported available")
	}
	if got := b.BreakerState(); got != "open" {
		t.Errorf("BreakerState = %q, want open", got)
	}

	open = false
	if !b.Available() {
		t.Error("backend with a closed breaker reported unavailable")
	}
}

func TestAvailableRequiresHealth(t *testing.T) {
	p := testPool(t, "a")
	b, _ := p.Get("a")

	b.SetHealthy(false)
	if b.Available() {
		t.Error("unhealthy backend reported available")
	}
}

func TestServesModel(t *testing.T) {
	p := testPool(t, "a")
	b, _ := p.Get("a")

	// No known models: assume it serves anything rather than nothing, so model
	// discovery failing degrades to "route anywhere".
	if !b.ServesModel("anything") {
		t.Error("backend with no model list should accept any model")
	}

	b.SetModels([]string{"mock-7b", "mock-13b"})
	if !b.ServesModel("mock-7b") {
		t.Error("ServesModel rejected a listed model")
	}
	if b.ServesModel("gpt-4") {
		t.Error("ServesModel accepted an unlisted model")
	}
}

func TestModelsReturnsACopy(t *testing.T) {
	p := testPool(t, "a")
	b, _ := p.Get("a")
	b.SetModels([]string{"m1"})

	got := b.Models()
	got[0] = "mutated"

	if b.Models()[0] != "m1" {
		t.Error("Models() exposed internal state to mutation")
	}
}

func TestPoolAvailableFiltersByModel(t *testing.T) {
	p := testPool(t, "a", "b", "c")
	ba, _ := p.Get("a")
	bb, _ := p.Get("b")
	bc, _ := p.Get("c")

	ba.SetModels([]string{"mock-7b"})
	bb.SetModels([]string{"mock-13b"})
	bc.SetModels([]string{"mock-7b"})

	got := p.Available("mock-7b")
	if len(got) != 2 {
		t.Fatalf("Available(mock-7b) returned %d backends, want 2", len(got))
	}

	if all := p.Available(""); len(all) != 3 {
		t.Errorf("Available(\"\") returned %d, want all 3", len(all))
	}
}

func TestPoolAvailableExcludesUnhealthy(t *testing.T) {
	p := testPool(t, "a", "b")
	ba, _ := p.Get("a")
	ba.SetHealthy(false)

	if got := p.Available(""); len(got) != 1 || got[0].Name != "b" {
		t.Errorf("Available returned %v, want only b", names(got))
	}
}

func TestPoolHealthyIgnoresCapacity(t *testing.T) {
	p := testPool(t, "a")
	b, _ := p.Get("a")
	rel := b.Reserve(5000) // over its 1000 ceiling
	defer rel()

	if len(p.Available("")) != 0 {
		t.Error("saturated backend should not be available")
	}
	// readyz asks a different question: could we ever serve?
	if len(p.Healthy()) != 1 {
		t.Error("saturated-but-healthy backend should still count as healthy")
	}
}

func TestAverageInflightTokens(t *testing.T) {
	p := testPool(t, "a", "b", "c")
	ba, _ := p.Get("a")
	bb, _ := p.Get("b")

	relA := ba.Reserve(300)
	relB := bb.Reserve(600)
	defer relA()
	defer relB()

	if got := AverageInflightTokens(p.All()); got != 300 {
		t.Errorf("AverageInflightTokens = %v, want 300", got)
	}
	if got := AverageInflightTokens(nil); got != 0 {
		t.Errorf("AverageInflightTokens(nil) = %v, want 0", got)
	}
}

func TestTotalInflightTokens(t *testing.T) {
	p := testPool(t, "a", "b")
	ba, _ := p.Get("a")
	rel := ba.Reserve(250)
	defer rel()

	if got := p.TotalInflightTokens(); got != 250 {
		t.Errorf("TotalInflightTokens = %d, want 250", got)
	}
}

func TestUnionModelsIsSortedAndDeduplicated(t *testing.T) {
	p := testPool(t, "a", "b")
	ba, _ := p.Get("a")
	bb, _ := p.Get("b")

	ba.SetModels([]string{"zeta", "alpha"})
	bb.SetModels([]string{"alpha", "mid"})

	got := UnionModels(p.All())
	want := []string{"alpha", "mid", "zeta"}

	if len(got) != len(want) {
		t.Fatalf("UnionModels = %v, want %v", got, want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Errorf("UnionModels = %v, want %v", got, want)
			break
		}
	}
}

func TestStatsSnapshot(t *testing.T) {
	p := testPool(t, "a")
	b, _ := p.Get("a")
	b.SetModels([]string{"m"})
	rel := b.Reserve(120)
	defer rel()

	s := b.Stats()
	if s.Name != "a" || s.InflightTokens != 120 || s.InflightRequests != 1 {
		t.Errorf("unexpected stats: %+v", s)
	}
	if s.TotalRequests != 1 || s.TotalTokens != 120 {
		t.Errorf("cumulative counters wrong: %+v", s)
	}
	if !s.Healthy || !s.Available {
		t.Errorf("expected healthy and available: %+v", s)
	}
}

// Load accounting is mutated by every concurrent request, so it must be exact
// under -race: a lost update permanently skews routing.
func TestConcurrentReserveRelease(t *testing.T) {
	p := testPool(t, "a")
	b, _ := p.Get("a")

	var wg sync.WaitGroup
	for i := 0; i < 100; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			rel := b.Reserve(10)
			_ = b.InflightTokens()
			_ = b.Available()
			rel()
		}()
	}
	wg.Wait()

	if got := b.InflightTokens(); got != 0 {
		t.Errorf("InflightTokens = %d after balanced reserve/release, want 0", got)
	}
	if got := b.InflightRequests(); got != 0 {
		t.Errorf("InflightRequests = %d, want 0", got)
	}
}

func names(bs []*Backend) []string {
	out := make([]string, len(bs))
	for i, b := range bs {
		out[i] = b.Name
	}
	return out
}
