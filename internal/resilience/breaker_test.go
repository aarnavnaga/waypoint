package resilience

import (
	"sync"
	"testing"
	"time"
)

// fakeClock lets tests control window expiry and cool-downs without sleeping.
type fakeClock struct {
	mu  sync.Mutex
	now time.Time
}

func newFakeClock() *fakeClock {
	return &fakeClock{now: time.Date(2026, 1, 1, 0, 0, 0, 0, time.UTC)}
}

func (c *fakeClock) Now() time.Time {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.now
}

func (c *fakeClock) Advance(d time.Duration) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.now = c.now.Add(d)
}

func testBreaker(clock *fakeClock) *Breaker {
	return NewBreaker(BreakerOptions{
		Window:           10 * time.Second,
		Buckets:          10,
		MinRequests:      10,
		ErrorRate:        0.5,
		TimeoutRate:      0.5,
		OpenDuration:     5 * time.Second,
		HalfOpenRequests: 2,
		Now:              clock.Now,
	})
}

func TestBreakerStartsClosed(t *testing.T) {
	b := testBreaker(newFakeClock())

	if got := b.State(); got != StateClosed {
		t.Errorf("State = %v, want closed", got)
	}
	if !b.Allow() {
		t.Error("closed breaker refused a request")
	}
}

// A sample of one must never trip the breaker: a single failure at startup
// would otherwise read as a 100% error rate.
func TestBreakerRequiresMinimumSample(t *testing.T) {
	b := testBreaker(newFakeClock())

	for i := 0; i < 9; i++ {
		b.Record(OutcomeFailure)
	}
	if got := b.State(); got != StateClosed {
		t.Errorf("State = %v after 9 failures (min sample 10), want closed", got)
	}

	b.Record(OutcomeFailure)
	if got := b.State(); got != StateOpen {
		t.Errorf("State = %v after reaching the minimum sample, want open", got)
	}
}

func TestBreakerTripsOnErrorRate(t *testing.T) {
	b := testBreaker(newFakeClock())

	// 6 failures / 10 requests = 60% > 50% threshold.
	for i := 0; i < 4; i++ {
		b.Record(OutcomeSuccess)
	}
	for i := 0; i < 6; i++ {
		b.Record(OutcomeFailure)
	}

	if got := b.State(); got != StateOpen {
		t.Errorf("State = %v, want open at a 60%% error rate", got)
	}
	if b.Allow() {
		t.Error("open breaker admitted a request")
	}
}

func TestBreakerStaysClosedBelowThreshold(t *testing.T) {
	b := testBreaker(newFakeClock())

	// 4 failures / 20 = 20%, below the 50% threshold.
	for i := 0; i < 16; i++ {
		b.Record(OutcomeSuccess)
	}
	for i := 0; i < 4; i++ {
		b.Record(OutcomeFailure)
	}

	if got := b.State(); got != StateClosed {
		t.Errorf("State = %v at a 20%% error rate, want closed", got)
	}
}

// Timeouts have their own threshold: a backend that accepts work and stalls is
// a different failure mode from one that errors quickly.
func TestBreakerTripsOnTimeoutRateAlone(t *testing.T) {
	b := NewBreaker(BreakerOptions{
		MinRequests: 10,
		ErrorRate:   0.9, // deliberately high so only the timeout rule can fire
		TimeoutRate: 0.3,
		Now:         newFakeClock().Now,
	})

	for i := 0; i < 6; i++ {
		b.Record(OutcomeSuccess)
	}
	for i := 0; i < 4; i++ {
		b.Record(OutcomeTimeout)
	}

	// 40% timeouts exceeds the 30% timeout threshold, while the combined 40%
	// error rate is below the 90% error threshold.
	if got := b.State(); got != StateOpen {
		t.Errorf("State = %v, want open from the timeout rate alone", got)
	}
}

func TestBreakerWindowExpiresOldResults(t *testing.T) {
	clock := newFakeClock()
	b := testBreaker(clock)

	// Nine failures: not yet enough to trip.
	for i := 0; i < 9; i++ {
		b.Record(OutcomeFailure)
	}

	// Let the whole window roll past.
	clock.Advance(11 * time.Second)

	// Those failures must no longer count, so ten successes keep it closed.
	for i := 0; i < 10; i++ {
		b.Record(OutcomeSuccess)
	}
	if got := b.State(); got != StateClosed {
		t.Errorf("State = %v, want closed: expired failures still counted", got)
	}

	if s := b.Stats(); s.Failures != 0 {
		t.Errorf("Stats.Failures = %d after the window expired, want 0", s.Failures)
	}
}

func TestBreakerHalfOpenAfterCooldown(t *testing.T) {
	clock := newFakeClock()
	b := testBreaker(clock)

	for i := 0; i < 10; i++ {
		b.Record(OutcomeFailure)
	}
	if b.Allow() {
		t.Fatal("open breaker admitted a request")
	}

	clock.Advance(5 * time.Second)

	if !b.Allow() {
		t.Error("breaker did not admit a probe after the cool-down")
	}
}

func TestBreakerHalfOpenLimitsProbes(t *testing.T) {
	clock := newFakeClock()
	b := testBreaker(clock) // HalfOpenRequests = 2

	for i := 0; i < 10; i++ {
		b.Record(OutcomeFailure)
	}
	clock.Advance(5 * time.Second)

	if !b.Allow() {
		t.Fatal("first probe refused")
	}
	if !b.Allow() {
		t.Fatal("second probe refused")
	}
	if b.Allow() {
		t.Error("third probe admitted; half-open concurrency is not bounded")
	}
}

func TestBreakerClosesAfterSuccessfulProbes(t *testing.T) {
	clock := newFakeClock()
	b := testBreaker(clock)

	for i := 0; i < 10; i++ {
		b.Record(OutcomeFailure)
	}
	clock.Advance(5 * time.Second)

	// Two successful probes (HalfOpenRequests = 2) should close it.
	b.Allow()
	b.Record(OutcomeSuccess)
	b.Allow()
	b.Record(OutcomeSuccess)

	if got := b.State(); got != StateClosed {
		t.Errorf("State = %v after successful probes, want closed", got)
	}
	// The window must be clear, or stale failures would immediately retrip it.
	if s := b.Stats(); s.Failures != 0 {
		t.Errorf("Stats.Failures = %d after closing, want 0", s.Failures)
	}
}

// Closing on a single success would flap: one lucky request should not reopen
// the floodgates onto a backend that is still mostly broken.
func TestBreakerDoesNotCloseOnOneSuccess(t *testing.T) {
	clock := newFakeClock()
	b := testBreaker(clock)

	for i := 0; i < 10; i++ {
		b.Record(OutcomeFailure)
	}
	clock.Advance(5 * time.Second)

	b.Allow()
	b.Record(OutcomeSuccess)

	if got := b.State(); got == StateClosed {
		t.Error("breaker closed after a single successful probe")
	}
}

func TestBreakerReopensOnFailedProbe(t *testing.T) {
	clock := newFakeClock()
	b := testBreaker(clock)

	for i := 0; i < 10; i++ {
		b.Record(OutcomeFailure)
	}
	clock.Advance(5 * time.Second)

	b.Allow()
	b.Record(OutcomeFailure)

	if b.Allow() {
		t.Error("breaker admitted a request right after a failed probe")
	}

	// The cool-down must restart from the failed probe, not the original trip.
	clock.Advance(4 * time.Second)
	if b.Allow() {
		t.Error("cool-down did not restart after the failed probe")
	}
	clock.Advance(2 * time.Second)
	if !b.Allow() {
		t.Error("breaker never re-probed after the restarted cool-down")
	}
}

// Results from requests issued before the breaker opened describe the past and
// must not shorten or extend the cool-down.
func TestBreakerIgnoresLateResultsWhileOpen(t *testing.T) {
	clock := newFakeClock()
	b := testBreaker(clock)

	for i := 0; i < 10; i++ {
		b.Record(OutcomeFailure)
	}
	before := b.Stats().Transitions

	for i := 0; i < 50; i++ {
		b.Record(OutcomeSuccess)
	}

	if got := b.State(); got != StateOpen {
		t.Errorf("State = %v, want open: late successes closed the breaker", got)
	}
	if after := b.Stats().Transitions; after != before {
		t.Errorf("transitions changed from %d to %d on late results", before, after)
	}
}

func TestBreakerReset(t *testing.T) {
	b := testBreaker(newFakeClock())
	for i := 0; i < 10; i++ {
		b.Record(OutcomeFailure)
	}

	b.Reset()

	if got := b.State(); got != StateClosed {
		t.Errorf("State = %v after Reset, want closed", got)
	}
	if s := b.Stats(); s.Total != 0 {
		t.Errorf("Stats.Total = %d after Reset, want 0", s.Total)
	}
}

func TestBreakerStats(t *testing.T) {
	b := testBreaker(newFakeClock())

	for i := 0; i < 5; i++ {
		b.Record(OutcomeSuccess)
	}
	b.Record(OutcomeFailure)
	b.Record(OutcomeTimeout)

	s := b.Stats()
	if s.Total != 7 {
		t.Errorf("Total = %d, want 7", s.Total)
	}
	if s.Failures != 1 || s.Timeouts != 1 {
		t.Errorf("Failures/Timeouts = %d/%d, want 1/1", s.Failures, s.Timeouts)
	}
	if got := s.ErrorRate; got < 0.28 || got > 0.29 {
		t.Errorf("ErrorRate = %.3f, want ~0.286", got)
	}
}

func TestStateString(t *testing.T) {
	tests := map[State]string{
		StateClosed:   "closed",
		StateOpen:     "open",
		StateHalfOpen: "half_open",
		State(99):     "unknown",
	}
	for state, want := range tests {
		if got := state.String(); got != want {
			t.Errorf("State(%d).String() = %q, want %q", state, got, want)
		}
	}
}

// Every request on a backend touches its breaker, so it must be race-free and
// must never admit more than HalfOpenRequests probes under concurrency.
func TestBreakerConcurrentAccess(t *testing.T) {
	clock := newFakeClock()
	b := testBreaker(clock)

	var wg sync.WaitGroup
	for g := 0; g < 16; g++ {
		wg.Add(1)
		go func(g int) {
			defer wg.Done()
			for i := 0; i < 200; i++ {
				if b.Allow() {
					if (g+i)%3 == 0 {
						b.Record(OutcomeFailure)
					} else {
						b.Record(OutcomeSuccess)
					}
				}
				_ = b.Stats()
				_ = b.State()
			}
		}(g)
	}
	wg.Wait()
}

func TestBreakerHalfOpenProbeLimitUnderConcurrency(t *testing.T) {
	clock := newFakeClock()
	b := NewBreaker(BreakerOptions{
		MinRequests:      10,
		ErrorRate:        0.5,
		OpenDuration:     time.Second,
		HalfOpenRequests: 3,
		Now:              clock.Now,
	})

	for i := 0; i < 10; i++ {
		b.Record(OutcomeFailure)
	}
	clock.Advance(2 * time.Second)

	var mu sync.Mutex
	admitted := 0
	var wg sync.WaitGroup
	for i := 0; i < 50; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			if b.Allow() {
				mu.Lock()
				admitted++
				mu.Unlock()
			}
		}()
	}
	wg.Wait()

	if admitted > 3 {
		t.Errorf("admitted %d concurrent probes, want at most 3", admitted)
	}
}

func TestRollingWindowAdvanceBeyondFullWindow(t *testing.T) {
	clock := newFakeClock()
	w := newRollingWindow(5, time.Second, clock.Now)

	w.record(OutcomeFailure)
	clock.Advance(time.Hour) // far beyond the whole window

	total, failures, _ := w.totals()
	if total != 0 || failures != 0 {
		t.Errorf("totals = (%d,%d) after a long gap, want (0,0)", total, failures)
	}
}

func TestRollingWindowPartialExpiry(t *testing.T) {
	clock := newFakeClock()
	w := newRollingWindow(5, time.Second, clock.Now)

	w.record(OutcomeFailure)
	clock.Advance(2 * time.Second)
	w.record(OutcomeSuccess)

	// Both are still inside the 5-second window.
	total, failures, _ := w.totals()
	if total != 2 || failures != 1 {
		t.Errorf("totals = (%d,%d), want (2,1)", total, failures)
	}

	// Push the first record out but keep the second.
	clock.Advance(4 * time.Second)
	total, failures, _ = w.totals()
	if total != 1 || failures != 0 {
		t.Errorf("totals = (%d,%d) after partial expiry, want (1,0)", total, failures)
	}
}

// Permits must be a pure query. Eligibility is checked speculatively for every
// backend on every request, so if that check consumed half-open probe slots,
// the slots would be exhausted by backends that are never dispatched to and the
// breaker could never close again.
func TestPermitsDoesNotConsumeProbeSlots(t *testing.T) {
	clock := newFakeClock()
	b := testBreaker(clock) // HalfOpenRequests = 2

	for i := 0; i < 10; i++ {
		b.Record(OutcomeFailure)
	}
	clock.Advance(5 * time.Second)

	// Thousands of eligibility checks must not burn the probe budget.
	for i := 0; i < 1000; i++ {
		if !b.Permits() {
			t.Fatalf("Permits returned false on iteration %d; it consumed state", i)
		}
	}

	// Both probe slots must still be available.
	if !b.Allow() {
		t.Error("first probe refused after repeated Permits calls")
	}
	if !b.Allow() {
		t.Error("second probe refused after repeated Permits calls")
	}
	if b.Allow() {
		t.Error("third probe admitted; Allow is not bounding probes")
	}
}

func TestPermitsReflectsState(t *testing.T) {
	clock := newFakeClock()
	b := testBreaker(clock)

	if !b.Permits() {
		t.Error("closed breaker did not permit")
	}

	for i := 0; i < 10; i++ {
		b.Record(OutcomeFailure)
	}
	if b.Permits() {
		t.Error("open breaker permitted during cool-down")
	}

	clock.Advance(5 * time.Second)
	if !b.Permits() {
		t.Error("breaker did not permit after the cool-down elapsed")
	}
}

func TestPermitsFalseWhenHalfOpenSlotsBusy(t *testing.T) {
	clock := newFakeClock()
	b := testBreaker(clock) // HalfOpenRequests = 2

	for i := 0; i < 10; i++ {
		b.Record(OutcomeFailure)
	}
	clock.Advance(5 * time.Second)

	b.Allow()
	b.Allow()

	if b.Permits() {
		t.Error("Permits returned true with all probe slots in flight")
	}
}
