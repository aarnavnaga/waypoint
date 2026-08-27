package resilience

import (
	"context"
	"net/http"
	"net/http/httptest"
	"sync"
	"testing"
	"time"

	"github.com/aarnavnaga/waypoint/internal/backend"
	"github.com/aarnavnaga/waypoint/internal/config"
)

func TestRetryBudgetAllowsWithinRatio(t *testing.T) {
	clock := newFakeClock()
	b := NewRetryBudget(RetryBudgetOptions{Percent: 10, TTL: time.Hour, Now: clock.Now})

	// 100 requests at 10% => 10 retries available.
	for i := 0; i < 100; i++ {
		b.Deposit()
	}

	granted := 0
	for i := 0; i < 50; i++ {
		if b.Withdraw() {
			granted++
		}
	}

	if granted > 11 || granted < 9 {
		t.Errorf("granted %d retries from 100 requests at 10%%, want ~10", granted)
	}
}

// The property that matters: when everything is failing, retries must stop
// rather than multiplying load on an already-struggling fleet.
func TestRetryBudgetStopsRetryStorm(t *testing.T) {
	clock := newFakeClock()
	b := NewRetryBudget(RetryBudgetOptions{Percent: 10, TTL: time.Hour, Now: clock.Now})

	totalRequests := 0
	totalRetries := 0

	// Simulate a total outage: every request wants to retry.
	for i := 0; i < 1000; i++ {
		b.Deposit()
		totalRequests++
		for attempt := 0; attempt < 3; attempt++ {
			if !b.Withdraw() {
				break
			}
			totalRetries++
		}
	}

	ratio := float64(totalRetries) / float64(totalRequests)
	t.Logf("retries/requests under total failure: %.3f", ratio)

	// Without a budget this would be 2.0 (two retries per request) — a 3x
	// amplification of load onto a dying backend.
	if ratio > 0.15 {
		t.Errorf("retry ratio %.3f exceeds the 10%% budget: storms are not bounded", ratio)
	}
}

func TestRetryBudgetDeniesWhenEmpty(t *testing.T) {
	clock := newFakeClock()
	b := NewRetryBudget(RetryBudgetOptions{Percent: 10, TTL: time.Hour, Now: clock.Now})

	// No deposits and no floor: nothing to spend.
	for i := 0; i < 20; i++ {
		b.Withdraw()
	}
	if s := b.Stats(); s.Granted != 0 {
		t.Errorf("granted %d retries from an empty budget", s.Granted)
	}
	if s := b.Stats(); s.Denied == 0 {
		t.Error("denials were not counted")
	}
}

func TestRetryBudgetMinimumFloor(t *testing.T) {
	clock := newFakeClock()
	// A low-traffic service must still be able to retry at all.
	b := NewRetryBudget(RetryBudgetOptions{
		Percent:             10,
		MinRetriesPerSecond: 1,
		TTL:                 10 * time.Second,
		Now:                 clock.Now,
	})

	if !b.Withdraw() {
		t.Error("budget floor did not permit any retry on a cold start")
	}
}

func TestRetryBudgetDecays(t *testing.T) {
	clock := newFakeClock()
	b := NewRetryBudget(RetryBudgetOptions{Percent: 100, TTL: time.Second, Now: clock.Now})

	for i := 0; i < 10; i++ {
		b.Deposit()
	}
	before := b.Stats().Tokens
	if before <= 0 {
		t.Fatalf("tokens = %v after deposits, want > 0", before)
	}

	// Tokens must expire, so a quiet period cannot bank budget for a later
	// burst.
	clock.Advance(10 * time.Second)
	b.Deposit()

	if after := b.Stats().Tokens; after >= before {
		t.Errorf("tokens = %v after decay, want below %v", after, before)
	}
}

func TestRetryBudgetCapsAccumulation(t *testing.T) {
	clock := newFakeClock()
	b := NewRetryBudget(RetryBudgetOptions{Percent: 100, TTL: time.Hour, Now: clock.Now})

	for i := 0; i < 100000; i++ {
		b.Deposit()
	}

	// Unbounded accumulation would let one quiet hour fund a huge retry burst.
	if tokens := b.Stats().Tokens; tokens > 1000 {
		t.Errorf("tokens = %v, want capped", tokens)
	}
}

func TestRetryBudgetConcurrent(t *testing.T) {
	b := NewRetryBudget(RetryBudgetOptions{Percent: 50, TTL: time.Hour})

	var wg sync.WaitGroup
	for g := 0; g < 16; g++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for i := 0; i < 500; i++ {
				b.Deposit()
				b.Withdraw()
				_ = b.Stats()
			}
		}()
	}
	wg.Wait()
}

func TestBackoffGrowsExponentially(t *testing.T) {
	b := NewBackoff(10*time.Millisecond, 10*time.Second)
	// Remove jitter so the growth curve itself can be asserted.
	b.rand = func() float64 { return 1.0 }

	tests := []struct {
		attempt int
		want    time.Duration
	}{
		{0, 10 * time.Millisecond},
		{1, 20 * time.Millisecond},
		{2, 40 * time.Millisecond},
		{3, 80 * time.Millisecond},
	}
	for _, tc := range tests {
		if got := b.Delay(tc.attempt); got != tc.want {
			t.Errorf("Delay(%d) = %s, want %s", tc.attempt, got, tc.want)
		}
	}
}

func TestBackoffRespectsMaximum(t *testing.T) {
	b := NewBackoff(10*time.Millisecond, 100*time.Millisecond)
	b.rand = func() float64 { return 1.0 }

	if got := b.Delay(20); got > 100*time.Millisecond {
		t.Errorf("Delay(20) = %s, exceeds the maximum", got)
	}
	// Must not overflow into a negative or absurd duration.
	if got := b.Delay(1000); got < 0 || got > 100*time.Millisecond {
		t.Errorf("Delay(1000) = %s, want a capped positive duration", got)
	}
}

// Full jitter is what decorrelates retries. Without it, every client that failed
// together retries together, reproducing the burst that caused the failure.
func TestBackoffFullJitterSpreadsDelays(t *testing.T) {
	b := NewBackoff(100*time.Millisecond, time.Second)

	seen := make(map[time.Duration]int)
	var min, max time.Duration = time.Hour, 0

	for i := 0; i < 500; i++ {
		d := b.Delay(3)
		seen[d]++
		if d < min {
			min = d
		}
		if d > max {
			max = d
		}
	}

	if len(seen) < 100 {
		t.Errorf("only %d distinct delays in 500 draws; jitter is not spreading retries", len(seen))
	}
	// Draws should cover a wide span of [0, cap), not cluster at the cap.
	if min > 100*time.Millisecond {
		t.Errorf("minimum delay %s is high; jitter does not reach toward zero", min)
	}
	if max < 400*time.Millisecond {
		t.Errorf("maximum delay %s is low; jitter does not reach the cap", max)
	}
}

func TestBackoffNegativeAttempt(t *testing.T) {
	b := NewBackoff(10*time.Millisecond, time.Second)
	if got := b.Delay(-5); got < 0 {
		t.Errorf("Delay(-5) = %s, want a non-negative duration", got)
	}
}

func TestBackoffWaitHonorsContext(t *testing.T) {
	b := NewBackoff(5*time.Second, 10*time.Second)

	ctx, cancel := context.WithTimeout(context.Background(), 20*time.Millisecond)
	defer cancel()

	start := time.Now()
	if b.Wait(ctx, 3) {
		t.Error("Wait reported success despite cancellation")
	}
	if elapsed := time.Since(start); elapsed > time.Second {
		t.Errorf("Wait blocked %s past cancellation", elapsed)
	}
}

func TestBackoffWaitCompletes(t *testing.T) {
	b := NewBackoff(time.Millisecond, 2*time.Millisecond)
	if !b.Wait(context.Background(), 0) {
		t.Error("Wait reported failure on an uncancelled context")
	}
}

// ---------- health checker ----------

func healthPool(t *testing.T, url string) (*backend.Pool, *backend.Backend) {
	t.Helper()

	pool, err := backend.NewPool([]config.Backend{{Name: "a", URL: url}})
	if err != nil {
		t.Fatalf("NewPool: %v", err)
	}
	b, _ := pool.Get("a")
	return pool, b
}

func TestHealthCheckerMarksUnhealthyAfterThreshold(t *testing.T) {
	healthy := true
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		if !healthy {
			w.WriteHeader(http.StatusServiceUnavailable)
			return
		}
		w.WriteHeader(http.StatusOK)
	}))
	defer srv.Close()

	pool, b := healthPool(t, srv.URL)
	hc := NewHealthChecker(pool, HealthCheckerOptions{
		UnhealthyThreshold: 3,
		HealthyThreshold:   2,
		Client:             srv.Client(),
		Logger:             quietLogger(),
	})

	healthy = false
	ctx := context.Background()

	// Hysteresis: two failures must not be enough.
	hc.Check(ctx, b)
	hc.Check(ctx, b)
	if !b.Healthy() {
		t.Error("backend removed after 2 failures, threshold is 3")
	}

	hc.Check(ctx, b)
	if b.Healthy() {
		t.Error("backend still healthy after 3 consecutive failures")
	}
}

func TestHealthCheckerRestoresAfterThreshold(t *testing.T) {
	healthy := false
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		if !healthy {
			w.WriteHeader(http.StatusServiceUnavailable)
			return
		}
		w.WriteHeader(http.StatusOK)
	}))
	defer srv.Close()

	pool, b := healthPool(t, srv.URL)
	hc := NewHealthChecker(pool, HealthCheckerOptions{
		UnhealthyThreshold: 2,
		HealthyThreshold:   2,
		Client:             srv.Client(),
		Logger:             quietLogger(),
	})

	ctx := context.Background()
	hc.Check(ctx, b)
	hc.Check(ctx, b)
	if b.Healthy() {
		t.Fatal("backend should be unhealthy")
	}

	healthy = true
	hc.Check(ctx, b)
	if b.Healthy() {
		t.Error("backend restored after a single success, threshold is 2")
	}
	hc.Check(ctx, b)
	if !b.Healthy() {
		t.Error("backend not restored after 2 consecutive successes")
	}
}

// A single flapping probe must not flip a backend out of rotation, because
// oscillating traffic is itself a cause of timeouts.
func TestHealthCheckerResetsCountersOnMixedResults(t *testing.T) {
	healthy := true
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		if !healthy {
			w.WriteHeader(http.StatusServiceUnavailable)
			return
		}
		w.WriteHeader(http.StatusOK)
	}))
	defer srv.Close()

	pool, b := healthPool(t, srv.URL)
	hc := NewHealthChecker(pool, HealthCheckerOptions{
		UnhealthyThreshold: 3,
		Client:             srv.Client(),
		Logger:             quietLogger(),
	})
	ctx := context.Background()

	for i := 0; i < 10; i++ {
		healthy = false
		hc.Check(ctx, b)
		hc.Check(ctx, b)
		healthy = true
		hc.Check(ctx, b) // resets the failure streak
	}

	if !b.Healthy() {
		t.Error("intermittent failures removed a backend that never failed 3 times consecutively")
	}
}

func TestHealthCheckerTreatsUnreachableAsFailure(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusOK)
	}))
	url := srv.URL
	srv.Close() // nothing is listening now

	pool, b := healthPool(t, url)
	hc := NewHealthChecker(pool, HealthCheckerOptions{
		UnhealthyThreshold: 1,
		Timeout:            200 * time.Millisecond,
		Logger:             quietLogger(),
	})

	hc.Check(context.Background(), b)
	if b.Healthy() {
		t.Error("unreachable backend reported healthy")
	}
}

func TestHealthCheckerRunStops(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusOK)
	}))
	defer srv.Close()

	pool, _ := healthPool(t, srv.URL)
	hc := NewHealthChecker(pool, HealthCheckerOptions{
		Interval: 5 * time.Millisecond,
		Client:   srv.Client(),
		Logger:   quietLogger(),
	})

	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan struct{})
	go func() {
		hc.Run(ctx)
		close(done)
	}()

	// Sample probe state before cancelling: a probe interrupted by cancellation
	// is intentionally not recorded, so reading after cancel would race.
	deadline := time.Now().Add(2 * time.Second)
	var probes int
	for time.Now().Before(deadline) {
		if ok, _ := hc.ProbeState("a"); ok > 0 {
			probes = ok
			break
		}
		time.Sleep(time.Millisecond)
	}
	if probes == 0 {
		t.Error("no successful probes were recorded")
	}

	cancel()
	select {
	case <-done:
	case <-time.After(time.Second):
		t.Fatal("Run did not stop on context cancellation")
	}
}

// Shutting down must not make healthy backends look sick: a probe aborted by
// the caller's own cancellation says nothing about the backend.
func TestHealthCheckerIgnoresProbesCancelledByCaller(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		<-r.Context().Done()
	}))
	defer srv.Close()

	pool, b := healthPool(t, srv.URL)
	hc := NewHealthChecker(pool, HealthCheckerOptions{
		UnhealthyThreshold: 1,
		Timeout:            5 * time.Second,
		Client:             srv.Client(),
		Logger:             quietLogger(),
	})

	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	hc.Check(ctx, b)

	if !b.Healthy() {
		t.Error("a probe cancelled by the caller marked the backend unhealthy")
	}
	if _, fail := hc.ProbeState("a"); fail != 0 {
		t.Errorf("consecutiveFail = %d after a cancelled probe, want 0", fail)
	}
}
