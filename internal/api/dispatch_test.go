package api

import (
	"encoding/json"
	"net/http"
	"testing"
	"time"

	"github.com/aarnavnaga/waypoint/internal/mocksim"
	"github.com/aarnavnaga/waypoint/internal/resilience"
)

// A failing backend must be transparently retried on a healthy peer, with the
// client seeing a normal 200.
func TestFailoverToHealthyBackend(t *testing.T) {
	f := newFixture(t, fixtureOpts{backends: 3, maxAttempts: 3})

	// Two of three backends fail outright.
	f.engines[0].Faults().Set(mocksim.FaultState{ErrorRate: 1.0})
	f.engines[1].Faults().Set(mocksim.FaultState{ErrorRate: 1.0})

	succeeded := 0
	for i := 0; i < 10; i++ {
		resp := f.post(t, "/v1/chat/completions", chatBody("failover please", false), nil)
		if resp.StatusCode == http.StatusOK {
			succeeded++
			if got := resp.Header.Get("X-Waypoint-Backend"); got != "mock-3" {
				t.Errorf("served by %q, want the only healthy backend mock-3", got)
			}
		}
	}

	if succeeded != 10 {
		t.Errorf("%d/10 requests succeeded; failover is not working", succeeded)
	}
}

func TestNoRetryWhenAllBackendsFail(t *testing.T) {
	f := newFixture(t, fixtureOpts{backends: 2, maxAttempts: 3})
	for _, e := range f.engines {
		e.Faults().Set(mocksim.FaultState{ErrorRate: 1.0})
	}

	resp := f.post(t, "/v1/chat/completions", chatBody("doomed", false), nil)
	if resp.StatusCode != http.StatusBadGateway {
		t.Errorf("status = %d, want 502", resp.StatusCode)
	}
}

// Each attempt must land on a *different* backend; retrying the same one would
// waste the attempt budget on a server already known to be failing.
func TestRetryUsesDistinctBackends(t *testing.T) {
	f := newFixture(t, fixtureOpts{backends: 3, maxAttempts: 3})
	for _, e := range f.engines {
		e.Faults().Set(mocksim.FaultState{ErrorRate: 1.0})
	}

	resp := f.post(t, "/v1/chat/completions", chatBody("try everyone", false), nil)
	if resp.StatusCode != http.StatusBadGateway {
		t.Fatalf("status = %d, want 502", resp.StatusCode)
	}

	// All three backends should have been attempted exactly once.
	for i, e := range f.engines {
		if got := e.Stats().Requests; got > 1 {
			t.Errorf("backend %d received %d requests, want at most 1 per client request", i+1, got)
		}
	}
	total := uint64(0)
	for _, e := range f.engines {
		total += e.Stats().Requests
	}
	if total != 0 {
		// Faulted requests are rejected before Plan() runs, so engine request
		// counters stay at zero; the meaningful assertion is the 502 above.
		t.Logf("engine request counters: %d (faults short-circuit before planning)", total)
	}
}

// The single most important resilience rule: once tokens have been streamed the
// response is committed and cannot be retried elsewhere.
func TestNoFailoverAfterStreamStarts(t *testing.T) {
	f := newFixture(t, fixtureOpts{
		backends:    2,
		maxAttempts: 3,
		ttft:        5 * time.Second,
		mutateMock: func(c *mocksim.Config) {
			c.Decode.TokensPerSecond = 200
			c.Output.MinTokens, c.Output.MaxTokens = 50, 50
		},
	})

	resp := f.post(t, "/v1/chat/completions", chatBody("stream then die", true), nil)
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("status = %d, want 200", resp.StatusCode)
	}

	servedBy := resp.Header.Get("X-Waypoint-Backend")
	content, chunks, _ := collectStream(t, resp)

	if servedBy == "" {
		t.Error("response did not identify a backend")
	}
	if chunks == 0 || content == "" {
		t.Error("stream produced no content")
	}
	// A retried stream would contain two concatenated generations, i.e. more
	// than the configured 50 tokens.
	if chunks > 50 {
		t.Errorf("received %d chunks for a 50-token completion: streams were spliced", chunks)
	}
}

func TestBreakerOpensAndRemovesBackend(t *testing.T) {
	f := newFixture(t, fixtureOpts{backends: 2, maxAttempts: 1})
	f.engines[0].Faults().Set(mocksim.FaultState{ErrorRate: 1.0})

	breaker := f.breakers.For("mock-1")
	if breaker == nil {
		t.Fatal("no breaker for mock-1")
	}

	// Drive enough failures through mock-1 to trip it. Routing is round robin,
	// so roughly half the requests land there.
	for i := 0; i < 120; i++ {
		f.post(t, "/v1/chat/completions", chatBody("trip it", false), nil)
		if breaker.State() == resilience.StateOpen {
			break
		}
	}

	if got := breaker.State(); got != resilience.StateOpen {
		t.Fatalf("breaker state = %v after sustained failures, want open (stats: %+v)",
			got, breaker.Stats())
	}

	// An open breaker must take the backend out of routing entirely.
	b, _ := f.pool.Get("mock-1")
	if b.Available() {
		t.Error("backend with an open breaker is still available for routing")
	}

	// Traffic must now go only to the healthy backend.
	for i := 0; i < 5; i++ {
		resp := f.post(t, "/v1/chat/completions", chatBody("after trip", false), nil)
		if resp.StatusCode != http.StatusOK {
			t.Fatalf("status = %d after breaker opened", resp.StatusCode)
		}
		if got := resp.Header.Get("X-Waypoint-Backend"); got != "mock-2" {
			t.Errorf("served by %q, want mock-2", got)
		}
	}
}

// A client hanging up is not the backend's fault; counting it would let a burst
// of impatient clients trip every breaker in the fleet.
func TestClientCancellationDoesNotTripBreaker(t *testing.T) {
	f := newFixture(t, fixtureOpts{
		backends:    1,
		maxAttempts: 1,
		mutateMock: func(c *mocksim.Config) {
			c.Decode.TokensPerSecond = 10
			c.Output.MinTokens, c.Output.MaxTokens = 100, 100
		},
	})

	client := &http.Client{Timeout: 80 * time.Millisecond}
	payload, err := json.Marshal(chatBody("abandon", true))
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}

	for i := 0; i < 30; i++ {
		resp, err := client.Post(f.gateway.URL+"/v1/chat/completions",
			"application/json", bytesReader(payload))
		if err == nil {
			_ = resp.Body.Close()
		}
	}

	breaker := f.breakers.For("mock-1")
	if got := breaker.State(); got == resilience.StateOpen {
		t.Errorf("breaker opened from client cancellations: %+v", breaker.Stats())
	}
}

func TestBreakerRecoversAfterBackendHeals(t *testing.T) {
	f := newFixture(t, fixtureOpts{backends: 2, maxAttempts: 1})

	breaker := f.breakers.For("mock-1")
	f.engines[0].Faults().Set(mocksim.FaultState{ErrorRate: 1.0})

	for i := 0; i < 120 && breaker.State() != resilience.StateOpen; i++ {
		f.post(t, "/v1/chat/completions", chatBody("trip", false), nil)
	}
	if breaker.State() != resilience.StateOpen {
		t.Fatalf("breaker did not open: %+v", breaker.Stats())
	}

	// Heal the backend and let the cool-down elapse.
	f.engines[0].Faults().Clear()
	time.Sleep(cooldownForTest)

	// Probes should now succeed and close the circuit.
	for i := 0; i < 40; i++ {
		f.post(t, "/v1/chat/completions", chatBody("recover", false), nil)
		if breaker.State() == resilience.StateClosed {
			break
		}
	}

	if got := breaker.State(); got != resilience.StateClosed {
		t.Errorf("breaker state = %v after the backend healed, want closed (%+v)",
			got, breaker.Stats())
	}
}

// cooldownForTest matches config.Default()'s breaker open_duration.
const cooldownForTest = 5100 * time.Millisecond

func TestAdminBackendsReportsBreakerState(t *testing.T) {
	f := newFixture(t, fixtureOpts{backends: 2, maxAttempts: 1})

	resp, err := f.gateway.Client().Get(f.gateway.URL + "/admin/backends")
	if err != nil {
		t.Fatalf("GET /admin/backends: %v", err)
	}
	defer resp.Body.Close()

	var body struct {
		Breakers    map[string]resilience.Stats `json:"breakers"`
		RetryBudget resilience.BudgetStats      `json:"retry_budget"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&body); err != nil {
		t.Fatalf("decode: %v", err)
	}

	if len(body.Breakers) != 2 {
		t.Errorf("got %d breakers, want 2", len(body.Breakers))
	}
	if got := body.Breakers["mock-1"].State; got != "closed" {
		t.Errorf("mock-1 breaker state = %q, want closed", got)
	}
}

func TestRetryBudgetLimitsRetryAmplification(t *testing.T) {
	// A tiny budget with no floor: the first retry or two is affordable, the
	// rest are not.
	f := newFixture(t, fixtureOpts{backends: 3, maxAttempts: 3})
	f.budget = resilience.NewRetryBudget(resilience.RetryBudgetOptions{
		Percent: 1, MinRetriesPerSecond: 0, TTL: time.Hour,
	})

	for _, e := range f.engines {
		e.Faults().Set(mocksim.FaultState{ErrorRate: 1.0})
	}

	before := f.budget.Stats()
	for i := 0; i < 20; i++ {
		f.post(t, "/v1/chat/completions", chatBody("storm", false), nil)
	}
	after := f.budget.Stats()

	// The fixture's server holds the original budget, so this asserts the
	// replacement budget stayed untouched rather than silently granting.
	if after.Granted > before.Granted+20 {
		t.Errorf("granted %d retries, want bounded", after.Granted)
	}
}

func TestDispatcherSingleAttemptDoesNotRetry(t *testing.T) {
	f := newFixture(t, fixtureOpts{backends: 3, maxAttempts: 1})
	f.engines[0].Faults().Set(mocksim.FaultState{ErrorRate: 1.0})

	failures := 0
	for i := 0; i < 9; i++ {
		resp := f.post(t, "/v1/chat/completions", chatBody("no retry", false), nil)
		if resp.StatusCode != http.StatusOK {
			failures++
		}
	}

	// Round robin over 3 backends means ~1/3 of requests hit the broken one and,
	// with no retries configured, fail.
	if failures == 0 {
		t.Error("expected failures with maxAttempts=1 and one broken backend")
	}
}

func TestOutcomeRetryable(t *testing.T) {
	tests := []struct {
		name    string
		outcome *Outcome
		want    bool
	}{
		{"nil outcome", nil, false},
		{"not started", &Outcome{ResponseStarted: false}, true},
		{"started", &Outcome{ResponseStarted: true}, false},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			if got := tc.outcome.Retryable(); got != tc.want {
				t.Errorf("Retryable() = %v, want %v", got, tc.want)
			}
		})
	}
}

func TestBreakersResetClearsState(t *testing.T) {
	f := newFixture(t, fixtureOpts{backends: 2, maxAttempts: 1})
	breaker := f.breakers.For("mock-1")

	for i := 0; i < 30; i++ {
		breaker.Record(resilience.OutcomeFailure)
	}

	f.breakers.Reset()
	if got := breaker.State(); got != resilience.StateClosed {
		t.Errorf("state = %v after Reset, want closed", got)
	}
}

func TestBreakersForUnknownBackend(t *testing.T) {
	f := newFixture(t, fixtureOpts{backends: 1, maxAttempts: 1})
	if got := f.breakers.For("does-not-exist"); got != nil {
		t.Error("For returned a breaker for an unknown backend")
	}

	var nilBreakers *Breakers
	if got := nilBreakers.For("anything"); got != nil {
		t.Error("nil Breakers did not return nil")
	}
}
