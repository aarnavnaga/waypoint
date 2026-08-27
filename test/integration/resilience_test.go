//go:build integration

package integration

import (
	"encoding/json"
	"fmt"
	"net"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/aarnavnaga/waypoint/internal/admission"
	"github.com/aarnavnaga/waypoint/internal/api"
	"github.com/aarnavnaga/waypoint/internal/backend"
	"github.com/aarnavnaga/waypoint/internal/config"
	"github.com/aarnavnaga/waypoint/internal/mocksim"
	"github.com/aarnavnaga/waypoint/internal/openai"
	"github.com/aarnavnaga/waypoint/internal/resilience"
	"github.com/aarnavnaga/waypoint/internal/router"
	"github.com/aarnavnaga/waypoint/internal/tokenizer"
)

// toxiproxyURL is where the control API is expected. Start it with:
//
//	docker run -d --name waypoint-toxiproxy -p 8474:8474 -p 19100-19110:19100-19110 \
//	  ghcr.io/shopify/toxiproxy:2.9.0
func toxiproxyURL() string {
	if v := os.Getenv("TOXIPROXY_URL"); v != "" {
		return v
	}
	return "http://localhost:8474"
}

// toxicStack is a gateway whose backends are reached through Toxiproxy, so
// transport-level faults can be injected between gateway and backend.
type toxicStack struct {
	*stack
	toxi       *ToxiproxyClient
	proxyNames []string
	breakers   *api.Breakers
}

// newToxicStack builds backends, fronts each with a Toxiproxy proxy, and points
// the gateway at the proxy addresses.
func newToxicStack(t *testing.T, backendCount int, maxAttempts int) *toxicStack {
	t.Helper()

	toxi := NewToxiproxyClient(toxiproxyURL())
	if !toxi.Available() {
		t.Skipf("Toxiproxy is not reachable at %s; start it with:\n"+
			"  docker run -d --name waypoint-toxiproxy -p 8474:8474 -p 19100-19110:19100-19110 ghcr.io/shopify/toxiproxy:2.9.0",
			toxiproxyURL())
	}

	ts := &toxicStack{stack: &stack{client: &http.Client{Timeout: 30 * time.Second}}, toxi: toxi}
	cfgBackends := make([]config.Backend, 0, backendCount)

	for i := 0; i < backendCount; i++ {
		mcfg := mocksim.DefaultConfig()
		mcfg.Name = fmt.Sprintf("mock-%d", i+1)
		mcfg.Seed = int64(i + 1)
		mcfg.Prefill.Base = mocksim.Duration{Duration: time.Millisecond}
		mcfg.Prefill.PerToken = mocksim.Duration{Duration: 20 * time.Microsecond}
		mcfg.Decode.TokensPerSecond = 2000
		mcfg.Output.MinTokens, mcfg.Output.MaxTokens = 8, 8

		engine := mocksim.NewEngine(mcfg)

		// Bind the backend to a host-visible address so the container can reach
		// it. Toxiproxy runs in Docker, so it needs host.docker.internal.
		ln, err := net.Listen("tcp", "0.0.0.0:0")
		if err != nil {
			t.Fatalf("listen: %v", err)
		}
		srv := &httptest.Server{
			Listener: ln,
			Config:   &http.Server{Handler: mocksim.NewServer(engine, quietLogger()).Handler()},
		}
		srv.Start()
		t.Cleanup(srv.Close)

		backendPort := ln.Addr().(*net.TCPAddr).Port
		proxyName := fmt.Sprintf("waypoint-test-%s-%d", mcfg.Name, time.Now().UnixNano())
		proxyPort := 19100 + i

		_, err = toxi.CreateProxy(proxyName,
			fmt.Sprintf("0.0.0.0:%d", proxyPort),
			fmt.Sprintf("host.docker.internal:%d", backendPort))
		if err != nil {
			t.Skipf("could not create Toxiproxy proxy (is Docker able to reach the host?): %v", err)
		}
		t.Cleanup(func() { _ = toxi.DeleteProxy(proxyName) })

		ts.backends = append(ts.backends, srv)
		ts.engines = append(ts.engines, engine)
		ts.proxyNames = append(ts.proxyNames, proxyName)

		cfgBackends = append(cfgBackends, config.Backend{
			Name:              mcfg.Name,
			URL:               fmt.Sprintf("http://localhost:%d", proxyPort),
			MaxInflightTokens: 200000,
		})
	}

	cfg := config.Default()
	cfg.Backends = cfgBackends
	cfg.Auth.APIKeys = []string{"integration-key"}
	cfg.Resilience.MaxAttempts = maxAttempts
	cfg.Resilience.TTFTTimeout = config.Duration{Duration: 1500 * time.Millisecond}
	cfg.Resilience.ConnectTimeout = config.Duration{Duration: 500 * time.Millisecond}
	cfg.Resilience.Breaker.MinRequests = 5
	cfg.Resilience.Breaker.OpenDuration = config.Duration{Duration: 2 * time.Second}
	cfg.Resilience.Breaker.HalfOpenRequests = 2

	pool, err := backend.NewPool(cfg.Backends)
	if err != nil {
		t.Fatalf("NewPool: %v", err)
	}
	ts.pool = pool

	est := tokenizer.NewCharHeuristic()
	rt, err := router.New(cfg.Routing, est)
	if err != nil {
		t.Fatalf("router.New: %v", err)
	}
	ts.router = rt

	proxy := api.NewProxy(&http.Client{
		Transport: &http.Transport{
			DialContext:         (&net.Dialer{Timeout: 500 * time.Millisecond}).DialContext,
			DisableCompression:  true,
			MaxIdleConnsPerHost: 32,
		},
	}, api.Timeouts{TTFT: cfg.Resilience.TTFTTimeout.Duration}, quietLogger())

	breakers := api.NewBreakers(pool, cfg.Resilience.Breaker)
	ts.breakers = breakers

	budget := resilience.NewRetryBudget(resilience.RetryBudgetOptions{
		Percent: 50, MinRetriesPerSecond: 100,
	})

	dispatcher := api.NewDispatcher(api.DispatcherOptions{
		Pool: pool, Router: rt, Proxy: proxy,
		Breakers:    breakers,
		Budget:      budget,
		Backoff:     resilience.NewBackoff(time.Millisecond, 10*time.Millisecond),
		MaxAttempts: maxAttempts,
		Logger:      quietLogger(),
	})

	srv := api.NewServer(api.Options{
		Config: cfg, Pool: pool, Router: rt, Proxy: proxy,
		Dispatcher: dispatcher, Breakers: breakers, Budget: budget,
		Admission: admission.New(admission.Options{MaxConcurrent: 64, MaxQueue: 256}),
		Estimator: est, Logger: quietLogger(), Version: "integration",
	})

	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("listen: %v", err)
	}
	httpSrv := &http.Server{Handler: srv.Handler(), ReadHeaderTimeout: 5 * time.Second}
	go func() { _ = httpSrv.Serve(ln) }()
	t.Cleanup(func() { _ = httpSrv.Close() })

	ts.gatewayURL = "http://" + ln.Addr().String()
	waitReady(t, ts.stack)
	return ts
}

func (ts *toxicStack) chatStatus(t *testing.T, prompt string) int {
	t.Helper()
	resp := ts.chat(t, openai.ChatRequest{
		Model:    "mock-7b",
		Messages: []openai.Message{{Role: openai.RoleUser, Content: prompt}},
	})
	return resp.StatusCode
}

// SPEC §8 resilience scenario (a): a backend disappears partway through a run.
// Toxiproxy severs the connection at the transport layer, which is a different
// failure from an HTTP error — the gateway sees a dead socket, not a 5xx.
func TestToxiproxyBackendKilledMidRun(t *testing.T) {
	ts := newToxicStack(t, 3, 3)

	// Baseline: everything healthy.
	for i := 0; i < 6; i++ {
		if got := ts.chatStatus(t, fmt.Sprintf("warmup %d", i)); got != http.StatusOK {
			t.Fatalf("warmup %d: status %d", i, got)
		}
	}

	// Kill one backend's transport entirely.
	if err := ts.toxi.SetEnabled(ts.proxyNames[0], false); err != nil {
		t.Fatalf("disable proxy: %v", err)
	}

	succeeded, failed := 0, 0
	for i := 0; i < 30; i++ {
		if ts.chatStatus(t, fmt.Sprintf("after kill %d", i)) == http.StatusOK {
			succeeded++
		} else {
			failed++
		}
	}

	availability := float64(succeeded) / float64(succeeded+failed)
	t.Logf("availability with 1 of 3 backends killed: %.1f%% (%d ok, %d failed)",
		availability*100, succeeded, failed)

	// Retries onto the two healthy peers should keep availability at 100%.
	if availability < 1.0 {
		t.Errorf("availability %.1f%%, want 100%%: failover did not cover the dead backend",
			availability*100)
	}

	// Restore and confirm the fleet recovers.
	if err := ts.toxi.SetEnabled(ts.proxyNames[0], true); err != nil {
		t.Fatalf("re-enable proxy: %v", err)
	}
	for i := 0; i < 5; i++ {
		if got := ts.chatStatus(t, fmt.Sprintf("after restore %d", i)); got != http.StatusOK {
			t.Errorf("status %d after restoring the backend", got)
		}
	}
}

// SPEC §8 resilience scenario (b): one backend becomes very slow.
// The TTFT deadline must convert that into a fast failure and a retry elsewhere,
// rather than letting every request inherit the slow backend's latency.
func TestToxiproxySlowBackendIsBypassed(t *testing.T) {
	ts := newToxicStack(t, 3, 3)

	// 2s of added latency against a 1.5s TTFT deadline.
	if err := ts.toxi.AddLatency(ts.proxyNames[0], "slow", 2000, 0); err != nil {
		t.Fatalf("add latency: %v", err)
	}
	t.Cleanup(func() { _ = ts.toxi.RemoveToxic(ts.proxyNames[0], "slow") })

	var slowest time.Duration
	succeeded := 0

	for i := 0; i < 15; i++ {
		start := time.Now()
		status := ts.chatStatus(t, fmt.Sprintf("slow test %d", i))
		elapsed := time.Since(start)

		if status == http.StatusOK {
			succeeded++
		}
		if elapsed > slowest {
			slowest = elapsed
		}
	}

	t.Logf("with one backend 2s slow: %d/15 succeeded, worst latency %s", succeeded, slowest)

	if succeeded < 14 {
		t.Errorf("%d/15 succeeded, want ~15: the slow backend was not bypassed", succeeded)
	}
	// A request that hit the slow backend should fail its TTFT deadline and be
	// retried, so worst-case latency is bounded well below 2 x 2s.
	if slowest > 4*time.Second {
		t.Errorf("worst latency %s; the TTFT deadline is not bounding slow backends", slowest)
	}
}

// The breaker should notice a backend that keeps failing and stop sending it
// traffic, which is visible as the backend becoming unavailable for routing.
func TestToxiproxyBreakerOpensOnDeadBackend(t *testing.T) {
	ts := newToxicStack(t, 3, 1) // no retries, so failures are visible

	if err := ts.toxi.SetEnabled(ts.proxyNames[0], false); err != nil {
		t.Fatalf("disable proxy: %v", err)
	}

	breaker := ts.breakers.For("mock-1")
	tripped := false
	tripAfter := 0

	for i := 0; i < 80; i++ {
		ts.chatStatus(t, fmt.Sprintf("trip %d", i))
		if breaker.State() == resilience.StateOpen {
			tripped = true
			tripAfter = i + 1
			break
		}
	}

	if !tripped {
		t.Fatalf("breaker never opened against a dead backend: %+v", breaker.Stats())
	}
	t.Logf("breaker opened after %d requests", tripAfter)

	b, _ := ts.pool.Get("mock-1")
	if b.Available() {
		t.Error("backend with an open breaker is still eligible for routing")
	}

	// With the dead backend excluded, everything should now succeed.
	for i := 0; i < 10; i++ {
		if got := ts.chatStatus(t, fmt.Sprintf("post trip %d", i)); got != http.StatusOK {
			t.Errorf("status %d after the breaker opened", got)
		}
	}
}

// Recovery: once the backend comes back, half-open probes should close the
// circuit and return it to rotation.
func TestToxiproxyBreakerRecovers(t *testing.T) {
	ts := newToxicStack(t, 2, 1)

	if err := ts.toxi.SetEnabled(ts.proxyNames[0], false); err != nil {
		t.Fatalf("disable proxy: %v", err)
	}

	breaker := ts.breakers.For("mock-1")
	for i := 0; i < 80 && breaker.State() != resilience.StateOpen; i++ {
		ts.chatStatus(t, fmt.Sprintf("trip %d", i))
	}
	if breaker.State() != resilience.StateOpen {
		t.Fatalf("breaker did not open: %+v", breaker.Stats())
	}
	openedAt := time.Now()

	// Heal the backend and wait out the cool-down.
	if err := ts.toxi.SetEnabled(ts.proxyNames[0], true); err != nil {
		t.Fatalf("re-enable proxy: %v", err)
	}
	time.Sleep(2500 * time.Millisecond)

	for i := 0; i < 60 && breaker.State() != resilience.StateClosed; i++ {
		ts.chatStatus(t, fmt.Sprintf("recover %d", i))
	}

	if got := breaker.State(); got != resilience.StateClosed {
		t.Errorf("breaker state = %v after recovery, want closed (%+v)", got, breaker.Stats())
	}
	t.Logf("breaker recovered %s after opening", time.Since(openedAt).Round(time.Millisecond))
}

// SPEC §8 resilience scenario (c): one backend returns errors for a share of
// requests. Application-level faults come from the mock itself, since Toxiproxy
// operates below HTTP and cannot synthesize a 500.
func TestBackendErrorRateIsAbsorbedByRetries(t *testing.T) {
	s := newStack(t, 3, nil) // MaxAttempts defaults to 3

	s.engines[0].Faults().Set(mocksim.FaultState{ErrorRate: 0.3})

	succeeded := 0
	const total = 40
	for i := 0; i < total; i++ {
		resp := s.chat(t, openai.ChatRequest{
			Model:    "mock-7b",
			Messages: []openai.Message{{Role: openai.RoleUser, Content: fmt.Sprintf("err %d", i)}},
		})
		if resp.StatusCode == http.StatusOK {
			succeeded++
		}
	}

	availability := float64(succeeded) / total
	t.Logf("availability with one backend at a 30%% error rate: %.1f%%", availability*100)

	// Without retries roughly 10% of traffic would fail: a third of requests
	// reach the bad backend and 30% of those error. With up to 3 attempts on
	// distinct backends, a request only fails if every attempt errors, which is
	// vanishingly unlikely when two of three backends are healthy.
	if availability < 0.98 {
		t.Errorf("availability %.1f%%, want ~100%%: retries are not absorbing a partial failure",
			availability*100)
	}
}

func TestToxiproxyClientSkipsCleanlyWhenUnavailable(t *testing.T) {
	c := NewToxiproxyClient("http://127.0.0.1:1")
	if c.Available() {
		t.Error("Available reported true against a dead address")
	}
}

func TestResilienceReportsRetryBudgetInAdmin(t *testing.T) {
	ts := newToxicStack(t, 2, 3)

	resp, err := ts.client.Get(ts.gatewayURL + "/admin/backends")
	if err != nil {
		t.Fatalf("GET /admin/backends: %v", err)
	}
	defer func() { _ = resp.Body.Close() }()

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
	for name, st := range body.Breakers {
		if !strings.Contains("closed open half_open", st.State) {
			t.Errorf("backend %s has unexpected breaker state %q", name, st.State)
		}
	}
}
