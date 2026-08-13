package mocksim

import (
	"math/rand"
	"sync"
	"time"
)

// FaultState describes injected failure behavior. It is the JSON body of the
// mock backend's /admin/faults endpoint.
type FaultState struct {
	// ErrorRate is the probability in [0,1] of returning HTTP 500.
	ErrorRate float64 `json:"error_rate" yaml:"error_rate"`
	// ExtraLatencyMS is added before the response begins.
	ExtraLatencyMS int `json:"extra_latency_ms" yaml:"extra_latency_ms"`
	// Hang makes the backend accept the request and never respond, which
	// exercises TTFT timeouts rather than connection errors.
	Hang bool `json:"hang" yaml:"hang"`
	// SlowTokens multiplies per-token decode cost, simulating a degraded peer.
	SlowTokens float64 `json:"slow_tokens" yaml:"slow_tokens"`
}

// Faults holds mutable fault-injection state.
//
// Toxiproxy covers transport-level faults (connection drops, bandwidth, added
// latency). These knobs cover *application*-level faults that a proxy cannot
// express: a server that returns 500s, or one that accepts a request and then
// stalls forever. Both are needed for the M8 resilience experiment.
type Faults struct {
	mu    sync.RWMutex
	state FaultState

	rngMu sync.Mutex
	rng   *rand.Rand
}

// NewFaults returns a Faults with the given initial state.
func NewFaults(initial FaultState, seed int64) *Faults {
	return &Faults{
		state: initial,
		rng:   rand.New(rand.NewSource(seed)), //nolint:gosec // simulation, not crypto
	}
}

// Set replaces the fault state.
func (f *Faults) Set(s FaultState) {
	if s.ErrorRate < 0 {
		s.ErrorRate = 0
	}
	if s.ErrorRate > 1 {
		s.ErrorRate = 1
	}
	if s.SlowTokens < 0 {
		s.SlowTokens = 0
	}

	f.mu.Lock()
	defer f.mu.Unlock()
	f.state = s
}

// Get returns the current fault state.
func (f *Faults) Get() FaultState {
	f.mu.RLock()
	defer f.mu.RUnlock()
	return f.state
}

// Clear resets all fault injection.
func (f *Faults) Clear() { f.Set(FaultState{}) }

// ShouldError reports whether this request should fail, sampling the configured
// error rate.
func (f *Faults) ShouldError() bool {
	rate := f.Get().ErrorRate
	if rate <= 0 {
		return false
	}
	if rate >= 1 {
		return true
	}

	f.rngMu.Lock()
	defer f.rngMu.Unlock()
	return f.rng.Float64() < rate
}

// ExtraLatency returns the configured pre-response delay.
func (f *Faults) ExtraLatency() time.Duration {
	return time.Duration(f.Get().ExtraLatencyMS) * time.Millisecond
}

// ShouldHang reports whether to stall instead of responding.
func (f *Faults) ShouldHang() bool { return f.Get().Hang }

// DecodeMultiplier returns the per-token slowdown factor (1.0 when healthy).
func (f *Faults) DecodeMultiplier() float64 {
	m := f.Get().SlowTokens
	if m <= 0 {
		return 1.0
	}
	return m
}
