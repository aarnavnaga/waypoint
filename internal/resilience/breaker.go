// Package resilience implements circuit breaking, health checking and bounded
// retries — the machinery that keeps one sick backend from degrading the fleet.
package resilience

import (
	"errors"
	"fmt"
	"sync"
	"time"
)

// ErrBreakerOpen is returned when a breaker refuses a request.
var ErrBreakerOpen = errors.New("circuit breaker is open")

// State is a circuit breaker's state.
type State int

const (
	// StateClosed passes all requests; the backend is believed healthy.
	StateClosed State = iota
	// StateOpen rejects all requests; the backend is believed broken.
	StateOpen
	// StateHalfOpen admits a few probes to test whether recovery has happened.
	StateHalfOpen
)

// String implements fmt.Stringer.
func (s State) String() string {
	switch s {
	case StateClosed:
		return "closed"
	case StateOpen:
		return "open"
	case StateHalfOpen:
		return "half_open"
	default:
		return "unknown"
	}
}

// Outcome classifies how a request finished.
type Outcome int

const (
	// OutcomeSuccess means the backend responded usefully.
	OutcomeSuccess Outcome = iota
	// OutcomeFailure means the backend errored or was unreachable.
	OutcomeFailure
	// OutcomeTimeout means the backend accepted the request but was too slow.
	//
	// Tracked separately from failure because the two have different causes and
	// different thresholds: a server returning 500s fast is a different problem
	// from one that accepts work and stalls, and the latter is more dangerous
	// because it ties up connections and queue slots while failing.
	OutcomeTimeout
)

// BreakerOptions configures a Breaker.
type BreakerOptions struct {
	// Window is the rolling period over which rates are computed.
	Window time.Duration
	// Buckets divides the window; more buckets means smoother expiry.
	Buckets int
	// MinRequests is the minimum sample size before a rate can trip the
	// breaker. Without it, the first failed request would read as a 100% error
	// rate and open the circuit on a sample of one.
	MinRequests int
	// ErrorRate in (0,1] trips the breaker.
	ErrorRate float64
	// TimeoutRate in (0,1] trips the breaker.
	TimeoutRate float64
	// OpenDuration is how long to stay open before probing.
	OpenDuration time.Duration
	// HalfOpenRequests is how many probes may run concurrently while half open.
	HalfOpenRequests int
	// Now is injectable for tests.
	Now func() time.Time
}

func (o *BreakerOptions) applyDefaults() {
	if o.Window <= 0 {
		o.Window = 10 * time.Second
	}
	if o.Buckets < 1 {
		o.Buckets = 10
	}
	if o.MinRequests < 1 {
		o.MinRequests = 20
	}
	if o.ErrorRate <= 0 || o.ErrorRate > 1 {
		o.ErrorRate = 0.5
	}
	if o.TimeoutRate <= 0 || o.TimeoutRate > 1 {
		o.TimeoutRate = 0.5
	}
	if o.OpenDuration <= 0 {
		o.OpenDuration = 5 * time.Second
	}
	if o.HalfOpenRequests < 1 {
		o.HalfOpenRequests = 3
	}
	if o.Now == nil {
		o.Now = time.Now
	}
}

// Breaker is a per-backend circuit breaker over a rolling window.
//
// The point of a breaker is not to protect the backend — it is to protect the
// *caller*. Continuing to send traffic to a backend that is failing wastes
// connections and queue slots on work that will not succeed, and worse, makes
// every request pay the timeout before failing. Opening the circuit converts a
// slow failure into a fast one, which frees capacity for backends that work.
type Breaker struct {
	opts BreakerOptions

	mu       sync.Mutex
	state    State
	window   *rollingWindow
	openedAt time.Time
	// halfOpenInFlight counts probes currently running.
	halfOpenInFlight int
	// halfOpenSucceeded counts probes that have succeeded this cycle.
	halfOpenSucceeded int

	// transitions is incremented on every state change, for metrics.
	transitions uint64
}

// NewBreaker builds a Breaker.
func NewBreaker(opts BreakerOptions) *Breaker {
	opts.applyDefaults()
	return &Breaker{
		opts:   opts,
		state:  StateClosed,
		window: newRollingWindow(opts.Buckets, opts.Window/time.Duration(opts.Buckets), opts.Now),
	}
}

// Permits reports whether the breaker would admit a request, *without*
// consuming anything.
//
// This exists because eligibility is queried speculatively — the router asks
// every backend whether it could serve, then picks one — while admission is a
// state transition that consumes a half-open probe slot. Using the mutating
// call for both wedges the breaker permanently in half-open: filtering burns
// all the probe slots on backends that are never dispatched to, so no Record
// ever arrives to close the circuit, and Allow then refuses forever.
//
// Query and transition must therefore be separate operations.
func (b *Breaker) Permits() bool {
	b.mu.Lock()
	defer b.mu.Unlock()

	switch b.state {
	case StateClosed:
		return true
	case StateOpen:
		return b.opts.Now().Sub(b.openedAt) >= b.opts.OpenDuration
	case StateHalfOpen:
		return b.halfOpenInFlight < b.opts.HalfOpenRequests
	default:
		return true
	}
}

// Allow reserves permission to send a request, transitioning state as needed.
//
// When it returns true while half open the caller has consumed a probe slot and
// MUST report the result via Record, or that slot is leaked.
func (b *Breaker) Allow() bool {
	b.mu.Lock()
	defer b.mu.Unlock()

	now := b.opts.Now()

	switch b.state {
	case StateClosed:
		return true

	case StateOpen:
		if now.Sub(b.openedAt) < b.opts.OpenDuration {
			return false
		}
		// Cool-down elapsed: admit a limited number of probes.
		b.transition(StateHalfOpen)
		b.halfOpenInFlight = 1
		return true

	case StateHalfOpen:
		if b.halfOpenInFlight >= b.opts.HalfOpenRequests {
			return false
		}
		b.halfOpenInFlight++
		return true

	default:
		return true
	}
}

// Record reports how a request finished.
func (b *Breaker) Record(outcome Outcome) {
	b.mu.Lock()
	defer b.mu.Unlock()

	switch b.state {
	case StateHalfOpen:
		if b.halfOpenInFlight > 0 {
			b.halfOpenInFlight--
		}

		if outcome == OutcomeSuccess {
			b.halfOpenSucceeded++
			// Only close once enough probes have proven recovery. Closing on a
			// single success would flap: one lucky request would reopen the
			// floodgates onto a backend that is still mostly broken.
			if b.halfOpenSucceeded >= b.opts.HalfOpenRequests {
				b.transition(StateClosed)
				b.window.reset()
			}
			return
		}

		// Any failure during probing means the backend is still unwell.
		b.transition(StateOpen)
		b.openedAt = b.opts.Now()
		return

	case StateClosed:
		b.window.record(outcome)
		b.evaluate()

	case StateOpen:
		// Late results from requests issued before the breaker opened. They
		// describe the past and must not influence the cool-down.
	}
}

// evaluate trips the breaker if the rolling window exceeds a threshold.
// Caller must hold b.mu.
func (b *Breaker) evaluate() {
	total, failures, timeouts := b.window.totals()
	if total < b.opts.MinRequests {
		return
	}

	errorRate := float64(failures+timeouts) / float64(total)
	timeoutRate := float64(timeouts) / float64(total)

	if errorRate >= b.opts.ErrorRate || timeoutRate >= b.opts.TimeoutRate {
		b.transition(StateOpen)
		b.openedAt = b.opts.Now()
	}
}

// transition changes state and resets half-open bookkeeping.
// Caller must hold b.mu.
func (b *Breaker) transition(to State) {
	if b.state == to {
		return
	}
	b.state = to
	b.transitions++
	b.halfOpenInFlight = 0
	b.halfOpenSucceeded = 0
}

// State returns the current state, accounting for an elapsed cool-down.
func (b *Breaker) State() State {
	b.mu.Lock()
	defer b.mu.Unlock()

	if b.state == StateOpen && b.opts.Now().Sub(b.openedAt) >= b.opts.OpenDuration {
		// Report the state a caller would observe, without consuming a probe.
		return StateHalfOpen
	}
	return b.state
}

// Stats is a snapshot of breaker state.
type Stats struct {
	State       string  `json:"state"`
	Total       int     `json:"window_total"`
	Failures    int     `json:"window_failures"`
	Timeouts    int     `json:"window_timeouts"`
	ErrorRate   float64 `json:"error_rate"`
	TimeoutRate float64 `json:"timeout_rate"`
	Transitions uint64  `json:"transitions"`
}

// Stats returns a snapshot.
func (b *Breaker) Stats() Stats {
	b.mu.Lock()
	defer b.mu.Unlock()

	total, failures, timeouts := b.window.totals()
	s := Stats{
		State:       b.state.String(),
		Total:       total,
		Failures:    failures,
		Timeouts:    timeouts,
		Transitions: b.transitions,
	}
	if total > 0 {
		s.ErrorRate = float64(failures+timeouts) / float64(total)
		s.TimeoutRate = float64(timeouts) / float64(total)
	}
	return s
}

// Reset returns the breaker to closed with an empty window.
func (b *Breaker) Reset() {
	b.mu.Lock()
	defer b.mu.Unlock()

	b.transition(StateClosed)
	b.window.reset()
}

// rollingWindow counts outcomes in time buckets, expiring old ones.
//
// A simple cumulative counter cannot work: a backend that failed 1,000 times
// yesterday and has succeeded ever since would stay tripped forever. Bucketing
// bounds how far back the breaker can see.
type rollingWindow struct {
	buckets  []windowBucket
	duration time.Duration
	now      func() time.Time
	// head is the index of the bucket covering the current instant.
	head int
	// headStart is when the head bucket began.
	headStart time.Time
}

type windowBucket struct {
	success int
	failure int
	timeout int
}

func newRollingWindow(count int, bucketDuration time.Duration, now func() time.Time) *rollingWindow {
	if count < 1 {
		count = 1
	}
	if bucketDuration <= 0 {
		bucketDuration = time.Second
	}
	return &rollingWindow{
		buckets:   make([]windowBucket, count),
		duration:  bucketDuration,
		now:       now,
		headStart: now(),
	}
}

// advance rolls the window forward, clearing buckets that have expired.
func (w *rollingWindow) advance() {
	elapsed := w.now().Sub(w.headStart)
	if elapsed < w.duration {
		return
	}

	steps := int(elapsed / w.duration)
	if steps >= len(w.buckets) {
		// The whole window expired; clear everything.
		for i := range w.buckets {
			w.buckets[i] = windowBucket{}
		}
		w.head = 0
		w.headStart = w.now()
		return
	}

	for i := 0; i < steps; i++ {
		w.head = (w.head + 1) % len(w.buckets)
		w.buckets[w.head] = windowBucket{}
	}
	w.headStart = w.headStart.Add(time.Duration(steps) * w.duration)
}

func (w *rollingWindow) record(outcome Outcome) {
	w.advance()
	switch outcome {
	case OutcomeSuccess:
		w.buckets[w.head].success++
	case OutcomeFailure:
		w.buckets[w.head].failure++
	case OutcomeTimeout:
		w.buckets[w.head].timeout++
	}
}

func (w *rollingWindow) totals() (total, failures, timeouts int) {
	w.advance()
	for _, b := range w.buckets {
		total += b.success + b.failure + b.timeout
		failures += b.failure
		timeouts += b.timeout
	}
	return total, failures, timeouts
}

func (w *rollingWindow) reset() {
	for i := range w.buckets {
		w.buckets[i] = windowBucket{}
	}
	w.head = 0
	w.headStart = w.now()
}

// String implements fmt.Stringer for debugging.
func (b *Breaker) String() string {
	s := b.Stats()
	return fmt.Sprintf("breaker{state=%s total=%d errRate=%.2f}", s.State, s.Total, s.ErrorRate)
}
