package resilience

import (
	"context"
	"math"
	"math/rand"
	"sync"
	"time"
)

// RetryBudget bounds retries as a fraction of total requests.
//
// This is the defense against retry storms, and it is the part people usually
// leave out. Capped attempts alone are not enough: if a backend starts failing
// everything, "3 attempts" means the fleet instantly emits 3x its normal
// request rate at a system that is already struggling, turning a partial outage
// into a total one. A budget makes retries a scarce resource — when failures are
// widespread the budget empties and retries stop, but when failures are rare
// (the case retries are actually *for*) there is always budget available.
//
// Implementation follows the token-bucket approach used by gRPC and Finagle:
// each request deposits `ratio` tokens, each retry withdraws one whole token.
// Tokens decay so the budget reflects recent traffic rather than all history.
type RetryBudget struct {
	mu     sync.Mutex
	tokens float64
	// ratio is retries permitted per request (0.1 = 10%).
	ratio float64
	// maxTokens caps accumulation so a long quiet period cannot bank enough
	// budget to fund an unbounded burst later.
	maxTokens float64
	// ttl is how long deposited tokens survive.
	ttl      time.Duration
	lastDrip time.Time
	now      func() time.Time

	granted uint64
	denied  uint64
}

// RetryBudgetOptions configures a RetryBudget.
type RetryBudgetOptions struct {
	// Percent is the share of requests that may be retried (0-100).
	Percent float64
	// MinRetriesPerSecond keeps a small floor of budget so that a low-traffic
	// service can still retry at all.
	MinRetriesPerSecond float64
	// TTL is how long tokens survive before decaying.
	TTL time.Duration
	// Now is injectable for tests.
	Now func() time.Time
}

// NewRetryBudget builds a RetryBudget.
func NewRetryBudget(opts RetryBudgetOptions) *RetryBudget {
	if opts.Percent < 0 {
		opts.Percent = 0
	}
	if opts.TTL <= 0 {
		opts.TTL = 10 * time.Second
	}
	if opts.Now == nil {
		opts.Now = time.Now
	}
	if opts.MinRetriesPerSecond < 0 {
		opts.MinRetriesPerSecond = 0
	}

	ratio := opts.Percent / 100
	// Cap accumulation at roughly one TTL of deposits at a moderate rate, plus
	// the configured floor.
	maxTokens := math.Max(opts.MinRetriesPerSecond*opts.TTL.Seconds(), 10)

	return &RetryBudget{
		tokens:    opts.MinRetriesPerSecond * opts.TTL.Seconds(),
		ratio:     ratio,
		maxTokens: maxTokens,
		ttl:       opts.TTL,
		lastDrip:  opts.Now(),
		now:       opts.Now,
	}
}

// Deposit records that a request was issued, adding retry budget.
func (b *RetryBudget) Deposit() {
	b.mu.Lock()
	defer b.mu.Unlock()

	b.decay()
	b.tokens += b.ratio
	if b.tokens > b.maxTokens {
		b.tokens = b.maxTokens
	}
}

// Withdraw attempts to spend one retry. It reports whether the retry is allowed.
func (b *RetryBudget) Withdraw() bool {
	b.mu.Lock()
	defer b.mu.Unlock()

	b.decay()
	if b.tokens < 1 {
		b.denied++
		return false
	}
	b.tokens--
	b.granted++
	return true
}

// decay expires tokens linearly over the TTL. Caller must hold b.mu.
func (b *RetryBudget) decay() {
	now := b.now()
	elapsed := now.Sub(b.lastDrip)
	if elapsed <= 0 {
		return
	}
	b.lastDrip = now

	if b.ttl <= 0 {
		return
	}
	// Lose a full budget's worth of tokens per TTL.
	loss := b.maxTokens * (elapsed.Seconds() / b.ttl.Seconds())
	b.tokens -= loss
	if b.tokens < 0 {
		b.tokens = 0
	}
}

// BudgetStats is a snapshot of budget usage.
type BudgetStats struct {
	Tokens  float64 `json:"tokens"`
	Granted uint64  `json:"granted_total"`
	Denied  uint64  `json:"denied_total"`
}

// Stats returns a snapshot.
func (b *RetryBudget) Stats() BudgetStats {
	b.mu.Lock()
	defer b.mu.Unlock()
	return BudgetStats{Tokens: b.tokens, Granted: b.granted, Denied: b.denied}
}

// Backoff computes retry delays with exponential growth and full jitter.
type Backoff struct {
	Base time.Duration
	Max  time.Duration
	// rand is injectable for deterministic tests.
	rand func() float64
}

// NewBackoff builds a Backoff with full jitter.
func NewBackoff(base, maxDelay time.Duration) *Backoff {
	if base <= 0 {
		base = 20 * time.Millisecond
	}
	if maxDelay < base {
		maxDelay = base
	}
	//nolint:gosec // jitter does not need cryptographic randomness
	r := rand.New(rand.NewSource(time.Now().UnixNano()))

	var mu sync.Mutex
	return &Backoff{
		Base: base,
		Max:  maxDelay,
		rand: func() float64 {
			mu.Lock()
			defer mu.Unlock()
			return r.Float64()
		},
	}
}

// Delay returns the wait before the given attempt (0-indexed).
//
// Full jitter — a uniform draw from [0, exponential cap) rather than the cap
// itself — is what prevents a thundering herd. If every client that failed at
// the same instant waited exactly the same backoff, they would all retry at the
// same instant too, reproducing the burst that caused the failure. Spreading
// retries uniformly across the interval decorrelates them.
func (b *Backoff) Delay(attempt int) time.Duration {
	if attempt < 0 {
		attempt = 0
	}

	// Guard against overflow for large attempt counts.
	capped := float64(b.Base) * math.Pow(2, float64(attempt))
	if capped > float64(b.Max) || math.IsInf(capped, 0) {
		capped = float64(b.Max)
	}

	return time.Duration(b.rand() * capped)
}

// Wait sleeps for the backoff delay or until ctx is done.
// It reports whether the wait completed.
func (b *Backoff) Wait(ctx context.Context, attempt int) bool {
	d := b.Delay(attempt)
	if d <= 0 {
		return ctx.Err() == nil
	}

	timer := time.NewTimer(d)
	defer timer.Stop()

	select {
	case <-timer.C:
		return true
	case <-ctx.Done():
		return false
	}
}
