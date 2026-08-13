package mocksim

import (
	"context"
	"errors"
	"math"
	"sync/atomic"
	"time"
)

// ErrQueueFull is returned when the admission queue is saturated.
var ErrQueueFull = errors.New("mocksim: queue full")

// Scheduler simulates continuous batching.
//
// A real engine runs a decode step over every sequence in the batch at once, so
// adding sequences raises aggregate throughput while *slowing each individual
// sequence*: the per-step cost grows with batch size (more attention work, more
// memory bandwidth). The scheduler reproduces that shape with a configurable
// curve, which is what makes "spread load evenly" a non-trivial decision rather
// than an obvious win.
type Scheduler struct {
	slots    chan struct{}
	maxBatch int
	maxQueue int

	active atomic.Int64
	queued atomic.Int64

	admitted atomic.Uint64
	rejected atomic.Uint64

	baseDecode time.Duration
	alpha      float64
	beta       float64
}

// SchedulerOptions configures batching behavior.
type SchedulerOptions struct {
	// MaxBatch is how many sequences decode concurrently.
	MaxBatch int
	// MaxQueue bounds sequences waiting for a slot; 0 means unbounded waiting.
	MaxQueue int
	// BaseDecode is the per-token decode cost at batch size 1.
	BaseDecode time.Duration
	// Alpha scales how much batch pressure slows each sequence.
	Alpha float64
	// Beta is the exponent on batch pressure (1.0 = linear).
	Beta float64
}

// NewScheduler builds a Scheduler from opts, applying defaults for zero values.
func NewScheduler(opts SchedulerOptions) *Scheduler {
	if opts.MaxBatch < 1 {
		opts.MaxBatch = 1
	}
	if opts.Beta <= 0 {
		opts.Beta = 1.0
	}
	if opts.Alpha < 0 {
		opts.Alpha = 0
	}

	s := &Scheduler{
		slots:      make(chan struct{}, opts.MaxBatch),
		maxBatch:   opts.MaxBatch,
		maxQueue:   opts.MaxQueue,
		baseDecode: opts.BaseDecode,
		alpha:      opts.Alpha,
		beta:       opts.Beta,
	}
	for i := 0; i < opts.MaxBatch; i++ {
		s.slots <- struct{}{}
	}
	return s
}

// Acquire reserves a batch slot, blocking until one is free, the context is
// cancelled, or the queue is full. The returned release function must be called
// exactly once.
func (s *Scheduler) Acquire(ctx context.Context) (release func(), err error) {
	// Fast path: a slot is immediately available, so nothing queues.
	select {
	case <-s.slots:
		s.active.Add(1)
		s.admitted.Add(1)
		return s.releaseOnce(), nil
	default:
	}

	if s.maxQueue > 0 && int(s.queued.Load()) >= s.maxQueue {
		s.rejected.Add(1)
		return nil, ErrQueueFull
	}

	s.queued.Add(1)
	defer s.queued.Add(-1)

	select {
	case <-s.slots:
		s.active.Add(1)
		s.admitted.Add(1)
		return s.releaseOnce(), nil
	case <-ctx.Done():
		return nil, ctx.Err()
	}
}

// releaseOnce returns a release function guarded against double-release, which
// would corrupt the slot count and let the batch exceed MaxBatch.
func (s *Scheduler) releaseOnce() func() {
	var done atomic.Bool
	return func() {
		if !done.CompareAndSwap(false, true) {
			return
		}
		s.active.Add(-1)
		s.slots <- struct{}{}
	}
}

// DecodeInterval returns the current per-token decode cost, which depends on how
// many sequences are decoding right now:
//
//	interval = base * (1 + alpha * (batch-1)^beta)
func (s *Scheduler) DecodeInterval() time.Duration {
	batch := s.active.Load()
	if batch < 1 {
		batch = 1
	}
	pressure := math.Pow(float64(batch-1), s.beta)
	return time.Duration(float64(s.baseDecode) * (1 + s.alpha*pressure))
}

// BatchSize reports how many sequences are decoding.
func (s *Scheduler) BatchSize() int { return int(s.active.Load()) }

// QueueDepth reports how many sequences are waiting for a slot.
func (s *Scheduler) QueueDepth() int { return int(s.queued.Load()) }

// MaxBatch reports the configured batch ceiling.
func (s *Scheduler) MaxBatch() int { return s.maxBatch }

// SchedulerStats is a point-in-time view of batching behavior.
type SchedulerStats struct {
	BatchSize  int    `json:"batch_size"`
	MaxBatch   int    `json:"max_batch"`
	QueueDepth int    `json:"queue_depth"`
	Admitted   uint64 `json:"admitted"`
	Rejected   uint64 `json:"rejected"`
	// DecodeIntervalMS is the per-token cost at the current batch size.
	DecodeIntervalMS float64 `json:"decode_interval_ms"`
}

// Stats returns a snapshot of scheduler counters.
func (s *Scheduler) Stats() SchedulerStats {
	return SchedulerStats{
		BatchSize:        s.BatchSize(),
		MaxBatch:         s.maxBatch,
		QueueDepth:       s.QueueDepth(),
		Admitted:         s.admitted.Load(),
		Rejected:         s.rejected.Load(),
		DecodeIntervalMS: float64(s.DecodeInterval()) / float64(time.Millisecond),
	}
}

// sleepCtx waits for d or until ctx is done, whichever comes first.
// It reports whether the full duration elapsed.
func sleepCtx(ctx context.Context, d time.Duration) bool {
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
