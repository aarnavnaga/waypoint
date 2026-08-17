// Package admission implements bounded admission control.
//
// The gateway must decide, before dispatching, whether it can serve a request
// at all. Without this an overload turns into unbounded queue growth: every
// request is accepted, every request gets slower, and clients time out *after*
// paying full backend cost. Shedding early with 503 keeps the work the system
// does useful.
package admission

import (
	"container/heap"
	"context"
	"errors"
	"sync"
	"sync/atomic"
	"time"
)

// ErrQueueFull is returned when the queue is at capacity and the request must
// be shed.
var ErrQueueFull = errors.New("admission: queue full")

// Class is a coarse priority band.
type Class int

const (
	// ClassInteractive is latency-sensitive traffic (the default).
	ClassInteractive Class = 0
	// ClassBatch is throughput-oriented traffic that may wait longer.
	ClassBatch Class = 1
)

// Ticket is a granted admission. Release must be called exactly once.
type Ticket struct {
	release func()
}

// Release returns the slot to the controller.
func (t *Ticket) Release() {
	if t != nil && t.release != nil {
		t.release()
	}
}

// Options configures a Controller.
type Options struct {
	// MaxConcurrent bounds simultaneously dispatched requests. 0 means
	// unbounded, which disables queueing entirely.
	MaxConcurrent int
	// MaxQueue bounds waiting requests. Beyond it, callers get ErrQueueFull.
	MaxQueue int
	// AgingInterval is how much waiting time counts as one priority level.
	// After waiting this long, a batch request competes with interactive ones.
	// Zero disables aging (strict priority).
	AgingInterval time.Duration
}

// Controller is a bounded, aging priority queue in front of dispatch.
//
// Ordering is by effective priority, then by arrival time. Effective priority
// decays with waiting time (see AgingInterval), so a flood of interactive
// requests cannot starve batch traffic indefinitely — strict priority queues
// are a classic source of indefinite postponement under sustained load.
type Controller struct {
	opts Options

	mu       sync.Mutex
	active   int
	waiters  waiterHeap
	seq      uint64
	shutdown bool

	queued   atomic.Int64
	admitted atomic.Uint64
	shed     atomic.Uint64

	now func() time.Time
}

// waiter is one queued request.
type waiter struct {
	class    Class
	enqueued time.Time
	// deadline encodes aging statically: enqueued + class*AgingInterval.
	// Ordering by it is equivalent to "priority that improves while waiting",
	// but unlike a priority recomputed at comparison time it never changes, so
	// the heap invariant stays valid without re-heapifying on every pop.
	deadline time.Time
	seq      uint64
	ready    chan struct{}
	// index is maintained by container/heap.
	index int
	// cancelled marks a waiter abandoned by its caller.
	cancelled bool
}

// New returns a Controller.
func New(opts Options) *Controller {
	if opts.MaxQueue < 0 {
		opts.MaxQueue = 0
	}
	return &Controller{
		opts:    opts,
		now:     time.Now,
		waiters: waiterHeap{aging: opts.AgingInterval},
	}
}

// Acquire blocks until a slot is free, the context is done, or the queue is
// full. The returned Ticket must be released.
func (c *Controller) Acquire(ctx context.Context, class Class) (*Ticket, error) {
	if c.opts.MaxConcurrent <= 0 {
		c.admitted.Add(1)
		return &Ticket{release: func() {}}, nil
	}

	c.mu.Lock()
	if c.shutdown {
		c.mu.Unlock()
		return nil, errors.New("admission: shutting down")
	}

	// Fast path: capacity is free and nobody is waiting ahead of us.
	if c.active < c.opts.MaxConcurrent && c.waiters.Len() == 0 {
		c.active++
		c.mu.Unlock()
		c.admitted.Add(1)
		return c.newTicket(), nil
	}

	if c.opts.MaxQueue > 0 && c.waiters.Len() >= c.opts.MaxQueue {
		c.mu.Unlock()
		c.shed.Add(1)
		return nil, ErrQueueFull
	}

	c.seq++
	enqueued := c.now()
	w := &waiter{
		class:    class,
		enqueued: enqueued,
		deadline: enqueued.Add(time.Duration(class) * c.opts.AgingInterval),
		seq:      c.seq,
		ready:    make(chan struct{}),
	}
	heap.Push(&c.waiters, w)
	c.mu.Unlock()
	c.queued.Add(1)

	select {
	case <-w.ready:
		c.queued.Add(-1)
		c.admitted.Add(1)
		return c.newTicket(), nil

	case <-ctx.Done():
		c.queued.Add(-1)
		c.mu.Lock()
		if w.index >= 0 {
			// Still queued: remove it.
			heap.Remove(&c.waiters, w.index)
			w.cancelled = true
			c.mu.Unlock()
			return nil, ctx.Err()
		}
		// Raced with a grant: we hold a slot we are not going to use.
		c.mu.Unlock()
		c.releaseSlot()
		return nil, ctx.Err()
	}
}

func (c *Controller) newTicket() *Ticket {
	var once sync.Once
	return &Ticket{release: func() { once.Do(c.releaseSlot) }}
}

// releaseSlot returns capacity and hands it to the highest-priority waiter.
func (c *Controller) releaseSlot() {
	c.mu.Lock()
	defer c.mu.Unlock()

	for c.waiters.Len() > 0 {
		w, ok := heap.Pop(&c.waiters).(*waiter)
		if !ok || w.cancelled {
			continue
		}
		// Hand our slot directly to the waiter; active stays unchanged.
		close(w.ready)
		return
	}
	c.active--
}

// Stats is a snapshot of admission behavior.
type Stats struct {
	Active        int    `json:"active"`
	Queued        int64  `json:"queued"`
	MaxConcurrent int    `json:"max_concurrent"`
	MaxQueue      int    `json:"max_queue"`
	Admitted      uint64 `json:"admitted_total"`
	Shed          uint64 `json:"shed_total"`
}

// Stats returns current admission counters.
func (c *Controller) Stats() Stats {
	c.mu.Lock()
	active := c.active
	c.mu.Unlock()

	return Stats{
		Active:        active,
		Queued:        c.queued.Load(),
		MaxConcurrent: c.opts.MaxConcurrent,
		MaxQueue:      c.opts.MaxQueue,
		Admitted:      c.admitted.Load(),
		Shed:          c.shed.Load(),
	}
}

// QueueDepth reports how many requests are waiting.
func (c *Controller) QueueDepth() int64 { return c.queued.Load() }

// Shutdown refuses new admissions and wakes every waiter.
func (c *Controller) Shutdown() {
	c.mu.Lock()
	defer c.mu.Unlock()

	c.shutdown = true
	for c.waiters.Len() > 0 {
		if w, ok := heap.Pop(&c.waiters).(*waiter); ok && !w.cancelled {
			close(w.ready)
		}
	}
}

// waiterHeap orders waiters by priority, then arrival.
type waiterHeap struct {
	items []*waiter
	// aging is how much wait time offsets one priority class. Zero selects
	// strict priority ordering instead of earliest-deadline-first.
	aging time.Duration
}

func (h waiterHeap) Len() int { return len(h.items) }

// Less implements two orderings:
//
//   - aging disabled: strict priority (class, then arrival). Simple, but a
//     sustained flood of high-priority work starves low-priority work forever.
//   - aging enabled: earliest deadline first, where deadline is
//     enqueued + class*aging. A batch request that has waited longer than one
//     aging interval overtakes a freshly arrived interactive one, which bounds
//     worst-case waiting instead of leaving it unbounded.
func (h waiterHeap) Less(i, j int) bool {
	a, b := h.items[i], h.items[j]

	if h.aging <= 0 {
		if a.class != b.class {
			return a.class < b.class
		}
		return a.seq < b.seq
	}

	if !a.deadline.Equal(b.deadline) {
		return a.deadline.Before(b.deadline)
	}
	return a.seq < b.seq
}

func (h waiterHeap) Swap(i, j int) {
	h.items[i], h.items[j] = h.items[j], h.items[i]
	h.items[i].index = i
	h.items[j].index = j
}

func (h *waiterHeap) Push(x any) {
	w, ok := x.(*waiter)
	if !ok {
		return
	}
	w.index = len(h.items)
	h.items = append(h.items, w)
}

func (h *waiterHeap) Pop() any {
	old := h.items
	n := len(old)
	w := old[n-1]
	old[n-1] = nil
	w.index = -1
	h.items = old[:n-1]
	return w
}
