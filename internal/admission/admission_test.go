package admission

import (
	"context"
	"errors"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

func TestUnboundedWhenMaxConcurrentZero(t *testing.T) {
	c := New(Options{MaxConcurrent: 0})

	for i := 0; i < 100; i++ {
		ticket, err := c.Acquire(context.Background(), ClassInteractive)
		if err != nil {
			t.Fatalf("Acquire %d: %v", i, err)
		}
		ticket.Release()
	}
}

func TestAcquireBoundsConcurrency(t *testing.T) {
	c := New(Options{MaxConcurrent: 2, MaxQueue: 10})
	ctx := context.Background()

	t1, err := c.Acquire(ctx, ClassInteractive)
	if err != nil {
		t.Fatalf("Acquire: %v", err)
	}
	t2, err := c.Acquire(ctx, ClassInteractive)
	if err != nil {
		t.Fatalf("Acquire: %v", err)
	}

	if got := c.Stats().Active; got != 2 {
		t.Errorf("Active = %d, want 2", got)
	}

	// Third must queue.
	done := make(chan struct{})
	go func() {
		t3, err := c.Acquire(ctx, ClassInteractive)
		if err == nil {
			t3.Release()
		}
		close(done)
	}()

	select {
	case <-done:
		t.Fatal("third Acquire succeeded while at capacity")
	case <-time.After(50 * time.Millisecond):
	}

	t1.Release()
	select {
	case <-done:
	case <-time.After(time.Second):
		t.Fatal("queued request was not admitted after a release")
	}
	t2.Release()
}

func TestQueueFullSheds(t *testing.T) {
	c := New(Options{MaxConcurrent: 1, MaxQueue: 1})
	ctx := context.Background()

	held, err := c.Acquire(ctx, ClassInteractive)
	if err != nil {
		t.Fatalf("Acquire: %v", err)
	}
	defer held.Release()

	// Fill the single queue slot.
	go func() {
		if ticket, err := c.Acquire(ctx, ClassInteractive); err == nil {
			ticket.Release()
		}
	}()

	waitFor(t, func() bool { return c.QueueDepth() == 1 }, "waiter to enqueue")

	if _, err := c.Acquire(ctx, ClassInteractive); !errors.Is(err, ErrQueueFull) {
		t.Errorf("Acquire error = %v, want ErrQueueFull", err)
	}
	if got := c.Stats().Shed; got != 1 {
		t.Errorf("Shed = %d, want 1", got)
	}
}

func TestAcquireRespectsContext(t *testing.T) {
	c := New(Options{MaxConcurrent: 1, MaxQueue: 10})

	held, err := c.Acquire(context.Background(), ClassInteractive)
	if err != nil {
		t.Fatalf("Acquire: %v", err)
	}
	defer held.Release()

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Millisecond)
	defer cancel()

	if _, err := c.Acquire(ctx, ClassInteractive); !errors.Is(err, context.DeadlineExceeded) {
		t.Errorf("Acquire error = %v, want DeadlineExceeded", err)
	}
	// The abandoned waiter must not stay in the queue.
	waitFor(t, func() bool { return c.QueueDepth() == 0 }, "queue to drain after cancellation")
}

// A cancelled waiter must not consume the slot handed to it, or capacity leaks
// one slot per cancellation until the gateway stops dispatching entirely.
func TestCancelledWaiterDoesNotLeakSlot(t *testing.T) {
	c := New(Options{MaxConcurrent: 1, MaxQueue: 10})

	held, err := c.Acquire(context.Background(), ClassInteractive)
	if err != nil {
		t.Fatalf("Acquire: %v", err)
	}

	ctx, cancel := context.WithCancel(context.Background())
	waiterDone := make(chan struct{})
	go func() {
		defer close(waiterDone)
		if ticket, err := c.Acquire(ctx, ClassInteractive); err == nil {
			ticket.Release()
		}
	}()

	waitFor(t, func() bool { return c.QueueDepth() == 1 }, "waiter to enqueue")
	cancel()
	<-waiterDone
	held.Release()

	// Capacity must be fully recovered.
	waitFor(t, func() bool { return c.Stats().Active == 0 }, "active count to return to zero")

	ticket, err := c.Acquire(context.Background(), ClassInteractive)
	if err != nil {
		t.Fatalf("Acquire after cancellation: %v", err)
	}
	ticket.Release()
}

func TestReleaseIsIdempotent(t *testing.T) {
	c := New(Options{MaxConcurrent: 1, MaxQueue: 1})

	ticket, err := c.Acquire(context.Background(), ClassInteractive)
	if err != nil {
		t.Fatalf("Acquire: %v", err)
	}
	ticket.Release()
	ticket.Release()
	ticket.Release()

	if got := c.Stats().Active; got != 0 {
		t.Errorf("Active = %d after repeated release, want 0", got)
	}
}

func TestNilTicketReleaseIsSafe(t *testing.T) {
	var ticket *Ticket
	ticket.Release() // must not panic
}

// Strict priority: with aging disabled, interactive work goes first.
func TestInteractiveBeatsBatch(t *testing.T) {
	c := New(Options{MaxConcurrent: 1, MaxQueue: 10})

	held, err := c.Acquire(context.Background(), ClassInteractive)
	if err != nil {
		t.Fatalf("Acquire: %v", err)
	}

	var order []string
	var mu sync.Mutex
	var wg sync.WaitGroup

	// Enqueue batch first so FIFO alone would serve it first.
	wg.Add(1)
	go func() {
		defer wg.Done()
		ticket, err := c.Acquire(context.Background(), ClassBatch)
		if err != nil {
			return
		}
		mu.Lock()
		order = append(order, "batch")
		mu.Unlock()
		ticket.Release()
	}()
	waitFor(t, func() bool { return c.QueueDepth() == 1 }, "batch waiter to enqueue")

	wg.Add(1)
	go func() {
		defer wg.Done()
		ticket, err := c.Acquire(context.Background(), ClassInteractive)
		if err != nil {
			return
		}
		mu.Lock()
		order = append(order, "interactive")
		mu.Unlock()
		ticket.Release()
	}()
	waitFor(t, func() bool { return c.QueueDepth() == 2 }, "interactive waiter to enqueue")

	held.Release()
	wg.Wait()

	mu.Lock()
	defer mu.Unlock()
	if len(order) != 2 || order[0] != "interactive" {
		t.Errorf("service order = %v, want interactive first", order)
	}
}

// Aging is what stops a sustained interactive flood from starving batch work.
func TestAgingPromotesStarvedBatchRequest(t *testing.T) {
	c := New(Options{MaxConcurrent: 1, MaxQueue: 10, AgingInterval: time.Second})

	// Freeze the clock so the test controls apparent waiting time.
	base := time.Now()
	current := base
	var clockMu sync.Mutex
	c.now = func() time.Time {
		clockMu.Lock()
		defer clockMu.Unlock()
		return current
	}

	held, err := c.Acquire(context.Background(), ClassInteractive)
	if err != nil {
		t.Fatalf("Acquire: %v", err)
	}

	var order []string
	var mu sync.Mutex
	var wg sync.WaitGroup

	// Batch request enqueues now...
	wg.Add(1)
	go func() {
		defer wg.Done()
		ticket, err := c.Acquire(context.Background(), ClassBatch)
		if err != nil {
			return
		}
		mu.Lock()
		order = append(order, "batch")
		mu.Unlock()
		ticket.Release()
	}()
	waitFor(t, func() bool { return c.QueueDepth() == 1 }, "batch waiter to enqueue")

	// ...then 5 seconds pass, which at a 1s aging interval outweighs the one
	// class level separating batch from interactive.
	clockMu.Lock()
	current = base.Add(5 * time.Second)
	clockMu.Unlock()

	wg.Add(1)
	go func() {
		defer wg.Done()
		ticket, err := c.Acquire(context.Background(), ClassInteractive)
		if err != nil {
			return
		}
		mu.Lock()
		order = append(order, "interactive")
		mu.Unlock()
		ticket.Release()
	}()
	waitFor(t, func() bool { return c.QueueDepth() == 2 }, "interactive waiter to enqueue")

	held.Release()
	wg.Wait()

	mu.Lock()
	defer mu.Unlock()
	if len(order) != 2 || order[0] != "batch" {
		t.Errorf("service order = %v, want the aged batch request first", order)
	}
}

func TestFIFOWithinSameClass(t *testing.T) {
	c := New(Options{MaxConcurrent: 1, MaxQueue: 10})

	held, err := c.Acquire(context.Background(), ClassInteractive)
	if err != nil {
		t.Fatalf("Acquire: %v", err)
	}

	var order []int
	var mu sync.Mutex
	var wg sync.WaitGroup

	for i := 0; i < 4; i++ {
		i := i
		wg.Add(1)
		go func() {
			defer wg.Done()
			ticket, err := c.Acquire(context.Background(), ClassInteractive)
			if err != nil {
				return
			}
			mu.Lock()
			order = append(order, i)
			mu.Unlock()
			ticket.Release()
		}()
		waitFor(t, func() bool { return c.QueueDepth() == int64(i+1) }, "waiter to enqueue")
	}

	held.Release()
	wg.Wait()

	mu.Lock()
	defer mu.Unlock()
	for i, got := range order {
		if got != i {
			t.Errorf("service order = %v, want ascending (FIFO within class)", order)
			break
		}
	}
}

func TestShutdownWakesWaiters(t *testing.T) {
	c := New(Options{MaxConcurrent: 1, MaxQueue: 10})

	held, err := c.Acquire(context.Background(), ClassInteractive)
	if err != nil {
		t.Fatalf("Acquire: %v", err)
	}
	defer held.Release()

	done := make(chan struct{})
	go func() {
		if ticket, err := c.Acquire(context.Background(), ClassInteractive); err == nil {
			ticket.Release()
		}
		close(done)
	}()
	waitFor(t, func() bool { return c.QueueDepth() == 1 }, "waiter to enqueue")

	c.Shutdown()

	select {
	case <-done:
	case <-time.After(time.Second):
		t.Fatal("Shutdown did not wake the queued waiter")
	}

	if _, err := c.Acquire(context.Background(), ClassInteractive); err == nil {
		t.Error("Acquire succeeded after Shutdown")
	}
}

func TestConcurrentAcquireNeverExceedsLimit(t *testing.T) {
	const limit = 4
	c := New(Options{MaxConcurrent: limit, MaxQueue: 1000})

	var current, peak atomic.Int64
	var wg sync.WaitGroup

	for i := 0; i < 200; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			ticket, err := c.Acquire(context.Background(), ClassInteractive)
			if err != nil {
				return
			}
			n := current.Add(1)
			for {
				old := peak.Load()
				if n <= old || peak.CompareAndSwap(old, n) {
					break
				}
			}
			time.Sleep(time.Millisecond)
			current.Add(-1)
			ticket.Release()
		}()
	}
	wg.Wait()

	if got := peak.Load(); got > limit {
		t.Errorf("peak concurrency = %d, exceeds limit %d", got, limit)
	}
	if got := c.Stats().Active; got != 0 {
		t.Errorf("Active = %d after drain, want 0", got)
	}
	if got := c.QueueDepth(); got != 0 {
		t.Errorf("QueueDepth = %d after drain, want 0", got)
	}
}

func waitFor(t *testing.T, cond func() bool, what string) {
	t.Helper()
	deadline := time.Now().Add(2 * time.Second)
	for time.Now().Before(deadline) {
		if cond() {
			return
		}
		time.Sleep(time.Millisecond)
	}
	t.Fatalf("timed out waiting for %s", what)
}
