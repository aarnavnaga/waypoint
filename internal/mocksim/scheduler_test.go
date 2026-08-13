package mocksim

import (
	"context"
	"errors"
	"sync"
	"testing"
	"time"
)

func TestSchedulerCapsBatchSize(t *testing.T) {
	s := NewScheduler(SchedulerOptions{MaxBatch: 3})
	ctx := context.Background()

	releases := make([]func(), 0, 3)
	for i := 0; i < 3; i++ {
		rel, err := s.Acquire(ctx)
		if err != nil {
			t.Fatalf("Acquire %d: %v", i, err)
		}
		releases = append(releases, rel)
	}

	if got := s.BatchSize(); got != 3 {
		t.Errorf("BatchSize = %d, want 3", got)
	}

	// A fourth acquire must block until a slot frees.
	acquired := make(chan struct{})
	go func() {
		rel, err := s.Acquire(ctx)
		if err == nil {
			rel()
		}
		close(acquired)
	}()

	select {
	case <-acquired:
		t.Fatal("fourth Acquire succeeded while batch was full")
	case <-time.After(50 * time.Millisecond):
	}

	releases[0]()
	select {
	case <-acquired:
	case <-time.After(time.Second):
		t.Fatal("Acquire did not proceed after a slot was released")
	}

	for _, rel := range releases[1:] {
		rel()
	}
}

func TestSchedulerQueueFull(t *testing.T) {
	s := NewScheduler(SchedulerOptions{MaxBatch: 1, MaxQueue: 1})
	ctx := context.Background()

	held, err := s.Acquire(ctx)
	if err != nil {
		t.Fatalf("first Acquire: %v", err)
	}
	defer held()

	// One waiter is allowed to queue.
	waiting := make(chan struct{})
	go func() {
		rel, err := s.Acquire(ctx)
		if err == nil {
			rel()
		}
		close(waiting)
	}()

	// Give the waiter time to register in the queue.
	deadline := time.After(time.Second)
	for s.QueueDepth() == 0 {
		select {
		case <-deadline:
			t.Fatal("waiter never entered the queue")
		default:
			time.Sleep(time.Millisecond)
		}
	}

	// The next one must be shed rather than queued.
	if _, err := s.Acquire(ctx); !errors.Is(err, ErrQueueFull) {
		t.Errorf("Acquire error = %v, want ErrQueueFull", err)
	}
	if s.Stats().Rejected != 1 {
		t.Errorf("Rejected = %d, want 1", s.Stats().Rejected)
	}

	held()
	<-waiting
}

func TestSchedulerRespectsContextCancellation(t *testing.T) {
	s := NewScheduler(SchedulerOptions{MaxBatch: 1})

	held, err := s.Acquire(context.Background())
	if err != nil {
		t.Fatalf("first Acquire: %v", err)
	}
	defer held()

	ctx, cancel := context.WithTimeout(context.Background(), 20*time.Millisecond)
	defer cancel()

	if _, err := s.Acquire(ctx); !errors.Is(err, context.DeadlineExceeded) {
		t.Errorf("Acquire error = %v, want DeadlineExceeded", err)
	}
}

func TestSchedulerDoubleReleaseIsSafe(t *testing.T) {
	s := NewScheduler(SchedulerOptions{MaxBatch: 2})

	rel, err := s.Acquire(context.Background())
	if err != nil {
		t.Fatalf("Acquire: %v", err)
	}
	rel()
	rel() // must be a no-op, not a slot leak

	if got := s.BatchSize(); got != 0 {
		t.Errorf("BatchSize after double release = %d, want 0", got)
	}
	// If the slot leaked, this would exceed MaxBatch.
	r1, _ := s.Acquire(context.Background())
	r2, _ := s.Acquire(context.Background())
	if got := s.BatchSize(); got != 2 {
		t.Errorf("BatchSize = %d, want 2", got)
	}
	r1()
	r2()
}

func TestDecodeIntervalGrowsWithBatch(t *testing.T) {
	s := NewScheduler(SchedulerOptions{
		MaxBatch:   8,
		BaseDecode: 10 * time.Millisecond,
		Alpha:      0.1,
		Beta:       1.0,
	})

	if got := s.DecodeInterval(); got != 10*time.Millisecond {
		t.Errorf("idle DecodeInterval = %s, want 10ms", got)
	}

	ctx := context.Background()
	rels := make([]func(), 0, 5)
	for i := 0; i < 5; i++ {
		rel, err := s.Acquire(ctx)
		if err != nil {
			t.Fatalf("Acquire: %v", err)
		}
		rels = append(rels, rel)
	}
	defer func() {
		for _, r := range rels {
			r()
		}
	}()

	// batch=5 → 10ms * (1 + 0.1*4) = 14ms
	want := 14 * time.Millisecond
	if got := s.DecodeInterval(); got != want {
		t.Errorf("DecodeInterval at batch 5 = %s, want %s", got, want)
	}
}

func TestDecodeIntervalSuperlinearCurve(t *testing.T) {
	s := NewScheduler(SchedulerOptions{
		MaxBatch:   16,
		BaseDecode: 10 * time.Millisecond,
		Alpha:      0.05,
		Beta:       2.0,
	})

	ctx := context.Background()
	rels := make([]func(), 0, 5)
	for i := 0; i < 5; i++ {
		rel, _ := s.Acquire(ctx)
		rels = append(rels, rel)
	}
	defer func() {
		for _, r := range rels {
			r()
		}
	}()

	// batch=5 → 10ms * (1 + 0.05*4^2) = 10ms * 1.8 = 18ms
	want := 18 * time.Millisecond
	if got := s.DecodeInterval(); got != want {
		t.Errorf("superlinear DecodeInterval = %s, want %s", got, want)
	}
}

func TestSchedulerConcurrentAcquireRelease(t *testing.T) {
	s := NewScheduler(SchedulerOptions{MaxBatch: 4})
	ctx := context.Background()

	var wg sync.WaitGroup
	var maxSeen int64
	var mu sync.Mutex

	for i := 0; i < 50; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			rel, err := s.Acquire(ctx)
			if err != nil {
				return
			}
			mu.Lock()
			if b := int64(s.BatchSize()); b > maxSeen {
				maxSeen = b
			}
			mu.Unlock()
			time.Sleep(time.Millisecond)
			rel()
		}()
	}
	wg.Wait()

	if maxSeen > 4 {
		t.Errorf("observed batch size %d, exceeds MaxBatch 4", maxSeen)
	}
	if got := s.BatchSize(); got != 0 {
		t.Errorf("final BatchSize = %d, want 0 (slot leak)", got)
	}
}

func TestSleepCtx(t *testing.T) {
	if !sleepCtx(context.Background(), time.Millisecond) {
		t.Error("sleepCtx returned false for a completed sleep")
	}
	if !sleepCtx(context.Background(), 0) {
		t.Error("sleepCtx(0) should report success")
	}

	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if sleepCtx(ctx, time.Second) {
		t.Error("sleepCtx returned true despite a cancelled context")
	}
}
