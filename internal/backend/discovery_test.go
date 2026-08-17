package backend

import (
	"context"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/aarnavnaga/waypoint/internal/config"
)

func discovererFor(t *testing.T, handler http.Handler) (*Discoverer, *Backend) {
	t.Helper()

	srv := httptest.NewServer(handler)
	t.Cleanup(srv.Close)

	pool, err := NewPool([]config.Backend{{Name: "a", URL: srv.URL}})
	if err != nil {
		t.Fatalf("NewPool: %v", err)
	}
	b, _ := pool.Get("a")

	return NewDiscoverer(pool, DiscovererOptions{
		Client:  srv.Client(),
		Timeout: time.Second,
		Logger:  testLogger(),
	}), b
}

func TestRefreshPopulatesModels(t *testing.T) {
	d, b := discovererFor(t, http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write([]byte(`{"object":"list","data":[{"id":"mock-7b"},{"id":"mock-13b"}]}`))
	}))

	if err := d.Refresh(context.Background(), b); err != nil {
		t.Fatalf("Refresh: %v", err)
	}

	models := b.Models()
	if len(models) != 2 || models[0] != "mock-7b" || models[1] != "mock-13b" {
		t.Errorf("Models = %v, want [mock-7b mock-13b]", models)
	}
}

// Discovery failing must not make a backend look like it serves nothing, which
// would take it out of rotation entirely.
func TestRefreshFailureLeavesModelsUnchanged(t *testing.T) {
	fail := false
	d, b := discovererFor(t, http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		if fail {
			w.WriteHeader(http.StatusInternalServerError)
			return
		}
		_, _ = w.Write([]byte(`{"object":"list","data":[{"id":"mock-7b"}]}`))
	}))

	if err := d.Refresh(context.Background(), b); err != nil {
		t.Fatalf("Refresh: %v", err)
	}

	fail = true
	if err := d.Refresh(context.Background(), b); err == nil {
		t.Error("Refresh returned nil on a 500 response")
	}
	if got := b.Models(); len(got) != 1 || got[0] != "mock-7b" {
		t.Errorf("Models = %v after a failed refresh, want the previous list", got)
	}
	if !b.ServesModel("mock-7b") {
		t.Error("backend stopped serving its known model after a failed refresh")
	}
}

func TestRefreshIgnoresEmptyList(t *testing.T) {
	empty := false
	d, b := discovererFor(t, http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		if empty {
			_, _ = w.Write([]byte(`{"object":"list","data":[]}`))
			return
		}
		_, _ = w.Write([]byte(`{"object":"list","data":[{"id":"mock-7b"}]}`))
	}))

	_ = d.Refresh(context.Background(), b)
	empty = true
	_ = d.Refresh(context.Background(), b)

	if got := b.Models(); len(got) != 1 {
		t.Errorf("Models = %v, want the earlier non-empty list retained", got)
	}
}

func TestRefreshRejectsMalformedBody(t *testing.T) {
	d, b := discovererFor(t, http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write([]byte(`not json`))
	}))

	if err := d.Refresh(context.Background(), b); err == nil {
		t.Error("Refresh accepted a malformed body")
	}
}

func TestRefreshHonorsContext(t *testing.T) {
	d, b := discovererFor(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		<-r.Context().Done()
	}))
	d.timeout = 50 * time.Millisecond

	start := time.Now()
	if err := d.Refresh(context.Background(), b); err == nil {
		t.Error("Refresh returned nil against a hanging server")
	}
	if elapsed := time.Since(start); elapsed > time.Second {
		t.Errorf("Refresh took %s; the timeout was not applied", elapsed)
	}
}

func TestRunRefreshesThenStops(t *testing.T) {
	d, b := discovererFor(t, http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write([]byte(`{"object":"list","data":[{"id":"discovered"}]}`))
	}))
	d.interval = 10 * time.Millisecond

	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan struct{})
	go func() {
		d.Run(ctx)
		close(done)
	}()

	deadline := time.Now().Add(2 * time.Second)
	for time.Now().Before(deadline) {
		if len(b.Models()) == 1 && b.Models()[0] == "discovered" {
			break
		}
		time.Sleep(5 * time.Millisecond)
	}
	if got := b.Models(); len(got) != 1 || got[0] != "discovered" {
		t.Errorf("Models = %v, want [discovered]", got)
	}

	cancel()
	select {
	case <-done:
	case <-time.After(time.Second):
		t.Fatal("Run did not return after context cancellation")
	}
}
