package backend

import (
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
	"net/http"
	"sync"
	"time"

	"github.com/aarnavnaga/waypoint/internal/openai"
)

// Discoverer keeps each backend's model list current.
//
// Models are discovered rather than configured so a fleet can be reconfigured
// (a server reloaded with a different model) without editing gateway config and
// restarting. Discovery failure is non-fatal: a backend whose list is unknown is
// treated as serving everything, so an outage of this loop degrades routing
// precision rather than availability.
type Discoverer struct {
	pool     *Pool
	client   *http.Client
	interval time.Duration
	timeout  time.Duration
	logger   *slog.Logger
}

// DiscovererOptions configures a Discoverer.
type DiscovererOptions struct {
	Client   *http.Client
	Interval time.Duration
	Timeout  time.Duration
	Logger   *slog.Logger
}

// NewDiscoverer builds a Discoverer.
func NewDiscoverer(pool *Pool, opts DiscovererOptions) *Discoverer {
	if opts.Client == nil {
		opts.Client = http.DefaultClient
	}
	if opts.Interval <= 0 {
		opts.Interval = 30 * time.Second
	}
	if opts.Timeout <= 0 {
		opts.Timeout = 2 * time.Second
	}
	if opts.Logger == nil {
		opts.Logger = slog.Default()
	}
	return &Discoverer{
		pool:     pool,
		client:   opts.Client,
		interval: opts.Interval,
		timeout:  opts.Timeout,
		logger:   opts.Logger,
	}
}

// Run refreshes immediately, then on each tick until ctx is cancelled.
func (d *Discoverer) Run(ctx context.Context) {
	d.RefreshAll(ctx)

	ticker := time.NewTicker(d.interval)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			d.RefreshAll(ctx)
		}
	}
}

// RefreshAll queries every backend in parallel.
func (d *Discoverer) RefreshAll(ctx context.Context) {
	var wg sync.WaitGroup
	for _, b := range d.pool.All() {
		wg.Add(1)
		go func(b *Backend) {
			defer wg.Done()
			if err := d.Refresh(ctx, b); err != nil {
				d.logger.Debug("model discovery failed",
					"backend", b.Name, "error", err)
			}
		}(b)
	}
	wg.Wait()
}

// Refresh queries one backend's /v1/models and updates its model list.
func (d *Discoverer) Refresh(ctx context.Context, b *Backend) error {
	reqCtx, cancel := context.WithTimeout(ctx, d.timeout)
	defer cancel()

	target := b.URL.JoinPath("/v1/models")
	req, err := http.NewRequestWithContext(reqCtx, http.MethodGet, target.String(), http.NoBody)
	if err != nil {
		return fmt.Errorf("build request: %w", err)
	}

	resp, err := d.client.Do(req)
	if err != nil {
		return fmt.Errorf("get %s: %w", target, err)
	}
	defer func() { _ = resp.Body.Close() }()

	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("%s returned %d", target, resp.StatusCode)
	}

	var list openai.ModelList
	if err := json.NewDecoder(resp.Body).Decode(&list); err != nil {
		return fmt.Errorf("decode model list: %w", err)
	}

	models := make([]string, 0, len(list.Data))
	for _, m := range list.Data {
		if m.ID != "" {
			models = append(models, m.ID)
		}
	}

	// An empty list is not an update: it would make the backend look like it
	// serves nothing, and the "unknown means anything" fallback is safer.
	if len(models) == 0 {
		return nil
	}

	b.SetModels(models)
	return nil
}
