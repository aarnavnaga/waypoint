// Command gateway runs the Waypoint LLM inference gateway.
package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"log/slog"
	"net"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/aarnavnaga/waypoint/internal/admission"
	"github.com/aarnavnaga/waypoint/internal/api"
	"github.com/aarnavnaga/waypoint/internal/backend"
	"github.com/aarnavnaga/waypoint/internal/config"
	"github.com/aarnavnaga/waypoint/internal/ratelimit"
	"github.com/aarnavnaga/waypoint/internal/redisclient"
	"github.com/aarnavnaga/waypoint/internal/resilience"
	"github.com/aarnavnaga/waypoint/internal/router"
	"github.com/aarnavnaga/waypoint/internal/tokenizer"
)

// version is injected at build time via -ldflags.
var version = "dev"

func main() {
	if err := run(); err != nil {
		slog.Error("gateway exited with error", "error", err)
		os.Exit(1)
	}
}

func run() error {
	configPath := flag.String("config", "configs/waypoint.yaml", "path to the YAML config file")
	showVersion := flag.Bool("version", false, "print version and exit")
	flag.Parse()

	if *showVersion {
		fmt.Println(version)
		return nil
	}

	cfg, err := config.Load(*configPath)
	if err != nil {
		return fmt.Errorf("load config: %w", err)
	}

	logger := newLogger(cfg.Telemetry.LogLevel)
	slog.SetDefault(logger)

	pool, err := backend.NewPool(cfg.Backends)
	if err != nil {
		return fmt.Errorf("build backend pool: %w", err)
	}

	estimator := tokenizer.NewCharHeuristic()

	rt, err := router.New(cfg.Routing, estimator)
	if err != nil {
		return fmt.Errorf("build router: %w", err)
	}

	proxy := api.NewProxy(newUpstreamClient(cfg), api.Timeouts{
		TTFT:    cfg.Resilience.TTFTTimeout.Duration,
		Request: cfg.Resilience.RequestTimeout.Duration,
	}, logger)

	admitter := admission.New(admission.Options{
		MaxConcurrent: cfg.Server.MaxConcurrent,
		MaxQueue:      cfg.Server.MaxQueue,
		AgingInterval: 2 * time.Second,
	})

	upstream := newUpstreamClient(cfg)

	discoverer := backend.NewDiscoverer(pool, backend.DiscovererOptions{
		Client:   upstream,
		Interval: 30 * time.Second,
		Timeout:  cfg.Resilience.ConnectTimeout.Duration,
		Logger:   logger,
	})

	// Redis backs rate limits and caches. It is created unconditionally so
	// /readyz can report on it, but nothing blocks startup on it being up:
	// the gateway must be able to start during a Redis outage.
	rdb := redisclient.New(cfg.Redis)
	defer func() { _ = rdb.Close() }()

	var limiter ratelimit.Limiter = ratelimit.AllowAll{}
	if cfg.RateLimit.Enabled {
		limiter = ratelimit.NewRedisLimiter(rdb, ratelimit.Options{
			RPM:       int64(cfg.RateLimit.RPM),
			TPM:       int64(cfg.RateLimit.TPM),
			FailOpen:  cfg.RateLimit.FailOpen,
			KeyPrefix: cfg.Redis.KeyPrefix,
			Timeout:   cfg.Redis.Timeout.Duration,
		})
	}

	breakers := api.NewBreakers(pool, cfg.Resilience.Breaker)

	budget := resilience.NewRetryBudget(resilience.RetryBudgetOptions{
		Percent:             cfg.Resilience.RetryBudgetPct,
		MinRetriesPerSecond: 1,
		TTL:                 10 * time.Second,
	})

	dispatcher := api.NewDispatcher(api.DispatcherOptions{
		Pool:     pool,
		Router:   rt,
		Proxy:    proxy,
		Breakers: breakers,
		Budget:   budget,
		Backoff: resilience.NewBackoff(
			cfg.Resilience.Backoff.Base.Duration,
			cfg.Resilience.Backoff.Max.Duration,
		),
		MaxAttempts: cfg.Resilience.MaxAttempts,
		Logger:      logger,
	})

	healthChecker := resilience.NewHealthChecker(pool, resilience.HealthCheckerOptions{
		Interval:           cfg.Resilience.HealthCheck.Interval.Duration,
		Timeout:            cfg.Resilience.HealthCheck.Timeout.Duration,
		Path:               cfg.Resilience.HealthCheck.Path,
		HealthyThreshold:   cfg.Resilience.HealthCheck.HealthyThreshold,
		UnhealthyThreshold: cfg.Resilience.HealthCheck.UnhealthyThreshold,
		Client:             upstream,
		Logger:             logger,
	})

	server := api.NewServer(api.Options{
		Config:     cfg,
		Pool:       pool,
		Router:     rt,
		Proxy:      proxy,
		Dispatcher: dispatcher,
		Breakers:   breakers,
		Budget:     budget,
		Admission:  admitter,
		Limiter:    limiter,
		Estimator:  estimator,
		Logger:     logger,
		Redis:      rdb,
		Version:    version,
	})

	srv := &http.Server{
		Addr:              net.JoinHostPort("", fmt.Sprintf("%d", cfg.Server.Port)),
		Handler:           server.Handler(),
		ReadHeaderTimeout: cfg.Server.ReadHeaderTimeout.Duration,
		// No WriteTimeout: streaming responses are long-lived by design, and a
		// write deadline would sever healthy long generations mid-stream.
		// Per-request bounds come from context deadlines instead.
	}

	logger.Info("starting waypoint gateway",
		"version", version,
		"policy", cfg.Routing.Policy,
		"backends", len(cfg.Backends),
		"port", cfg.Server.Port,
		"max_concurrent", cfg.Server.MaxConcurrent,
		"max_queue", cfg.Server.MaxQueue,
		"ratelimit_enabled", cfg.RateLimit.Enabled,
		"ratelimit_fail_open", cfg.RateLimit.FailOpen,
	)

	ctx, stop := signal.NotifyContext(context.Background(), syscall.SIGINT, syscall.SIGTERM)
	defer stop()

	// Discover which models each backend serves, and keep it current.
	go discoverer.Run(ctx)
	// Active probing complements the breakers' passive signal: a breaker cannot
	// notice recovery on a backend it has stopped sending traffic to.
	go healthChecker.Run(ctx)

	errCh := make(chan error, 1)
	go func() {
		if err := srv.ListenAndServe(); err != nil && !errors.Is(err, http.ErrServerClosed) {
			errCh <- err
		}
	}()

	select {
	case err := <-errCh:
		return fmt.Errorf("listen: %w", err)
	case <-ctx.Done():
		logger.Info("shutdown signal received", "grace", cfg.Server.ShutdownGrace.Duration)
	}

	// Stop admitting first so queued requests fail fast rather than waiting out
	// the grace period for capacity that is never coming.
	admitter.Shutdown()

	shutdownCtx, cancel := context.WithTimeout(context.Background(), cfg.Server.ShutdownGrace.Duration)
	defer cancel()
	if err := srv.Shutdown(shutdownCtx); err != nil {
		return fmt.Errorf("shutdown: %w", err)
	}

	logger.Info("gateway stopped cleanly")
	return nil
}

// newUpstreamClient builds the HTTP client used for backend calls.
//
// Streaming shapes every choice here: compression is disabled because a
// compressing proxy buffers until it has enough bytes to compress, which adds
// latency to exactly the first token; there is no client-level Timeout because
// that would cap total generation time rather than time-to-first-token, and
// phase-specific deadlines are enforced by context instead.
func newUpstreamClient(cfg config.Config) *http.Client {
	transport := &http.Transport{
		DialContext: (&net.Dialer{
			Timeout:   cfg.Resilience.ConnectTimeout.Duration,
			KeepAlive: 30 * time.Second,
		}).DialContext,
		// Inference calls are long-lived and few per host, but connection reuse
		// still matters: a fresh TCP+TLS handshake per request would show up
		// directly in TTFT.
		MaxIdleConns:          256,
		MaxIdleConnsPerHost:   64,
		IdleConnTimeout:       90 * time.Second,
		TLSHandshakeTimeout:   cfg.Resilience.ConnectTimeout.Duration,
		ExpectContinueTimeout: time.Second,
		DisableCompression:    true,
		ForceAttemptHTTP2:     true,
	}

	return &http.Client{Transport: transport}
}

func newLogger(level string) *slog.Logger {
	var lvl slog.Level
	if err := lvl.UnmarshalText([]byte(level)); err != nil {
		lvl = slog.LevelInfo
	}
	return slog.New(slog.NewJSONHandler(os.Stdout, &slog.HandlerOptions{
		Level: lvl,
		ReplaceAttr: func(_ []string, a slog.Attr) slog.Attr {
			if a.Key == slog.TimeKey {
				a.Value = slog.StringValue(time.Now().UTC().Format(time.RFC3339Nano))
			}
			return a
		},
	}))
}
