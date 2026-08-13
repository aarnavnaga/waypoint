// Command mockbackend simulates an LLM inference server: prefill cost
// proportional to uncached prompt tokens, a block-level KV prefix cache,
// continuous batching whose per-token cost rises with batch size, and
// injectable faults.
//
// It exists so routing and resilience work can be measured without a GPU.
package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"log/slog"
	"net/http"
	"os"
	"os/signal"
	"strconv"
	"syscall"
	"time"

	"github.com/aarnavnaga/waypoint/internal/mocksim"
)

var version = "dev"

func main() {
	if err := run(); err != nil {
		slog.Error("mock backend exited with error", "error", err)
		os.Exit(1)
	}
}

func run() error {
	configPath := flag.String("config", "", "path to a mock backend YAML config (optional)")
	showVersion := flag.Bool("version", false, "print version and exit")
	flag.Parse()

	if *showVersion {
		fmt.Println(version)
		return nil
	}

	cfg, err := loadConfig(*configPath)
	if err != nil {
		return err
	}

	logger := slog.New(slog.NewJSONHandler(os.Stdout, &slog.HandlerOptions{Level: slog.LevelInfo})).
		With("backend", cfg.Name)
	slog.SetDefault(logger)

	engine := mocksim.NewEngine(cfg)
	server := mocksim.NewServer(engine, logger)
	// /admin/crash must genuinely terminate the process: the M8 resilience
	// experiment needs a backend that dies mid-run, not one that merely errors.
	server.SetExitFunc(func(code int) {
		logger.Error("exiting on admin request", "code", code)
		os.Exit(code)
	})

	srv := &http.Server{
		Addr:              cfg.Addr,
		Handler:           server.Handler(),
		ReadHeaderTimeout: 10 * time.Second,
		// No write timeout: streaming responses are long-lived by design.
	}

	logger.Info("mock backend listening",
		"addr", cfg.Addr,
		"version", version,
		"models", cfg.Models,
		"max_batch", cfg.Batching.MaxBatch,
		"prefix_block_size", cfg.PrefixCache.BlockSize,
		"prefix_capacity_blocks", cfg.PrefixCache.CapacityBlocks,
		"decode_tps", cfg.Decode.TokensPerSecond,
	)

	ctx, stop := signal.NotifyContext(context.Background(), syscall.SIGINT, syscall.SIGTERM)
	defer stop()

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
		logger.Info("shutdown signal received")
	}

	shutdownCtx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	if err := srv.Shutdown(shutdownCtx); err != nil {
		return fmt.Errorf("shutdown: %w", err)
	}
	return nil
}

// loadConfig reads YAML when given, then applies MOCKBACKEND_* overrides so one
// image can run as mock-1/2/3 in Compose without three config files.
func loadConfig(path string) (mocksim.Config, error) {
	cfg := mocksim.DefaultConfig()
	if path != "" {
		loaded, err := mocksim.LoadConfig(path)
		if err != nil {
			return mocksim.Config{}, err
		}
		cfg = loaded
	}

	if v := os.Getenv("MOCKBACKEND_NAME"); v != "" {
		cfg.Name = v
	}
	if v := os.Getenv("MOCKBACKEND_ADDR"); v != "" {
		cfg.Addr = v
	}
	if v := os.Getenv("MOCKBACKEND_SEED"); v != "" {
		seed, err := strconv.ParseInt(v, 10, 64)
		if err != nil {
			return mocksim.Config{}, fmt.Errorf("MOCKBACKEND_SEED: %w", err)
		}
		cfg.Seed = seed
	}
	if v := os.Getenv("MOCKBACKEND_MAX_BATCH"); v != "" {
		n, err := strconv.Atoi(v)
		if err != nil {
			return mocksim.Config{}, fmt.Errorf("MOCKBACKEND_MAX_BATCH: %w", err)
		}
		cfg.Batching.MaxBatch = n
	}
	if v := os.Getenv("MOCKBACKEND_PREFIX_CAPACITY_BLOCKS"); v != "" {
		n, err := strconv.Atoi(v)
		if err != nil {
			return mocksim.Config{}, fmt.Errorf("MOCKBACKEND_PREFIX_CAPACITY_BLOCKS: %w", err)
		}
		cfg.PrefixCache.CapacityBlocks = n
	}

	if err := cfg.Validate(); err != nil {
		return mocksim.Config{}, err
	}
	return cfg, nil
}
