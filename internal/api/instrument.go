package api

import (
	"context"
	"strconv"
	"time"

	"github.com/aarnavnaga/waypoint/internal/telemetry"
)

// StartGaugeCollector periodically samples state that is naturally a gauge:
// per-backend load, health, breaker state and queue depth.
//
// These are sampled on a ticker rather than updated on every request because
// they describe *current state*, not events. Updating them inline would add
// metric writes to the hot path to produce values a scrape reads at most once
// per interval anyway.
func (s *Server) StartGaugeCollector(ctx context.Context, interval time.Duration) {
	if s.metrics == nil {
		return
	}
	if interval <= 0 {
		interval = time.Second
	}

	ticker := time.NewTicker(interval)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			s.collectGauges()
		}
	}
}

func (s *Server) collectGauges() {
	m := s.metrics

	for _, b := range s.pool.All() {
		m.BackendInflightToken.WithLabelValues(b.Name).Set(float64(b.InflightTokens()))

		healthy := 0.0
		if b.Available() {
			healthy = 1
		}
		m.BackendHealthy.WithLabelValues(b.Name).Set(healthy)

		if s.breakers != nil {
			if breaker := s.breakers.For(b.Name); breaker != nil {
				m.BreakerState.WithLabelValues(b.Name).
					Set(telemetry.BreakerStateValue(breaker.State().String()))
			}
		}
	}

	if s.admission != nil {
		stats := s.admission.Stats()
		m.QueueDepth.Set(float64(stats.Queued))
		m.AdmissionActive.Set(float64(stats.Active))
	}
}

// recordRequest emits the per-request metrics for a completed exchange.
func (s *Server) recordRequest(status int, backendName, cacheResult string, ttft, duration time.Duration, completionTokens int) {
	if s.metrics == nil {
		return
	}

	m := s.metrics
	policy := s.router.PolicyName()
	statusLabel := strconv.Itoa(status)
	if backendName == "" {
		backendName = "none"
	}

	m.RequestsTotal.WithLabelValues(statusLabel, backendName, policy, cacheResult).Inc()
	m.DurationSeconds.WithLabelValues(statusLabel, backendName, policy).Observe(duration.Seconds())

	// TTFT is only meaningful when tokens were actually produced; recording a
	// zero for every error would drag the percentiles down and hide real
	// regressions.
	if ttft > 0 {
		m.TTFTSeconds.WithLabelValues(backendName, policy, cacheResult).Observe(ttft.Seconds())
	}

	// Throughput is measured over the *decode* phase only — total duration
	// includes prefill, which would understate tokens/sec for long prompts.
	if completionTokens > 0 && duration > ttft && ttft > 0 {
		decode := (duration - ttft).Seconds()
		if decode > 0 {
			m.TokensPerSecond.WithLabelValues(backendName).Observe(float64(completionTokens) / decode)
		}
	}
}

// recordCacheResult updates cache counters.
func (s *Server) recordCacheResult(source string, hit bool) {
	if s.metrics == nil {
		return
	}
	if hit {
		s.metrics.CacheHitsTotal.WithLabelValues(source).Inc()
		return
	}
	s.metrics.CacheMissesTotal.WithLabelValues(source).Inc()
}

// recordRateLimitRejection counts a 429 by the resource that bound.
func (s *Server) recordRateLimitRejection(resource string) {
	if s.metrics == nil {
		return
	}
	s.metrics.RateLimitRejections.WithLabelValues(resource).Inc()
}
