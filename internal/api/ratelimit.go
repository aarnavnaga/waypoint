package api

import (
	"net/http"
	"strconv"
	"strings"
	"time"

	"github.com/aarnavnaga/waypoint/internal/openai"
	"github.com/aarnavnaga/waypoint/internal/ratelimit"
)

// checkRateLimit consumes budget for the request and writes a 429 if denied.
// It reports whether the request may proceed, and which resource bound.
func (s *Server) checkRateLimit(w http.ResponseWriter, r *http.Request, estTokens int64) (allowed bool, resource string) {
	if !s.cfg.RateLimit.Enabled {
		return true, ""
	}

	// Limits are keyed by API key. With authentication disabled there is no
	// tenant to attribute usage to, so fall back to the peer address — coarse,
	// but better than one global bucket shared by every caller.
	key := APIKeyFromContext(r.Context())
	if key == "" {
		key = clientIP(r)
	}

	decision, err := s.limiter.Allow(r.Context(), key, estTokens)
	if err != nil {
		// A degraded decision still carries the configured fail-open or
		// fail-closed verdict; log once and honor it.
		s.logger.WarnContext(r.Context(), "rate limiter degraded",
			"error", err, "fail_open", s.cfg.RateLimit.FailOpen, "allowed", decision.Allowed)
	}

	writeRateLimitHeaders(w, decision)

	if decision.Allowed {
		return true, ""
	}

	retryAfter := decision.RetryAfter
	if retryAfter <= 0 {
		retryAfter = time.Second
	}
	// Retry-After is specified in whole seconds; always round up, since
	// rounding down would invite a retry that is still too early.
	seconds := int(retryAfter.Seconds())
	if time.Duration(seconds)*time.Second < retryAfter {
		seconds++
	}
	if seconds < 1 {
		seconds = 1
	}
	w.Header().Set("Retry-After", strconv.Itoa(seconds))

	writeAPIError(w, http.StatusTooManyRequests,
		"rate limit exceeded for "+decision.Resource+"; retry in "+strconv.Itoa(seconds)+"s",
		openai.ErrTypeRateLimit, "rate_limit_exceeded")
	return false, decision.Resource
}

// writeRateLimitHeaders emits the conventional X-RateLimit-* headers so clients
// can self-throttle instead of discovering limits by being rejected.
func writeRateLimitHeaders(w http.ResponseWriter, d ratelimit.Decision) {
	if d.Degraded {
		return
	}
	h := w.Header()
	h.Set("X-RateLimit-Limit", strconv.FormatInt(d.Limit, 10))
	h.Set("X-RateLimit-Remaining", strconv.FormatInt(d.Remaining, 10))
}

// clientIP extracts a caller address, preferring forwarding headers set by a
// trusted load balancer.
func clientIP(r *http.Request) string {
	if v := r.Header.Get("X-Forwarded-For"); v != "" {
		// The left-most entry is the original client.
		if first, _, found := strings.Cut(v, ","); found {
			return strings.TrimSpace(first)
		}
		return strings.TrimSpace(v)
	}
	if v := r.Header.Get("X-Real-IP"); v != "" {
		return strings.TrimSpace(v)
	}
	return r.RemoteAddr
}
