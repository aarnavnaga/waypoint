// Package router selects which backend should serve a request.
//
// Policies are pluggable so their effect can be measured against each other
// under identical load; see bench/ and SPEC §8.
package router

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"sync/atomic"

	"github.com/aarnavnaga/waypoint/internal/backend"
	"github.com/aarnavnaga/waypoint/internal/config"
	"github.com/aarnavnaga/waypoint/internal/openai"
	"github.com/aarnavnaga/waypoint/internal/tokenizer"
)

// ErrNoBackends means no backend was eligible to serve the request.
var ErrNoBackends = errors.New("router: no available backend")

// Request is the routing-relevant view of a chat completion.
type Request struct {
	// Model is the requested model ID.
	Model string
	// PrefixKey is the leading span of the prompt used for affinity hashing.
	// Only the first routing.prefix_tokens tokens are included, so requests
	// sharing a system prompt hash together even when their questions differ.
	PrefixKey string
	// EstimatedTokens is the approximate cost used for load accounting.
	EstimatedTokens int64
}

// Policy chooses a backend from the eligible set.
//
// Implementations must be safe for concurrent use and must not block: Pick runs
// on the request path before any backend work begins.
type Policy interface {
	Pick(ctx context.Context, req *Request, backends []*backend.Backend) (*backend.Backend, error)
	// Name identifies the policy in metrics, logs and benchmark results.
	Name() string
}

// Router holds the active policy and allows swapping it at runtime.
//
// Runtime switching exists for benchmarking: comparing policies by restarting
// the gateway would also clear the backends' prefix caches, so each policy would
// be measured against a cold fleet and the comparison would be meaningless.
type Router struct {
	policy    atomic.Pointer[Policy]
	cfg       config.Routing
	estimator tokenizer.Estimator
}

// New builds a Router with the configured policy.
func New(cfg config.Routing, est tokenizer.Estimator) (*Router, error) {
	if est == nil {
		est = tokenizer.NewCharHeuristic()
	}
	r := &Router{cfg: cfg, estimator: est}
	if err := r.SetPolicy(cfg.Policy); err != nil {
		return nil, err
	}
	return r, nil
}

// SetPolicy swaps the active policy by name.
func (r *Router) SetPolicy(name string) error {
	p, err := buildPolicy(name, r.cfg)
	if err != nil {
		return err
	}
	r.policy.Store(&p)
	return nil
}

// PolicyName returns the active policy's name.
func (r *Router) PolicyName() string {
	if p := r.policy.Load(); p != nil {
		return (*p).Name()
	}
	return ""
}

// Pick selects a backend for req.
func (r *Router) Pick(ctx context.Context, req *Request, backends []*backend.Backend) (*backend.Backend, error) {
	if len(backends) == 0 {
		return nil, ErrNoBackends
	}
	p := r.policy.Load()
	if p == nil {
		return nil, errors.New("router: no policy configured")
	}
	return (*p).Pick(ctx, req, backends)
}

// BuildRequest derives the routing view of a chat request.
func (r *Router) BuildRequest(req *openai.ChatRequest) *Request {
	return &Request{
		Model:           req.Model,
		PrefixKey:       PrefixKey(req, r.cfg.PrefixTokens),
		EstimatedTokens: r.EstimateTokens(req),
	}
}

// EstimateTokens approximates a request's total cost: prompt tokens plus the
// output it is allowed to generate.
//
// Output is included because load is about what a backend is *committed* to
// produce, not what it has already consumed. A request with max_tokens=4000 will
// occupy a decode slot far longer than one with max_tokens=50, even though their
// prompts are the same size.
func (r *Router) EstimateTokens(req *openai.ChatRequest) int64 {
	var prompt int
	for _, m := range req.Messages {
		prompt += r.estimator.Count(m.Content)
		// Each message carries a few tokens of role/formatting overhead.
		prompt += 4
	}

	output := defaultAssumedOutputTokens
	if req.MaxTokens != nil && *req.MaxTokens > 0 {
		output = *req.MaxTokens
	}
	return int64(prompt + output)
}

// defaultAssumedOutputTokens is charged when a request sets no max_tokens.
// Underestimating here makes the gateway over-admit; overestimating wastes
// capacity. 256 matches the mock backend's default output ceiling.
const defaultAssumedOutputTokens = 256

// PrefixKey builds the affinity key from the leading tokens of a prompt.
//
// Only a bounded prefix is used, for two reasons. First, it is what the backend
// can actually reuse: KV blocks match on a *prefix*, so two conversations that
// share a system prompt but diverge later still share cache up to the point of
// divergence. Second, it makes multi-turn conversations sticky — turn 5 of a
// chat hashes to the same backend as turn 1, because the first N tokens have not
// changed, so the whole conversation accumulates on one server's cache.
func PrefixKey(req *openai.ChatRequest, prefixTokens int) string {
	if prefixTokens < 1 {
		prefixTokens = 1
	}

	var b strings.Builder
	// The model is part of the key: identical text on a different model is a
	// different KV cache entirely.
	b.WriteString(req.Model)
	b.WriteByte('\n')

	remaining := prefixTokens
	for _, m := range req.Messages {
		if remaining <= 0 {
			break
		}
		b.WriteString(m.Role)
		b.WriteByte(':')

		toks := tokenizer.Split(m.Content)
		if len(toks) > remaining {
			toks = toks[:remaining]
		}
		remaining -= len(toks)

		for _, t := range toks {
			b.WriteString(t)
		}
		b.WriteByte('\n')
	}
	return b.String()
}

// buildPolicy constructs a policy by name.
func buildPolicy(name string, cfg config.Routing) (Policy, error) {
	switch name {
	case config.PolicyRoundRobin:
		return NewRoundRobin(), nil
	case config.PolicyLeastLoaded, config.PolicyPrefixAffinity, config.PolicyPrefixAffinityLoadAware:
		return nil, fmt.Errorf("routing policy %q is not implemented yet", name)
	default:
		return nil, fmt.Errorf("unknown routing policy %q (valid: %s)",
			name, strings.Join(config.ValidPolicies, ", "))
	}
}

// RoundRobin distributes requests evenly, ignoring load and locality.
//
// This is the baseline every other policy is measured against. It is the obvious
// choice for stateless, uniform-cost backends and the wrong choice here: it
// maximizes the chance a prompt lands on a server that has never seen its
// prefix.
type RoundRobin struct {
	counter atomic.Uint64
}

// NewRoundRobin returns a RoundRobin policy.
func NewRoundRobin() *RoundRobin { return &RoundRobin{} }

// Name implements Policy.
func (*RoundRobin) Name() string { return config.PolicyRoundRobin }

// Pick implements Policy.
func (p *RoundRobin) Pick(_ context.Context, _ *Request, backends []*backend.Backend) (*backend.Backend, error) {
	if len(backends) == 0 {
		return nil, ErrNoBackends
	}
	// Subtract 1 so the first call uses index 0.
	idx := (p.counter.Add(1) - 1) % uint64(len(backends))
	return backends[idx], nil
}
