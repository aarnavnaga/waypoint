package mocksim

import (
	"context"
	"sync/atomic"
	"time"

	"github.com/aarnavnaga/waypoint/internal/openai"
	"github.com/aarnavnaga/waypoint/internal/tokenizer"
)

// Engine is the simulation core: it turns a chat request into a timed sequence
// of output tokens, accounting for prefix-cache reuse and batch pressure.
//
// It is deliberately transport-free so the timing model can be unit tested
// without HTTP.
type Engine struct {
	cfg    Config
	cache  *PrefixCache
	sched  *Scheduler
	gen    *Generator
	faults *Faults

	totalRequests   atomic.Uint64
	totalPromptToks atomic.Uint64
	totalOutputToks atomic.Uint64
	totalCachedToks atomic.Uint64
}

// NewEngine builds an Engine from cfg.
func NewEngine(cfg Config) *Engine {
	return &Engine{
		cfg:   cfg,
		cache: NewPrefixCache(cfg.PrefixCache.BlockSize, cfg.PrefixCache.CapacityBlocks),
		sched: NewScheduler(SchedulerOptions{
			MaxBatch:   cfg.Batching.MaxBatch,
			MaxQueue:   cfg.Batching.MaxQueue,
			BaseDecode: cfg.Decode.Interval(),
			Alpha:      cfg.Decode.Alpha,
			Beta:       cfg.Decode.Beta,
		}),
		gen:    NewGenerator(uint64(cfg.Seed)),
		faults: NewFaults(cfg.Faults, cfg.Seed),
	}
}

// Faults exposes the fault-injection knobs for the admin endpoint.
func (e *Engine) Faults() *Faults { return e.faults }

// Config returns the engine's configuration.
func (e *Engine) Config() Config { return e.cfg }

// Plan is the precomputed cost and content of one completion.
type Plan struct {
	PromptTokens int
	CachedTokens int
	// PrefillDuration is the simulated time before the first token, derived
	// only from the *uncached* portion of the prompt.
	PrefillDuration time.Duration
	OutputTokens    int
	// Text holds one fragment per output token.
	Text []string
}

// UncachedTokens reports how many prompt tokens had to be prefilled.
func (p Plan) UncachedTokens() int { return p.PromptTokens - p.CachedTokens }

// Plan computes the cost of a request and consults (and warms) the prefix cache.
//
// This is where the central simulated behavior lives: a request whose prompt
// prefix is already resident skips its prefill cost, which is what makes
// prefix-aware routing worth anything.
func (e *Engine) Plan(req *openai.ChatRequest) Plan {
	promptText := PromptKey(req)
	toks := tokenizer.Split(promptText)

	cached := e.cache.CachedTokens(toks)
	uncached := len(toks) - cached
	if uncached < 0 {
		uncached = 0
	}

	prefill := e.cfg.Prefill.Base.Duration +
		time.Duration(uncached)*e.cfg.Prefill.PerToken.Duration

	n := e.gen.OutputLength(req, e.cfg.Output.MinTokens, e.cfg.Output.MaxTokens)

	e.totalRequests.Add(1)
	e.totalPromptToks.Add(uint64(len(toks)))
	e.totalCachedToks.Add(uint64(cached))
	e.totalOutputToks.Add(uint64(n))

	return Plan{
		PromptTokens:    len(toks),
		CachedTokens:    cached,
		PrefillDuration: prefill,
		OutputTokens:    n,
		Text:            e.gen.Generate(req, n),
	}
}

// Emitter receives each generated token fragment. Returning an error aborts
// generation, which is how a disconnected client stops wasting backend time.
type Emitter func(index int, fragment string) error

// Admit waits for a batch slot and then simulates prefill. It returns once the
// model would be ready to emit its first token.
//
// Admission and prefill are separated from decoding so a caller can delay
// writing *any* response bytes until prefill finishes. That matters for two
// reasons: client-observed TTFT then genuinely includes prefill cost (otherwise
// prefix-cache savings would be invisible to a benchmark), and a request shed
// for a full queue can still receive a proper HTTP error instead of a truncated
// 200 stream.
//
// The returned release function must be called exactly once.
func (e *Engine) Admit(ctx context.Context, plan Plan) (release func(), err error) {
	release, err = e.sched.Acquire(ctx)
	if err != nil {
		return nil, err
	}

	if !sleepCtx(ctx, plan.PrefillDuration) {
		release()
		return nil, ctx.Err()
	}
	return release, nil
}

// Decode emits the planned tokens at the current batch-dependent rate. The
// caller must already hold a batch slot from Admit.
//
// The interval is re-read per token rather than computed up front, so a sequence
// that starts alone and is later joined by seven others genuinely slows down —
// the defining behavior of continuous batching.
func (e *Engine) Decode(ctx context.Context, plan Plan, emit Emitter) error {
	multiplier := e.faults.DecodeMultiplier()

	for i, fragment := range plan.Text {
		if err := ctx.Err(); err != nil {
			return err
		}
		if err := emit(i, fragment); err != nil {
			return err
		}

		interval := e.sched.DecodeInterval()
		if multiplier != 1.0 {
			interval = time.Duration(float64(interval) * multiplier)
		}
		if !sleepCtx(ctx, interval) {
			return ctx.Err()
		}
	}
	return nil
}

// Execute runs a full completion: admit, prefill, then decode.
func (e *Engine) Execute(ctx context.Context, plan Plan, emit Emitter) error {
	release, err := e.Admit(ctx, plan)
	if err != nil {
		return err
	}
	defer release()

	return e.Decode(ctx, plan, emit)
}

// Stats is the /stats response body.
type Stats struct {
	Name        string           `json:"name"`
	Models      []string         `json:"models"`
	Requests    uint64           `json:"requests"`
	PromptToks  uint64           `json:"prompt_tokens_total"`
	CachedToks  uint64           `json:"cached_prompt_tokens_total"`
	OutputToks  uint64           `json:"output_tokens_total"`
	PrefixCache PrefixCacheStats `json:"prefix_cache"`
	Scheduler   SchedulerStats   `json:"scheduler"`
	Faults      FaultState       `json:"faults"`
}

// Stats returns a snapshot of engine counters.
func (e *Engine) Stats() Stats {
	return Stats{
		Name:        e.cfg.Name,
		Models:      e.cfg.Models,
		Requests:    e.totalRequests.Load(),
		PromptToks:  e.totalPromptToks.Load(),
		CachedToks:  e.totalCachedToks.Load(),
		OutputToks:  e.totalOutputToks.Load(),
		PrefixCache: e.cache.Stats(),
		Scheduler:   e.sched.Stats(),
		Faults:      e.faults.Get(),
	}
}

// Reset clears cache contents and counters between benchmark runs.
func (e *Engine) Reset() {
	e.cache.Reset()
	e.totalRequests.Store(0)
	e.totalPromptToks.Store(0)
	e.totalCachedToks.Store(0)
	e.totalOutputToks.Store(0)
}
