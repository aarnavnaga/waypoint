# Waypoint — Design & Decision Log

This document is the running record of every nontrivial design decision: what was
chosen, what else was considered, and why the alternative lost. It is written to be
read in a design review, so each entry states the tradeoff rather than just the outcome.

---

## 1. System shape

Waypoint is a **stateless** L7 gateway. Every replica can serve every request; all
shared state (rate-limit buckets, caches, vector index) lives in Redis. Routing state
that is *not* shared — in-flight token counts, breaker state, health — is per-replica
and deliberately so (see §4.3).

```
client → LB → gateway replica → [policy picks backend] → model server
                   ↕                                    ↑
             Redis (limits, caches)          Toxiproxy (fault injection)
```

---

## 2. Decision log

### D1 — Go standard library `net/http` over a framework (chi/gin/echo)

**Chosen:** stdlib `net/http` with Go 1.22 pattern-based routing (`GET /v1/models`).

**Alternatives:** chi (middleware ergonomics), gin (speed, large ecosystem).

**Why:** the routing surface is ~8 endpoints. Go 1.22's `ServeMux` gained method and
wildcard patterns, which removes the main historical reason to reach for chi. The
gateway's hot path is a streaming proxy, where framework overhead and allocation
behavior matter more than handler ergonomics, and stdlib gives the most direct control
over flushing and body lifecycle. Fewer dependencies also keeps the supply chain small
for something that sits in front of every request.

**Cost:** we hand-roll middleware chaining and parameter extraction. Accepted — it is
roughly 40 lines.

### D2 — Configuration: YAML file as the source of truth, env vars for deployment knobs only

**Chosen:** full config in YAML; a short, explicit allow-list of `WAYPOINT_*` env
overrides (port, policy, Redis address, OTLP endpoint, log level, embedder URL, API keys).

**Alternatives:** (a) env-var-first (12-factor purist), (b) a reflection-driven
auto-override of every field.

**Why:** benchmarks must be reproducible, and a run is only reproducible if the config
that produced it can be committed as a single artifact. Env-var-first scatters the
experiment definition across a shell script. The reflection approach was rejected
because implicit overrides make it hard to tell, reading a result file, which knob was
actually in effect. The explicit list is the subset that genuinely differs per
*deployment* rather than per *experiment*.

**Also:** YAML decoding runs with `KnownFields(true)` so a typo (`prot: 8080`) is a
startup error instead of a silently ignored field that quietly reverts to a default.

### D3 — `Duration` wrapper type for YAML

**Chosen:** a `config.Duration` struct wrapping `time.Duration` with a custom
`UnmarshalYAML`, so the file can say `ttft_timeout: 5s`.

**Alternatives:** integer milliseconds (`ttft_timeout_ms: 5000`).

**Why:** units in the field name are a well-known source of operational error, and
`5s` is self-documenting in a config review. The cost is a wrapper type that must be
dereferenced as `.Duration` at use sites.

### D4 — Defaults live in code (`config.Default()`), not in the YAML file

**Chosen:** `Load` starts from `Default()` and decodes the file over it.

**Why:** a user's config file should express *intent* (what differs from sane behavior),
not restate 60 fields. It also means adding a new tunable does not break existing config
files — important because benchmark configs are committed and re-run later.

### D5 — Token estimation: byte heuristic on the hot path, split tokenizer in simulation

**Chosen:** two `tokenizer.Estimator` implementations behind one interface.
`CharHeuristic` (bytes ÷ 4) runs on the gateway's request path; `Split`, a
deterministic word/subword splitter, runs inside the mock backend.

**Alternatives:** (a) a real BPE tokenizer (tiktoken port) everywhere, (b) the
heuristic everywhere.

**Why:** routing and admission control need a count that is *consistent and monotone
in prompt size*, not exact — a 5% error in estimated load does not change which
backend is least loaded. A real tokenizer costs a model-specific vocabulary (a
megabytes-large asset that must match the backend's model) and turns an O(1) length
read into an O(n) scan on every request. The simulation side is different: prefix
*blocks* must align with plausible token boundaries, so there the slower splitter earns
its cost.

**Measured:** `TestHeuristicAgreesWithSplit` reports **4.9% relative error** across a
mixed prose/code corpus. The test fails above 35%, so a regression in either estimator
is caught rather than silently degrading routing quality. Interface placement means
swapping in a real tokenizer later is a one-line change at construction.

### D6 — Prefix cache keyed by *chained* block hashes, not per-block content hashes

**Chosen:** `hash[i] = FNV1a(hash[i-1] || tokens of block i)`.

**Alternatives:** hash each block independently by content.

**Why:** a transformer's KV values for a token depend on every preceding token, so a
block of identical text at a different position — or after a different prefix — holds
*different* KV state and must not be reused. Independent per-block hashing would
produce false hits and a simulation that reports cache savings that a real engine could
never deliver. This mirrors how vLLM and SGLang key their block tables.
`TestBlockHashesAreChained` and `TestPrefixCacheStopsAtFirstMiss` pin the behavior.

A trailing *partial* block is never cached, also matching real engines, which can only
commit a block once it is full.

### D7 — Admission and prefill happen before any response byte is written

**Chosen:** `Engine.Admit` (acquire batch slot → simulate prefill) is separate from
`Engine.Decode`. The HTTP handler calls `Admit` *before* writing status, headers or the
SSE role chunk.

**Why — this was a real bug, caught by measurement, not by tests.** The first
implementation wrote the `role: assistant` chunk immediately and then paid prefill
inside the streaming loop. Every unit test still passed, because they assert on token
*content*, not on when bytes hit the wire. But measuring with curl showed cold and warm
TTFT were identical at ~0.9 ms: the client saw its first byte before any simulated work
happened, so prefix-cache savings were **invisible to any client-side measurement** —
and the entire M8 routing benchmark would have reported noise.

After the split, the same measurement shows **235 ms cold vs. 10 ms warm** (~23x) on a
~900-token repeated prompt.

A second benefit: a request shed because the batch queue is full can now return a real
`503` with an OpenAI-shaped error body, instead of a truncated `200` stream — you
cannot un-send a status line.

`TestAdmitBlocksForPrefillBeforeDecoding` now guards the ordering directly.

### D8 — Simulation lives in `internal/mocksim`, not in `cmd/mockbackend`

**Chosen:** the prefix cache, scheduler, generator and fault state are a library;
`cmd/mockbackend` is thin wiring.

**Why:** SPEC §9's layout lists `cmd/mockbackend/`, and this adds a package the spec
does not name. The justification is testability — the timing model, LRU eviction and
batch-pressure curve are the parts most likely to be wrong, and they are far easier to
test as a library (`Engine.Plan` / `Admit` / `Decode`) than through HTTP. The gateway's
own view of a backend remains `internal/backend`, which is a different concern.

---

## 3. Deferred / open questions

- **Retry budget accounting** (M4): per-replica vs. Redis-shared. Leaning per-replica
  for hot-path latency; to be recorded in M4.
- **Semantic cache quality proxy** (M6/M8): how to score a cached answer against a
  fresh one without a human label set.

---

## 4. Notes for reviewers

### 4.1 Why the gateway does not implement batching

Continuous batching and KV-cache management belong to the model server, where the
scheduler can see token-level state. The gateway's leverage is *placement*: sending a
request to the server most likely to already hold its prompt prefix, and away from
servers whose batch is saturated. Everything in `internal/router` exists to exploit
backend-side behavior, not to replace it.

### 4.2 Why a mock backend instead of a real one for benchmarks

Routing gains come from prefix-cache hit rate and load distribution. Those are
observable without a GPU *provided* the backend faithfully simulates (a) prefill cost
proportional to uncached prefix tokens, (b) decode throughput that degrades with batch
size, and (c) a block-level LRU prefix cache. A simulated backend also makes runs
deterministic, which a real model server is not. Real backends stay supported but
optional (M10).

### 4.3 Why in-flight load is per-replica, not shared in Redis

Reading a shared counter on every routing decision adds a network round trip to the
hot path and makes the gateway's availability depend on Redis for *routing*, not just
for limits. With N replicas each seeing a random 1/N sample of traffic, per-replica
load estimates are biased but highly correlated with true load, which is sufficient for
choosing among backends. The cost — replicas can transiently disagree — is bounded by
the bounded-load constraint in the prefix-affinity policy. Revisit if replica count
grows past the point where each replica's sample is too thin.
