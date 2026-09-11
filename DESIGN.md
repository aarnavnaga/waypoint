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

---

## 5. M2 decisions — pass-through proxy

### D9 — Nothing is written to the client until the upstream response is known good

**Chosen:** `Proxy.Dispatch` writes no status line, headers or body until it has an
upstream response it intends to relay. `Outcome.ResponseStarted` records the moment
that changes.

**Why:** HTTP has no undo. Once `200 OK` and a few SSE chunks are on the wire, the
gateway cannot retry on another backend, cannot convert the response into a `502`, and
cannot apologize — the client is already parsing a stream. Making "has the response
started?" an explicit, single-source-of-truth field (rather than inferring it from
error types) is what lets M4's retry logic be correct by construction: `Retryable()` is
simply `!ResponseStarted`.

This also means a 5xx from the backend never reaches the client as a 5xx body; it
becomes a gateway-generated `502` with an OpenAI-shaped envelope, and the attempt stays
eligible for failover.

### D10 — 4xx is passed through; 5xx is not

**Chosen:** upstream `4xx` is relayed verbatim to the client and marks the response
started. Upstream `5xx` is converted into a dispatch error and left retryable.

**Why:** the distinction is *whose fault it is*. A 400 means the request is malformed —
every backend will reject it identically, so retrying wastes capacity and delays the
error the client needs to see. A 503 means that particular server is unhappy, which
says nothing about its peers.

### D11 — Separate TTFT and total-request deadlines

**Chosen:** two independent bounds. The TTFT timer cancels the attempt if no *content*
token arrives in time, and is stopped the moment one does; the overall request deadline
comes from the parent context.

**Alternatives:** a single `http.Client.Timeout`.

**Why:** a client timeout would bound *total* generation, which is exactly wrong for
streaming — a healthy 4,000-token answer legitimately takes minutes, while a backend
that accepts a connection and then stalls should be detected in seconds. These are
different failure modes and need different limits. The mock backend's `hang` fault
exists to test precisely this: the TCP connection is fine, so a transport-level timeout
would not fire.

The role chunk (`delta.role` with empty content) deliberately does **not** satisfy the
TTFT deadline — counting it would mean a stalled backend could hold a request open
forever by sending one empty chunk, and it would understate measured TTFT by the whole
prefill.

### D12 — The `Authorization` header is not forwarded upstream

**Chosen:** an explicit allow-list of passthrough headers (`traceparent`, `tracestate`,
`X-Request-Id`, `X-Waypoint-Cache`). Client credentials are dropped.

**Why:** the client's API key authenticates it to *Waypoint*, not to the model servers.
Forwarding it spreads tenant credentials across every backend in the fleet, so a single
compromised or misconfigured model server leaks keys for every tenant that ever routed
to it. The backends sit on a trusted network and have their own (absent) auth model.

### D13 — Admission control is an aging priority queue, not a plain semaphore

**Chosen:** `internal/admission` is a bounded heap ordered by effective priority, where
waiting time improves a request's position.

**Why aging:** strict priority queues starve. A sustained flood of interactive traffic
would leave batch requests queued indefinitely, which shows up as a p99.9 that never
recovers rather than as an outage.

**How it is implemented — and a bug this caused.** The first version computed effective
priority *inside* the heap's `Less` method from a clock read at pop time. This does not
work: `container/heap` establishes its ordering invariant when an element is **pushed**,
so mutating the comparison function afterwards does not reorder anything already in the
heap. The aging test failed, correctly, showing a freshly arrived interactive request
beating a batch request that had waited five seconds.

The fix is the standard one: convert aging into a **static deadline** computed once at
enqueue (`deadline = enqueued + class × agingInterval`) and order by it. This is
earliest-deadline-first, is equivalent to a priority that improves while waiting, and
keeps the heap invariant valid without re-heapifying on every release.

### D14 — `statusRecorder` must forward `Flush`

A middleware that wraps `http.ResponseWriter` to capture the status code silently breaks
streaming, because the inner handler's `w.(http.Flusher)` assertion then fails against
the wrapper and every token sits in a buffer. The logging middleware's recorder
implements `Flush()` explicitly for this reason. This is a classic, quiet Go HTTP bug:
nothing errors, responses just stop streaming.

---

## 6. M3 decisions — routing policies

### D15 — Consistent hashing with bounded loads for prefix affinity

**Chosen:** a hash ring with ~150 virtual nodes per backend, plus the bounded-load rule
from Mirrokni, Thorup & Zadimoghaddam (2016): a key goes to its preferred backend only
if that backend is under a ceiling of `c × average load`, otherwise the walk continues
clockwise.

**Alternatives:** (a) `hash(prefix) % N`, (b) rendezvous (highest-random-weight) hashing,
(c) plain consistent hashing with no load bound.

**Why not modulo:** the eligible backend set changes constantly — health flaps, breakers
open, servers saturate. Removing one backend of N under modulo hashing remaps
essentially *every* key, which invalidates every prefix cache in the fleet
simultaneously. That turns a single backend failure into a fleet-wide cold-cache event,
precisely when capacity is already reduced. Measured here: removing 1 of 5 backends
remaps **19.3%** of keys (ideal 1/N = 20%), versus ~80% for modulo.

**Why not rendezvous:** rendezvous hashing gives better key distribution without virtual
nodes and is simpler to reason about, but lookup is O(N) per request against O(log N)
for a sorted ring, and it has no natural place to attach the bounded-load rule — the
"walk to the next node" step is what makes spillover cheap and ordered. With N small
(3–10 backends) the O(N) cost would be acceptable; the deciding factor was bounded-load
support.

**Why bounded loads:** prompt popularity is heavily skewed in practice (a handful of
system prompts serve most traffic). Plain consistent hashing has no safety valve, so the
backend owning the most popular prompt is buried while its peers idle. The ceiling
converts unbounded tail latency into a bounded one. `TestPrefixAffinityBoundedLoadUnderSkew`
drives 200 requests that *all* share one prefix and asserts no backend exceeds the
ceiling, while also asserting the load genuinely spilled to more than one backend (so
the test cannot pass trivially).

### D16 — Two affinity policies that differ only in their fallback

`prefix_affinity` walks to the **next backend on the ring** when the preferred one is
over capacity; `prefix_affinity_load_aware` jumps to the **globally least loaded**.

The first keeps a stable, locality-preserving fallback order — the spillover target for
a given key is always the same backend, so even overflow traffic has cache affinity. The
second optimizes purely for balance. Which is better depends on how skewed the workload
is and how expensive a cache miss is, which is an empirical question; both exist so M8
can answer it rather than this document asserting it.

### D17 — FNV-1a alone is not an adequate ring hash

**This was a real bug, and the kind that produces plausible-looking wrong benchmarks.**

The ring originally hashed with FNV-1a. All unit tests about *stability* and *bounded
load* passed, because those properties do not depend on the hash being well distributed.
The distribution test did not:

```
arc share of keyspace:  mock-1 10.8%  mock-2 40.1%  mock-3 30.0%  mock-4 19.1%
key landing counts:     mock-1 90     mock-2 100    mock-3 210    mock-4 0
```

One backend of four owned 40% of the keyspace; another received **none** of a 400-key
sample. The cause is FNV-1a's weak avalanche on short, highly similar inputs — and ring
keys are exactly that (`mock-1#0`, `mock-2#0`, …, differing in one byte). Correlated
inputs produce correlated hashes, so virtual nodes clump instead of spreading, defeating
the entire purpose of having 150 of them.

The fix is to run the FNV output through MurmurHash3's 64-bit finalizer (`mix64`), an
invertible bit mixer that costs a handful of ALU operations. After it:

```
arc share of keyspace:  mock-1 25.8%  mock-2 22.6%  mock-3 26.3%  mock-4 25.3%
key landing counts:     mock-1 511    mock-2 456    mock-3 542    mock-4 491   (of 2000)
```

**Why not `hash/maphash`:** it is a good, fast, well-distributed hash — and completely
wrong here, because its seed is randomized per process. Two gateway replicas would build
different rings and send the same prefix to different backends, fragmenting the caches
the policy exists to exploit. Ring hashing must be deterministic across processes and
across restarts.

`TestHashRingArcsAreBalanced`, `TestHashRingKeyDistribution` and `TestMix64Avalanche`
now guard this, since nothing else would have caught it.

### D18 — The ring is cached and rebuilt only when the backend set changes

The eligible set is recomputed per request (health, breaker state, capacity, model), but
it changes rarely. Rebuilding a 450–1500 point sorted ring on every request would be
real hot-path cost for no benefit, so the ring is memoized behind an atomic pointer keyed
by a fingerprint of the backend names. A set change transparently rebuilds exactly once.

---

## 7. M4 decisions — resilience

### D19 — `Permits()` and `Allow()` are separate breaker operations

**This was a real bug that wedged the breaker permanently.**

The breaker originally exposed only `Allow()`, which both answers "may I send?"
*and* consumes one of the limited half-open probe slots. `backend.Available()` was wired
to it, so the router called `Allow()` for **every backend on every request** while
filtering candidates — including backends it then did not dispatch to.

The consequence: within a few requests, all half-open probe slots were consumed by
eligibility checks that never produced a matching `Record()`. With no results arriving,
`halfOpenSucceeded` never reached its threshold, `Allow()` refused everything, and the
breaker sat in half-open forever. A backend that had fully recovered could never return
to rotation. `TestBreakerRecoversAfterBackendHeals` caught it;
`TestPermitsDoesNotConsumeProbeSlots` now pins the invariant.

The fix separates the two concerns:

- `Permits()` — a pure query, used for candidate filtering. No mutation, safe to call
  thousands of times.
- `Allow()` — reserves a probe slot, called exactly once at dispatch, and obliges the
  caller to call `Record()`.

The general lesson: **a method that both answers a question and mutates state cannot be
used for speculative evaluation**, and routing is inherently speculative — it asks every
candidate before choosing one.

### D20 — Retry budget, not just capped attempts

Capping attempts at 3 bounds *one request's* amplification but not the *fleet's*. If a
backend starts failing everything, every request retries, and the fleet instantly emits
3x its normal rate at a system already in trouble. That is how a partial outage becomes
a total one.

The budget is a token bucket: each request deposits `retry_budget_pct/100` tokens, each
retry withdraws one. When failures are rare — the case retries are actually *for* —
budget is always available. When failures are widespread the budget empties and retries
stop. `TestRetryBudgetStopsRetryStorm` drives 1,000 requests where every attempt fails
and measures a retries/requests ratio of **0.099** against a configured 10%, versus 2.0
with no budget.

Tokens decay over a TTL so a quiet period cannot bank budget to fund a later burst, and
accumulation is capped for the same reason.

**Open question resolved from M0:** the budget is per-replica, not Redis-shared. Sharing
it would add a network round trip to the failure path — precisely when the system is
least healthy — and the guarantee it buys (exact global ratio) is not worth it. With N
replicas each enforcing the ratio locally, the fleet-wide ratio is the same in
expectation.

### D21 — Full jitter on backoff

Delay is drawn uniformly from `[0, min(max, base·2^attempt))` rather than being set to
the cap. If every client that failed at the same instant waits the same backoff, they
all retry at the same instant too, reproducing the burst that caused the failure. Full
jitter decorrelates them. `TestBackoffFullJitterSpreadsDelays` asserts the draws
actually span the interval instead of clustering.

### D22 — Client cancellation is not a backend failure

When a client hangs up mid-stream the dispatcher records *nothing* against the breaker.
Counting it would mean a burst of impatient clients — or one aggressive client timeout
setting — could trip every breaker in the fleet and cause a real outage from a
non-problem. `TestClientCancellationDoesNotTripBreaker` drives 30 abandoned streams and
asserts the breaker stays closed.

The same reasoning applies to active health probes: a probe aborted because the
*gateway* is shutting down is discarded rather than recorded, so a restart does not
leave every backend marked unhealthy on the way out.

### D23 — Active probing and passive breakers are both required

They answer different questions and neither subsumes the other.

A circuit breaker observes **real traffic**, so it detects failures that only appear
under load — the kind a `/healthz` endpoint returning 200 will never reveal. But it has
a blind spot: once it opens, it stops sending traffic, so it has no way to learn that
the backend recovered. That is what the half-open probe is for, and it is a deliberately
thin trickle.

An active health checker probes **regardless of traffic**, so it can notice recovery on
a backend nobody is using. But it only sees whatever the probe endpoint reports, which
is usually a trivial handler that stays healthy while the inference path is broken.

Health flips require N *consecutive* agreeing probes, not one. A single probe is a noisy
sample, and flipping a backend out of rotation on one timeout makes traffic oscillate —
which is itself a cause of timeouts.

### Measured resilience behavior (Toxiproxy, 3 backends)

| Scenario | Result |
|---|---|
| One backend killed mid-run (transport severed) | **100%** availability (30/30) |
| One backend +2s latency, 1.5s TTFT deadline | 15/15 succeeded, worst latency **1.52s** |
| Breaker trip against a dead backend | opened after **13** requests |
| Breaker recovery after backend restored | closed **2.56s** after opening |
| One backend at a 30% error rate, 3 attempts | **100%** availability |

Toxiproxy covers transport faults (severed connections, added latency); the mock
backend's own knobs cover application faults (HTTP 500s, accept-then-stall) that a
TCP-level proxy cannot synthesize.

---

## 8. M5 decisions — distributed rate limiting

### D24 — The token bucket is a Redis Lua script, not GET/compute/SET

**Chosen:** one atomic Lua script evaluating and debiting both buckets.

**Why:** with a read-modify-write sequence, two replicas checking the same key
concurrently both read the old value, both conclude they are under the limit, and both
admit. With N replicas under load the effective limit becomes some multiple of the
configured one, and it is not even a stable multiple — it depends on contention. Redis
runs scripts atomically on a single thread, so check-and-decrement cannot interleave.

Measured: `TestRateLimitHoldsAcrossTwoReplicas` drives 200 requests through two
independent gateway processes sharing one key against an RPM of 40, and gets **exactly
40 allowed, 160 limited**. `TestConcurrentCallersDoNotOverAdmit` releases 100 goroutines
simultaneously against a bucket of 20 and admits at most 23.

### D25 — Both limits are evaluated before either is debited

A request that fails the *token* limit must not consume *request* budget. Otherwise a
client that is persistently over its token quota would also drain its request bucket
with calls that were never served — being punished twice for one rejection, and taking
longer to recover than the limit intends. The script computes both verdicts first, then
debits only on success. `TestDeniedRequestDoesNotConsumeOtherBucket` pins this.

### D26 — Token bucket over fixed or sliding window counters

**Chosen:** token bucket (capacity + refill rate).

**Alternatives:** fixed-window counters, sliding-window log.

**Why:** fixed windows allow a 2x burst across a boundary — a client can spend its whole
minute's budget in the last second of one window and again in the first second of the
next. A sliding-window log is exact but stores a timestamp per request, which is
unbounded memory per key. A token bucket bounds burst to the bucket capacity, refills
smoothly rather than in steps, and stores two numbers per key regardless of traffic.

Buckets start **full**, so a first-time caller is not penalized for having no history.

### D27 — Fail open by default

When Redis is unreachable the limiter returns the configured verdict and marks the
decision `Degraded`.

**Why open:** rate limits exist to protect backend capacity from abuse. Failing closed
converts a Redis outage — a degradation of a *protective* subsystem — into a total
serving outage for every tenant. An unprotected but working API is nearly always better
than a protected but dead one, and the backends still have their own admission control
and breakers as a second line of defense.

**When closed is right:** if the limit encodes a billing or safety boundary rather than
a capacity guard, over-serving may be worse than not serving, so the behavior is
configurable.

This only works if failure is detected *fast* — a fail-open limiter that still waits a
5-second Redis timeout has added 5 seconds to every request during the outage. Redis
timeouts are therefore short (200ms default) and `TestRedisFailureIsFast` asserts it.

### D28 — Rate limiting runs before admission control

An over-quota request should cost as little as possible. Admitting it into the queue
first would let an over-quota tenant occupy concurrency slots that paying traffic needs,
which is backwards — the cheapest possible rejection is the point.

### D29 — Keyed by API key, falling back to client IP

With authentication enabled the key is the tenant. With it disabled there is no tenant
to attribute usage to, so the limiter falls back to the peer address (honoring
`X-Forwarded-For` from a trusted LB). Coarse, but far better than a single global bucket
in which one noisy caller starves everyone.

---

## 9. M6 decisions — caching

### D30 — Exact first, then semantic

Exact lookup is one Redis `GET`. Semantic lookup costs an embedding round trip plus a
vector search, and carries a correctness risk exact matching does not have. There is
never a case where trying semantic first is better, so the order is fixed.

### D31 — Only `temperature == 0` is cacheable

This is a **correctness** rule, not an optimization. Above temperature 0 the model is
sampling: two identical requests are *supposed* to produce different answers, and that
variation is what the caller asked for. Serving a cached response silently removes it.
A request with no temperature at all defaults to 1.0 and is therefore not cacheable
either — a subtlety worth stating, because "they didn't set it" is easy to mistake for
"they don't care".

### D32 — Cache-key normalization is deliberately conservative

Only leading/trailing whitespace is trimmed. Case, punctuation and internal spacing are
preserved, because a model can legitimately answer differently for each.

The asymmetry drives this: **a missed hit costs latency; a wrong hit returns an answer to
a question the user did not ask.** Aggressive normalization buys a slightly better hit
rate by taking on correctness risk, which is the wrong trade for a cache in front of a
model. `TestExactKeyPreservesMeaningfulDifferences` pins it.

`stream` is excluded from the key: the content is identical and only the framing
differs, so both forms should share one entry.

### D33 — Chunk boundaries are stored, not just the joined text

An entry keeps the original per-token fragments so a cached streaming response replays
with the same granularity a live one had. Storing only the joined string would make
cached responses arrive as one giant chunk — observably different for any client that
renders incrementally, and a regression users would notice as "the cache makes it feel
worse even though it's faster".

### D34 — Semantic cache guards: similarity is necessary but not sufficient

A semantic hit requires **all** of: identical model, identical sampling parameters,
identical preceding conversation context, and similarity above the threshold. Only the
final user message is compared by embedding.

Why only the final message: embedding the whole conversation lets a long shared history
dominate the vector, so two unrelated questions after the same 2,000-token system prompt
look nearly identical. The preceding context is folded in as an exact-match hash instead,
which is strictly safer — different conversations can never collide even if their last
messages are identical.

Why parameters must match: a cached greedy answer is not a valid response to a request
with different decoding settings, however similar the question.

### D35 — FLAT vector index over HNSW

A response cache holds thousands of entries, not millions. At that size an exhaustive
scan is fast and exact, while HNSW adds recall tuning (`M`, `efConstruction`,
`efRuntime`) and the possibility of silently missing a valid neighbor. HNSW becomes the
right call an order of magnitude higher.

### D36 — Cache writes are asynchronous and failures are always non-fatal

The client already has its bytes when the write happens, so storing must never be on the
path that delays them. Writes use a **detached context**: the request context is
cancelled the instant the handler returns, so inheriting it would abort essentially every
cache write.

Every cache failure — Redis down, embedder timing out, a corrupt stored entry —
degrades to a miss. A cache that can take down the service it accelerates is worse than
no cache at all.

### D37 — Response capture tees rather than buffers

Caching a streamed response requires observing it in flight, because it is written
incrementally and is gone once sent. `captureWriter` forwards bytes to the client
*first*, then parses the SSE it saw. Buffering the whole response to cache it afterwards
would convert streaming into non-streaming, destroying the property M2 exists to provide.

Two details this requires: writes do not align with event boundaries, so parsing happens
up to the last complete `\n\n` with the remainder kept pending; and the wrapper must
implement `Flush()`, or wrapping the `ResponseWriter` silently disables streaming.

Capture is skipped entirely when the request is not cacheable, so the common
`temperature > 0` path pays nothing.

### Measured cache behavior

| Measurement | Result |
|---|---|
| Exact hit vs. live request (non-streaming) | **25.9x faster** (51.3ms → 2.0ms) |
| Cached streaming replay | 10 of 10 original chunk boundaries preserved |
| Backend requests served on a cache hit | **0** |
| Semantic hit on a reworded question | similarity **0.849** (threshold 0.6) |

---

## 10. M7 decisions — observability

### D38 — The root span is started in middleware, not in the handler

**This was a real bug.** The root span was originally created inside
`handleChatCompletions`, which sits *inside* the logging middleware. The access log
therefore ran with the pre-span context, and every access log line was emitted without a
`trace_id`. Traces were fine and logs were fine; the *link between them* — the single
most useful thing about having both — was silently missing.

Moving span creation into a `traced` middleware that wraps the logger fixes it: every
inner layer now runs inside the span. Verified end to end — an access log line carrying
`trace_id=8c5273ec…` retrieves exactly that trace from Jaeger, showing
`request 47.5ms → ratelimit 4.4ms, admission 0.0ms, cache_lookup 0.5ms, dispatch 42.1ms`.

The middleware also extracts W3C trace context from inbound headers, so the gateway
appears as a span inside a caller's trace rather than starting a disconnected one, and
skips `/healthz`, `/readyz` and `/metrics`, which are scraped constantly and would bury
real traffic.

### D39 — Custom latency buckets

Prometheus's default histogram buckets top out at 10 seconds. Inference spans roughly
1ms (cache hit) to 60s+ (long generation), so with the defaults every slow generation
lands in `+Inf` and p99 becomes a straight line at the top bucket — precisely the metric
you need during an incident. Buckets here run 1ms → 120s.

### D40 — Metric series are pre-created at startup

Prometheus does not export a metric family that has never been observed. A freshly
started gateway therefore reports *nothing* for retries, cache misses or per-backend load
until the first one occurs, so dashboards show "No data" instead of zero and an alert on
`rate(waypoint_retries_total[5m])` cannot distinguish "no retries" from "metric does not
exist". `Metrics.InitSeries` pre-creates every series whose labels are known at startup.

### D41 — Gauges are sampled on a ticker, not written per request

Per-backend load, health and breaker state describe *current state*, not events. Writing
them on every request would add metric updates to the hot path to produce values a scrape
reads once per interval anyway. A 1-second collector samples them instead. Counters and
histograms, which describe events, are still written inline.

### D42 — Breaker state is encoded as an ordered scale

Gauges must be numeric but breaker state is categorical. Encoding `closed=0 < half_open=1
< open=2` rather than arbitrary codes means `max_over_time()` and threshold alerts behave
sensibly — `waypoint_breaker_state > 0` is "something is wrong" without enumerating
states.

### D43 — Label cardinality is bounded deliberately

Backend names, routing policy, model IDs and status codes are labels (small, bounded
sets). API keys, prompt hashes, request IDs and raw error strings are **not** — each
distinct value is a separate time series, and unbounded labels are the standard way to
take down a Prometheus. Upstream errors are therefore labeled by a classified `kind`
(`unreachable` / `server_error` / `timeout` / `other`) rather than by error text. The
high-cardinality identifiers live in traces and logs, which are sampled and indexed for
exactly that.

### D44 — Tracing never blocks startup or the request path

Spans are exported by a batching processor, so collector latency never reaches a request.
If the OTLP endpoint is unset or the exporter cannot be built, the gateway falls back to
a no-op tracer and logs a warning rather than failing. **This design caught its own bug:**
an `semconv` version mismatch (v1.26.0 against a `resource.Default()` built with v1.43.0)
made `resource.Merge` fail with "conflicting Schema URL". Because tracing degrades
gracefully, the gateway started and served traffic perfectly while exporting nothing —
which is the right behavior, and also exactly why the mismatch needed an explicit test
(`TestInitTracingBuildsResource`) rather than being noticed by things breaking.
