# Waypoint

A stateless, horizontally scalable gateway that sits between clients and a pool of LLM
model servers. It speaks the OpenAI API and adds prefix-aware routing, admission control,
response caching, distributed rate limiting, circuit breaking and full observability.

Every performance claim below comes from a script in [`bench/`](bench/) that anyone can
re-run. Nothing is hand-entered.

---

## The problem

A model server — vLLM, TGI, Ollama — is very good at one thing: turning batches of token
sequences into output tokens. It is not in the business of knowing which tenant is over
quota, which replica already holds a prompt's KV cache, or what to do when a peer starts
timing out. Put several of them behind a general-purpose load balancer and you get two
specific failures:

**Round robin destroys cache locality.** LLM requests are *not* interchangeable across
servers. A server that already holds your prompt's prefix in its KV cache skips prefill
entirely — in this repo's simulation, a 23x difference in time-to-first-token (235 ms
cold vs 10 ms warm). Round robin maximizes the chance a prompt lands somewhere that has
never seen it.

**Request counts are a terrible proxy for load.** One 32k-token prompt occupies far more
KV memory and decode time than five 100-token prompts, but a connection-counting balancer
ranks them the other way around.

Waypoint is the control plane for that fleet: it places requests well, stops bad ones
early, and degrades predictably when a backend misbehaves.

---

## Architecture

```
                      ┌──────────────┐
   clients ─────────► │  Envoy / LB  │
                      └──────┬───────┘
              ┌──────────────┴──────────────┐
        ┌─────▼──────┐               ┌──────▼─────┐
        │ Gateway 1  │               │ Gateway 2  │   stateless, identical
        └─────┬──────┘               └──────┬─────┘
              │                             │
              └──────────┬──────────────────┘
                         │
          ┌──────────────┼───────────────┐
          │              │               │
    ┌─────▼─────┐  ┌─────▼──────┐  ┌─────▼──────────┐
    │   Redis   │  │ Embeddings │  │   Prometheus   │
    │  limits,  │  │  sidecar   │  │ Jaeger, Grafana│
    │  caches,  │  │ (MiniLM)   │  └────────────────┘
    │  vectors  │  └────────────┘
    └───────────┘
                         │
          ┌──────────────┼───────────────┐
          ▼              ▼               ▼
      ┌────────┐    ┌────────┐     ┌────────┐
      │ mock-1 │    │ mock-2 │     │ mock-3 │   model servers
      └────────┘    └────────┘     └────────┘
          ▲ Toxiproxy fronts each one for fault injection
```

Each request moves through these stages, and each is a span in the trace:

| Stage | What it does | Fails with |
|---|---|---|
| **Ingress** | Parse, authenticate the API key, start the root span | `400`, `401` |
| **Rate limit** | Atomic Redis token bucket on requests *and* tokens | `429` + `Retry-After` |
| **Cache** | Exact-match hash, then semantic vector search | — (hit returns immediately) |
| **Admission** | Bounded aging priority queue | `503` + `Retry-After` |
| **Routing** | Policy picks from the eligible backend set | `503` / `404` |
| **Dispatch** | Stream SSE through, retry on failure *before* first token | `502`, `504` |
| **Egress** | Cache asynchronously, record metrics, close the span | — |

Design decisions, alternatives considered, and the bugs found along the way are in
**[DESIGN.md](DESIGN.md)**.

---

## Quickstart

```bash
make up          # Redis, 3 mock backends behind Toxiproxy, gateway, Prometheus, Jaeger, Grafana
make smoke       # one streaming request through the gateway
```

| Service | URL |
|---|---|
| Gateway | http://localhost:8080 |
| Grafana | http://localhost:3000 (admin/admin) |
| Jaeger | http://localhost:16686 |
| Prometheus | http://localhost:9090 |

```bash
make test          # unit tests with the race detector
make integration   # Compose-backed integration suite
make bench-routing # routing policy comparison
make bench-report  # regenerate tables and charts from results
```

The mock backend exists so routing and resilience work can be measured **without a GPU**.
It simulates prefill cost proportional to uncached prompt tokens, a block-level KV prefix
cache with LRU eviction, and continuous batching whose per-token cost rises with batch
size. Real backends (Ollama, vLLM, any OpenAI-compatible server) are supported by pointing
`backends[].url` at them.

---

## Headline results

Measured on an Apple M-series laptop, 3 simulated backends, 30-second open-loop runs.
Full tables, charts and the environment capture: **[bench/results/](bench/results/)**.

### Routing: prefix affinity vs round robin

At 20 RPS, with KV cache capacity scarce relative to the prompt working set (the
production condition — KV cache is GPU memory):

| Policy | Prefix hit rate | TTFT p50 | Load imbalance |
|---|---:|---:|---:|
| `round_robin` | 18.5% | 55.4 ms | 1.00x |
| `prefix_affinity` | **29.2%** | **49.1 ms** | 1.23x |

**+57% relative prefix-cache hit rate and 11% lower p50 TTFT**, at the cost of load
imbalance rising from 1.00x to 1.23x. That tradeoff is the entire point of
`prefix_affinity_load_aware`, which caps imbalance by abandoning affinity when a backend
exceeds `c x` the pool average.

### Caching

30% verbatim repeats, 25% reworded repeats, `temperature=0`:

| Configuration | Hit rate | E2E p50 |
|---|---:|---:|
| off | 0% | 592 ms |
| exact only | 52.2% | **4 ms** |
| exact + semantic | **~83%** | 16 ms |

The exact cache turns a 592 ms median into 4 ms. The semantic layer adds the ~25% of
traffic that is reworded rather than repeated, at the cost of an embedding round trip.
p95/p99 barely move in either case, which is expected — the tail is made of cache
*misses*, which still pay full generation cost.

### Resilience

One of three backends faulted for the whole run:

| Scenario | Availability | TTFT p99 |
|---|---:|---:|
| baseline | 100% | 74 ms |
| backend hangs | **100%** | 5,068 ms |
| backend +2s latency | **100%** | 2,072 ms |
| backend 30% errors | 94% | 81 ms |

A hung or slow backend costs **no availability** — requests fail their TTFT deadline and
retry on a healthy peer, with tail latency bounded by that deadline. Under a 30% error
rate availability is 94% rather than 100%, and that is the **retry budget working as
designed**: roughly 10% of traffic fails its first attempt, which is exactly the
configured 10% budget, so some requests find no budget left and fail rather than
amplifying load on a struggling fleet.

### Gateway overhead and shared limits

| Measurement | Result |
|---|---|
| Latency added by the gateway (p50, unsaturated) | **+1.5 ms** |
| Latency added at p95/p99 | within run-to-run noise |
| Requests allowed at 1 / 2 / 4 replicas (shared RPM 300) | **424 / 424 / 424** |

A per-process rate limiter would have admitted roughly 4x the configured rate at 4
replicas. The limit holds because check-and-decrement runs as an atomic Redis Lua script
rather than as read-modify-write in the application.

---

## What's in here

| Package | Responsibility |
|---|---|
| `internal/api` | HTTP surface, streaming proxy, dispatch with retries, response capture |
| `internal/router` | Four routing policies; consistent hashing with bounded loads |
| `internal/resilience` | Circuit breakers, health checking, retry budget, backoff |
| `internal/ratelimit` | Redis Lua token bucket (requests *and* tokens) |
| `internal/cache` | Exact-match and semantic caches, embeddings client |
| `internal/admission` | Bounded aging priority queue |
| `internal/telemetry` | Prometheus metrics, OTel tracing, trace-correlated logging |
| `internal/mocksim` | Backend simulator: prefill, prefix cache, continuous batching, faults |
| `internal/sse` | Server-Sent Events codec (fuzzed) |
| `bench/` | k6 workload, experiment runner, report generator |

Test coverage on `internal/` is 83%, including fuzz targets for request and SSE parsing,
race-detector runs on every package, and a Toxiproxy-backed integration suite that
injects transport-level faults.

---

## Documentation

- **[DESIGN.md](DESIGN.md)** — every nontrivial decision, the alternatives rejected, and
  the bugs that changed the design. Written to be read in a design review.
- **[bench/results/](bench/results/)** — generated tables, charts and raw JSON.
- **[SPEC.md](SPEC.md)** — the design specification: requirements, milestones and benchmark plan.

---

## Status and known gaps

Milestones M0–M9 of [SPEC.md](SPEC.md) are complete. Known gaps:

- **No recording of the Grafana dashboard during a failover yet.** The dashboard is
  provisioned and verified ([`deploy/grafana/dashboards/waypoint.json`](deploy/grafana/dashboards/waypoint.json),
  18 panels), and metrics/traces are verified end to end.
- **Results come from a simulated backend.** The *relative* claims should transfer —
  policies are exploiting prefill cost, batch pressure and prefix caching, all of which
  real engines exhibit — but absolute latencies depend on GPU, quantization and
  tensor-parallel degree, none of which are simulated. The cost model is committed in
  [`configs/mockbackend.yaml`](configs/mockbackend.yaml) so a reader can judge whether
  its assumptions resemble their hardware.
- **Benchmarks are single-run.** The semantic-threshold sweep in particular showed
  run-to-run variance (52%–84% hit rate for the same configuration), documented in the
  results rather than smoothed over. Multiple repetitions with confidence intervals would
  be the next improvement.
- **M10 (stretch) is not done**: no Kubernetes manifests, no hedged requests, and real
  backends are supported but untested here.
