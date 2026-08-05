# Waypoint: Distributed LLM Inference Gateway

Design specification: what the gateway does, how it is built, and how its claims are measured.

## 1. Summary

Waypoint is a stateless, horizontally scalable gateway that sits between clients and a pool of LLM model servers. It exposes an OpenAI-compatible API and adds:

- routing that knows which server already has a prompt prefix cached and how loaded each server is
- admission control and queueing
- an exact-match cache and a semantic response cache
- per-key rate limiting shared across gateway replicas
- circuit breaking, retries and failover
- full observability through metrics, traces and logs

The project has to prove its value with reproducible benchmarks. Every optimization is measured against a baseline under a realistic workload.

## 2. Goals

- A production-quality Go service with clean interfaces, real tests and CI.
- Pluggable routing policies whose impact on latency and throughput is measured.
- Graceful behavior under failure: a backend dies, gets slow or starts erroring.
- A one-command local setup (`make up`) that runs on a laptop with no GPU.
- A README and design doc that read like an internal engineering design review.

## 3. Non-goals

- Training or fine-tuning models.
- Implementing a real model server. Continuous batching and KV caching happen inside the backend. The gateway's job is to route in a way that takes advantage of them.
- Multi-region deployment, auth beyond API keys, or a billing system.
- A frontend UI. Grafana is the UI.

## 4. Tech stack

| Area | Choice |
|---|---|
| Gateway | Go 1.22+, standard library net/http, log/slog |
| Shared state | Redis Stack (rate limits, exact-match cache, vector index for semantic cache) |
| Embeddings sidecar | Python 3.11, FastAPI, sentence-transformers (all-MiniLM-L6-v2), CPU only |
| Mock backend | Go (in this repo) |
| Real backends (optional) | Ollama locally, vLLM on a cloud GPU, any OpenAI-compatible API |
| Load balancer in front of replicas | Envoy or nginx |
| Metrics / tracing / dashboards | Prometheus, OpenTelemetry to Jaeger, Grafana (provisioned dashboards) |
| Fault injection | Toxiproxy |
| Load testing | k6 |
| Packaging | Docker, Docker Compose; Kubernetes manifests as a stretch goal |
| CI | GitHub Actions: go test -race, golangci-lint, integration tests, build images |

Keep Go dependencies minimal. Allowed: Redis client, OpenTelemetry SDK, Prometheus client, a YAML config parser, testify. Explain any other dependency in DESIGN.md.

## 5. Architecture

```
                 ┌──────────────┐
 clients ──────► │ Envoy / LB   │
                 └──────┬───────┘
              ┌─────────┴─────────┐
        ┌─────▼─────┐       ┌─────▼─────┐
        │ Gateway 1 │       │ Gateway 2 │   (stateless, identical)
        └─────┬─────┘       └─────┬─────┘
              │   Redis Stack (limits, caches, vectors)
              │   Embeddings sidecar
              │
     ┌────────┼──────────┬───────────┐
     ▼        ▼          ▼           ▼
  mock-1   mock-2     mock-3    (optional real backends)
     ▲ Toxiproxy sits in front of each backend for fault injection
```

How a request moves through the gateway:

1. **Ingress**: parse the request, authenticate the API key, start a trace span.
2. **Rate limit**: check the token bucket in Redis, keyed by API key, counting both requests and estimated tokens. Return 429 with Retry-After if the key is over its limit.
3. **Cache lookup**: exact-match (hash of the normalized request) first, then semantic (cosine similarity at or above the threshold, same model, same parameters, temperature == 0 only).
4. **Admission control**: a bounded priority queue. When the queue is full, shed load with 503.
5. **Routing**: the policy picks a backend from the healthy set.
6. **Dispatch**: proxy to the backend and stream the response back to the client as SSE.
7. **Resilience**: retry another backend only if no tokens have been streamed yet. Use per-backend circuit breakers.
8. **Egress**: write the cache entry asynchronously, record metrics and close the span.

## 6. Components

### 6.1 API (OpenAI-compatible)

- `POST /v1/chat/completions`: supports `stream: true` (SSE) and `stream: false`.
- `GET /v1/models`: union of models available across healthy backends.
- `GET /healthz` (liveness), `GET /readyz` (has at least one healthy backend and can reach Redis).
- `GET /metrics`: Prometheus metrics.
- `GET /admin/backends`: status, breaker state, load and prefix-cache stats for each backend.
- Errors use the OpenAI error JSON shape.

### 6.2 Mock backend (cmd/mockbackend)

This is what lets the routing gains show up in benchmarks without a GPU. It must simulate:

- Prefill and decode costs: time to first token (TTFT) grows with uncached prompt tokens; decode runs at a configurable tokens per second.
- Prefix KV cache: an LRU of prompt-prefix hashes in fixed-size token blocks, with configurable capacity. Cached prefix tokens skip prefill cost.
- Continuous batching: a configurable maximum batch size. Per-token decode latency rises with the current batch size, following a configurable curve.
- Streaming: OpenAI-format SSE chunks with deterministic text (seeded), so responses are reproducible.
- Fault knobs through flags or an admin endpoint: error rate, extra latency, a hang mode, and a crash.
- `/stats` endpoint: cache hit rate, current batch size, queue depth.

All parameters live in YAML so benchmark runs are reproducible.

### 6.3 Routing policies (internal/router)

All policies implement one interface:

```go
type Policy interface {
    Pick(ctx context.Context, req *Request, backends []*Backend) (*Backend, error)
}
```

Implement these four:

- `round_robin`: the baseline.
- `least_loaded`: picks the backend with the fewest outstanding estimated tokens, not requests.
- `prefix_affinity`: consistent hashing with bounded loads on the hash of the first N tokens of the prompt (system prompt plus earliest turns), so shared prefixes land on the backend that already has them cached.
- `prefix_affinity_load_aware`: like prefix_affinity, but falls back to least_loaded when the chosen backend's load is more than a factor c above the pool average. c is configurable.

The policy is selected by config and can be changed at runtime through /admin.

Token estimation should be a fast approximation: a byte-length heuristic, or a real tokenizer behind an interface. Document the tradeoff.

### 6.4 Resilience (internal/resilience)

- Per-backend circuit breaker with closed, open and half-open states. It trips on the error rate or the timeout rate over a rolling window.
- Health checking: active (periodic probes) plus passive (outcomes of real requests).
- Retries: exponential backoff with full jitter, capped attempts, and a retry budget (retries may be at most X% of requests) so retries can't snowball into a retry storm. Never retry after the first streamed token.
- Timeouts: separate deadlines for connecting, TTFT and the whole request, all propagated through context.
- Stretch goal: hedged requests for non-streaming calls (send to a second backend if the first one passes p95 latency).

### 6.5 Rate limiting (internal/ratelimit)

- A token bucket implemented as an atomic Redis Lua script, so the limit holds across gateway replicas.
- Two limits per key: requests per minute and tokens per minute.
- If Redis is down, configurable behavior: fail open (default) or fail closed. Document why.

### 6.6 Caching (internal/cache)

- Exact-match cache: the key is the SHA-256 of normalized `{model, messages, params}`, stored in Redis with a TTL.
- Semantic cache: embed the final user message plus a hash of the conversation context through the sidecar, then run a KNN search in a Redis vector index.
  - Only allowed when `temperature == 0` and the model and parameters match.
  - The similarity threshold is configurable. Benchmarks report the tradeoff between hit rate and answer quality at several thresholds.
- Cached streaming responses are replayed as SSE.
- Caching can be bypassed per request with an `X-Waypoint-Cache: bypass` header.

### 6.7 Observability (internal/telemetry)

Prometheus metrics, at minimum:

- `waypoint_requests_total{status, backend, policy, cache}`
- `waypoint_ttft_seconds` (histogram), `waypoint_request_duration_seconds` (histogram)
- `waypoint_tokens_per_second` (histogram)
- `waypoint_queue_depth`, `waypoint_backend_inflight_tokens{backend}`
- `waypoint_cache_hits_total{type}`, `waypoint_cache_misses_total{type}`
- `waypoint_breaker_state{backend}`, `waypoint_retries_total{reason}`
- `waypoint_ratelimit_rejections_total`

Tracing: OpenTelemetry spans for each stage in Section 5, exported to Jaeger. Pass trace context through to the backends.

Logging: structured JSON through slog, including the trace ID.

Grafana: a provisioned dashboard showing RPS, p50/p95/p99 TTFT and end-to-end latency, error rate, cache hit rate, per-backend load and breaker states.

## 7. Configuration

One YAML file, overridable with env vars. Example:

```yaml
server: { port: 8080, max_queue: 512 }
routing: { policy: prefix_affinity_load_aware, prefix_tokens: 256, load_factor: 1.25 }
backends:
  - { name: mock-1, url: http://toxiproxy:9001, max_inflight_tokens: 32000 }
  - { name: mock-2, url: http://toxiproxy:9002, max_inflight_tokens: 32000 }
  - { name: mock-3, url: http://toxiproxy:9003, max_inflight_tokens: 32000 }
resilience: { retry_budget_pct: 10, max_attempts: 3, ttft_timeout: 5s, request_timeout: 120s }
ratelimit: { rpm: 600, tpm: 200000, fail_open: true }
cache: { exact_ttl: 1h, semantic: { enabled: true, threshold: 0.92 } }
```

## 8. Benchmarks (bench/)

Every performance claim has to come from a script that anyone can rerun.

Workload generator (k6 plus a Go or Python prompt generator):

- K system prompts with Zipf-distributed popularity.
- Multi-turn conversations (2 to 8 turns) that reuse prefixes.
- Input and output lengths drawn from configurable distributions.
- A configurable share of repeated or near-duplicate questions (for the caches).
- Open-loop load at fixed arrival rates (constant RPS), not closed-loop.

Experiments (each one writes a JSON result, and a script turns the results into Markdown tables and PNG charts):

- **Routing**: all four policies at 3 or more RPS levels. Report p50/p95/p99 TTFT, end-to-end latency, throughput, backend prefix-cache hit rate and load imbalance.
- **Caching**: off, exact only, and exact plus semantic at 3 thresholds. Report hit rate, latency and a quality proxy (similarity of the cached answer to the fresh one on a labeled set).
- **Resilience**: use Toxiproxy to (a) kill one backend partway through a run, (b) add 2s of latency to one backend, (c) make one backend return 30% errors. Report the success rate, the latency impact and the time until the breaker trips and later recovers.
- **Horizontal scaling**: 1 vs 2 vs 4 gateway replicas. Show rate limits holding across replicas and the gateway's own overhead (p99 added latency versus calling a backend directly).

Results are committed to `bench/results/` with the hardware and config recorded. Never write a number into the README that a script didn't produce.

## 9. Repo layout

```
cmd/gateway/          cmd/mockbackend/
internal/api/         internal/router/      internal/resilience/
internal/ratelimit/   internal/cache/       internal/telemetry/
internal/config/      internal/backend/
sidecar/embeddings/   deploy/compose/       deploy/k8s/ (stretch)
deploy/grafana/       bench/                docs/
.github/workflows/    Makefile   README.md  DESIGN.md  SPEC.md
```
