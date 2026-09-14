#!/usr/bin/env bash
#
# Waypoint benchmark runner.
#
# Each experiment starts a clean stack, drives it with the k6 workload, and
# writes a JSON result. Everything that could change a number — config,
# hardware, git SHA — is recorded alongside the result so a run can be audited
# or reproduced later.
#
# Usage:
#   bench/run.sh routing     # compare all four routing policies
#   bench/run.sh caching     # cache off / exact / exact+semantic at 3 thresholds
#   bench/run.sh resilience  # backend kill, slow backend, error injection
#   bench/run.sh scaling     # 1 vs 2 vs 4 gateway replicas
#   bench/run.sh all

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

BIN_DIR="$REPO_ROOT/bin"
RESULTS_DIR="$REPO_ROOT/bench/results"
RAW_DIR="$RESULTS_DIR/raw"
TMP_DIR="${TMPDIR:-/tmp}/waypoint-bench-$$"

GATEWAY_PORT="${GATEWAY_PORT:-18080}"
BACKEND_BASE_PORT="${BACKEND_BASE_PORT:-19001}"
BACKEND_COUNT="${BACKEND_COUNT:-3}"
REDIS_ADDR="${REDIS_ADDR:-localhost:6379}"
API_KEY="bench-key"

# Workload shape. Kept modest so a full sweep finishes on a laptop; raise
# DURATION for tighter percentiles.
DURATION="${DURATION:-30s}"
RPS_LEVELS="${RPS_LEVELS:-10 20 40}"

# Working-set and capacity shape, which is what makes routing policy matter.
#
# KV cache capacity must be SCARCE relative to the prompt working set, because
# that is the production condition — KV cache is GPU memory. If every backend
# can hold every prefix, affinity routing buys nothing and all four policies
# measure identically: an early run of this benchmark showed exactly that, with
# all policies landing within 1ms of each other at a 0.6 hit rate.
#
# 16 system prompts x ~22 blocks each is a ~350-block working set; 64 blocks per
# backend means a backend holds roughly 3 of them, so where a prompt is routed
# determines whether it hits.
SYSTEM_PROMPTS="${SYSTEM_PROMPTS:-16}"
PREFIX_CAPACITY_BLOCKS="${PREFIX_CAPACITY_BLOCKS:-24}"
# Must stay below the system-prompt length so the affinity key is stable.
PREFIX_TOKENS="${PREFIX_TOKENS:-128}"

mkdir -p "$RESULTS_DIR" "$RAW_DIR" "$TMP_DIR"

log()  { printf '\033[36m[bench]\033[0m %s\n' "$*" >&2; }
warn() { printf '\033[33m[bench]\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[31m[bench]\033[0m %s\n' "$*" >&2; exit 1; }

# ---------- prerequisites ----------

require() {
  command -v "$1" >/dev/null 2>&1 || die "$1 is required but not installed"
}

require k6
require go
require curl
require python3

redis_available() {
  (exec 3<>/dev/tcp/${REDIS_ADDR%%:*}/${REDIS_ADDR##*:}) 2>/dev/null && exec 3>&- && return 0
  return 1
}

# ---------- build ----------

build() {
  log "building binaries"
  mkdir -p "$BIN_DIR"
  go build -o "$BIN_DIR/gateway" ./cmd/gateway
  go build -o "$BIN_DIR/mockbackend" ./cmd/mockbackend
}

# ---------- stack lifecycle ----------

BACKEND_PIDS=()
GATEWAY_PIDS=()

start_backends() {
  log "starting $BACKEND_COUNT mock backends"
  for i in $(seq 1 "$BACKEND_COUNT"); do
    local port=$((BACKEND_BASE_PORT + i - 1))
    MOCKBACKEND_ADDR=":$port" \
    MOCKBACKEND_NAME="mock-$i" \
    MOCKBACKEND_SEED="$i" \
    MOCKBACKEND_PREFIX_CAPACITY_BLOCKS="$PREFIX_CAPACITY_BLOCKS" \
      "$BIN_DIR/mockbackend" -config configs/mockbackend.yaml \
      >"$TMP_DIR/mock-$i.log" 2>&1 &
    BACKEND_PIDS+=($!)
  done

  for i in $(seq 1 "$BACKEND_COUNT"); do
    local port=$((BACKEND_BASE_PORT + i - 1))
    wait_for "http://localhost:$port/healthz" "mock-$i"
  done
}

# write_gateway_config emits a config for one gateway replica.
write_gateway_config() {
  local path="$1" port="$2" policy="$3" cache_enabled="$4" semantic="$5" threshold="$6" ratelimit="$7"

  {
    echo "server: { port: $port, max_queue: 1024, max_concurrent: 256 }"
    # prefix_tokens must cover the SHARED span and stop before the part that
    # varies per request. The benchmark's system prompts are ~169 tokens, so a
    # 256-token budget reaches into the first user question and produces a
    # different affinity key for every request — which silently disables
    # affinity routing while looking correctly configured.
    echo "routing: { policy: $policy, prefix_tokens: $PREFIX_TOKENS, load_factor: 1.25, virtual_nodes: 150 }"
    echo "backends:"
    for i in $(seq 1 "$BACKEND_COUNT"); do
      local bport=$((BACKEND_BASE_PORT + i - 1))
      echo "  - { name: mock-$i, url: \"http://localhost:$bport\", max_inflight_tokens: 64000 }"
    done
    echo "resilience: { retry_budget_pct: 10, max_attempts: 3, ttft_timeout: 5s, request_timeout: 120s }"
    echo "ratelimit: { enabled: $ratelimit, rpm: 100000, tpm: 100000000, fail_open: true }"
    echo "cache: { enabled: $cache_enabled, exact_ttl: 1h, semantic: { enabled: $semantic, threshold: $threshold, embedder_url: \"http://localhost:8000\", dimensions: 384 } }"
    echo "redis: { addr: \"$REDIS_ADDR\", key_prefix: \"bench-$(date +%s)\" }"
    echo "telemetry: { log_level: warn, otlp_endpoint: \"\" }"
    echo "auth: { api_keys: [\"$API_KEY\"] }"
  } >"$path"
}

start_gateway() {
  local cfg="$1" port="$2" label="${3:-gw}"
  "$BIN_DIR/gateway" -config "$cfg" >"$TMP_DIR/$label.log" 2>&1 &
  GATEWAY_PIDS+=($!)
  wait_for "http://localhost:$port/healthz" "$label"
}

wait_for() {
  local url="$1" name="$2"
  for _ in $(seq 1 100); do
    if curl -fsS "$url" >/dev/null 2>&1; then return 0; fi
    sleep 0.1
  done
  die "$name never became healthy at $url"
}

stop_stack() {
  for pid in "${GATEWAY_PIDS[@]:-}"; do kill "$pid" 2>/dev/null || true; done
  for pid in "${BACKEND_PIDS[@]:-}"; do kill "$pid" 2>/dev/null || true; done
  GATEWAY_PIDS=()
  BACKEND_PIDS=()
  wait 2>/dev/null || true
}

cleanup() {
  stop_stack
  rm -rf "$TMP_DIR"
}
trap cleanup EXIT INT TERM

# reset_backends clears prefix caches and counters so each run starts cold.
# Without this, the first policy measured would warm the caches for the rest and
# every later policy would look artificially good.
reset_backends() {
  for i in $(seq 1 "$BACKEND_COUNT"); do
    local port=$((BACKEND_BASE_PORT + i - 1))
    curl -fsS -X POST "http://localhost:$port/admin/reset" >/dev/null 2>&1 || true
  done
}

backend_stats_json() {
  local out="["
  for i in $(seq 1 "$BACKEND_COUNT"); do
    local port=$((BACKEND_BASE_PORT + i - 1))
    local body
    body="$(curl -fsS "http://localhost:$port/stats" 2>/dev/null || echo '{}')"
    out+="$body"
    [ "$i" -lt "$BACKEND_COUNT" ] && out+=","
  done
  echo "$out]"
}

# sample_breakers appends a timestamped breaker snapshot every 500ms.
sample_breakers() {
  local out="$1"
  local start
  start=$(python3 -c 'import time; print(time.time())')

  while true; do
    local now elapsed body
    now=$(python3 -c 'import time; print(time.time())')
    elapsed=$(python3 -c "print(round($now - $start, 2))")
    body="$(curl -fsS "http://localhost:$GATEWAY_PORT/admin/backends" 2>/dev/null || echo '{}')"
    python3 - "$elapsed" "$body" >>"$out" <<'INNER'
import json, sys
elapsed, body = sys.argv[1], sys.argv[2]
try:
    admin = json.loads(body)
except Exception:
    admin = {}
states = {k: v.get("state") for k, v in (admin.get("breakers") or {}).items()}
print(json.dumps({"t": float(elapsed), "states": states}))
INNER
    sleep 0.5
  done
}

set_policy() {
  curl -fsS -X POST "http://localhost:$GATEWAY_PORT/admin/routing" \
    -H 'Content-Type: application/json' \
    -d "{\"policy\":\"$1\"}" >/dev/null
}

# ---------- environment capture ----------

env_json() {
  python3 - "$REPO_ROOT" <<'PY'
import json, platform, subprocess, sys, os

repo = sys.argv[1]

def run(cmd):
    try:
        return subprocess.check_output(cmd, shell=True, stderr=subprocess.DEVNULL,
                                       text=True, cwd=repo).strip()
    except Exception:
        return "unknown"

cpu = run("sysctl -n machdep.cpu.brand_string") if platform.system() == "Darwin" else run("lscpu | grep 'Model name' | cut -d: -f2")
cores = os.cpu_count()
mem = run("sysctl -n hw.memsize") if platform.system() == "Darwin" else run("grep MemTotal /proc/meminfo")

print(json.dumps({
    "os": platform.platform(),
    "arch": platform.machine(),
    "cpu": cpu.strip(),
    "cores": cores,
    "memory_bytes": mem if not mem.isdigit() else int(mem),
    "go_version": run("go version"),
    "k6_version": run("k6 version"),
    "git_sha": run("git rev-parse --short HEAD"),
    "git_dirty": run("git status --porcelain") != "",
}, indent=2))
PY
}

# ---------- k6 invocation ----------

# run_k6 <label> [extra env assignments...]
run_k6() {
  local label="$1"; shift
  log "  k6: $label"

  env \
    GATEWAY="http://localhost:$GATEWAY_PORT" \
    API_KEY="$API_KEY" \
    DURATION="$DURATION" \
    RESULT_LABEL="$label" \
    "$@" \
    k6 run --quiet --no-usage-report bench/workload.js >"$TMP_DIR/$label.out" 2>&1 || {
      warn "k6 run $label exited non-zero; see $TMP_DIR/$label.out"
    }

  cat "$TMP_DIR/$label.out" | tail -14 >&2
}

# ---------- experiment: routing ----------

experiment_routing() {
  log "EXPERIMENT: routing policies"
  build
  start_backends

  local cfg="$TMP_DIR/gw-routing.yaml"
  write_gateway_config "$cfg" "$GATEWAY_PORT" "round_robin" "false" "false" "0.92" "false"
  start_gateway "$cfg" "$GATEWAY_PORT" "gw-routing"

  local results="$TMP_DIR/routing-results.json"
  echo "[" >"$results"
  local first=1

  for policy in round_robin least_loaded prefix_affinity prefix_affinity_load_aware; do
    for rps in $RPS_LEVELS; do
      log " policy=$policy rps=$rps"

      set_policy "$policy"
      # Cold caches for every cell, or earlier runs warm later ones.
      reset_backends
      sleep 1

      local label="routing_${policy}_${rps}"
      run_k6 "$label" RPS="$rps" STREAM=true DUPLICATE_RATE=0 MAX_TOKENS=64 \
        SYSTEM_PROMPTS="$SYSTEM_PROMPTS"

      local stats
      backend_stats_json >"$TMP_DIR/${label}-stats.json"

      [ $first -eq 0 ] && echo "," >>"$results"
      first=0
      python3 - "$RAW_DIR/$label.json" "$policy" "$rps" "$stats" >>"$results" <<'PY'
import json, sys
raw_path, policy, rps, stats_json = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]

try:
    with open(raw_path) as f:
        data = json.load(f)
except Exception as exc:
    print(json.dumps({"policy": policy, "rps": rps, "error": str(exc)}))
    sys.exit()

m = data.get("metrics", {})
def v(name, field, default=None):
    return m.get(name, {}).get("values", {}).get(field, default)

stats = json.loads(stats_json)
hits = sum(s.get("prefix_cache", {}).get("block_hits", 0) for s in stats)
lookups = sum(s.get("prefix_cache", {}).get("block_lookups", 0) for s in stats)
per_backend = {s.get("name", f"b{i}"): s.get("requests", 0) for i, s in enumerate(stats)}

counts = list(per_backend.values()) or [0]
mean = sum(counts) / len(counts) if counts else 0
imbalance = (max(counts) / mean) if mean else 0

print(json.dumps({
    "policy": policy,
    "rps": rps,
    "requests": v("http_reqs", "count", 0),
    "success_rate": v("waypoint_success", "rate", 0),
    "ttft_p50_ms": v("waypoint_ttft", "p(50)"),
    "ttft_p95_ms": v("waypoint_ttft", "p(95)"),
    "ttft_p99_ms": v("waypoint_ttft", "p(99)"),
    "e2e_p50_ms": v("waypoint_e2e", "p(50)"),
    "e2e_p95_ms": v("waypoint_e2e", "p(95)"),
    "e2e_p99_ms": v("waypoint_e2e", "p(99)"),
    "throughput_rps": v("http_reqs", "rate", 0),
    "prefix_cache_hit_rate": (hits / lookups) if lookups else 0,
    "requests_per_backend": per_backend,
    "load_imbalance_max_over_mean": imbalance,
}, indent=2))
PY
    done
  done

  echo "]" >>"$results"
  finalize "routing" "$results"
  stop_stack
}

# ---------- experiment: caching ----------

experiment_caching() {
  log "EXPERIMENT: caching"
  if ! redis_available; then
    warn "Redis not reachable at $REDIS_ADDR; skipping caching experiment"
    warn "  start it with: docker run -d -p 6379:6379 redis/redis-stack-server:7.4.0-v1"
    return
  fi

  build
  start_backends

  local results="$TMP_DIR/caching-results.json"
  echo "[" >"$results"
  local first=1

  # Configurations: cache off, exact only, then exact+semantic at 3 thresholds.
  # Each needs its own gateway because cache wiring is set at construction.
  local configs=(
    "off:false:false:0.92"
    "exact:true:false:0.92"
    "semantic_0.80:true:true:0.80"
    "semantic_0.90:true:true:0.90"
    "semantic_0.95:true:true:0.95"
  )

  for entry in "${configs[@]}"; do
    IFS=':' read -r name cache_enabled semantic threshold <<<"$entry"

    # Semantic configurations need the embeddings sidecar.
    if [ "$semantic" = "true" ] && ! curl -fsS http://localhost:8000/healthz >/dev/null 2>&1; then
      warn "embeddings sidecar not reachable; skipping $name"
      continue
    fi

    log " cache=$name"
    local cfg="$TMP_DIR/gw-cache-$name.yaml"
    write_gateway_config "$cfg" "$GATEWAY_PORT" "prefix_affinity_load_aware" \
      "$cache_enabled" "$semantic" "$threshold" "false"

    start_gateway "$cfg" "$GATEWAY_PORT" "gw-cache-$name"
    reset_backends
    sleep 1

    local label="caching_${name}"
    # 30% verbatim repeats and 25% rewordings: the exact cache can serve the
    # first group, only the semantic cache can serve the second.
    run_k6 "$label" RPS=20 STREAM=false TEMPERATURE=0 \
      DUPLICATE_RATE=0.30 NEAR_DUPLICATE_RATE=0.25 MAX_TOKENS=48

    [ $first -eq 0 ] && echo "," >>"$results"
    first=0
    python3 - "$RAW_DIR/$label.json" "$name" "$threshold" >>"$results" <<'PY'
import json, sys
raw_path, name, threshold = sys.argv[1], sys.argv[2], sys.argv[3]

try:
    with open(raw_path) as f:
        data = json.load(f)
except Exception as exc:
    print(json.dumps({"config": name, "error": str(exc)}))
    sys.exit()

m = data.get("metrics", {})
def v(n, f, d=None):
    return m.get(n, {}).get("values", {}).get(f, d)

hits = v("waypoint_cache_hits", "count", 0) or 0
misses = v("waypoint_cache_misses", "count", 0) or 0
total = hits + misses

print(json.dumps({
    "config": name,
    "threshold": float(threshold),
    "requests": v("http_reqs", "count", 0),
    "cache_hits": hits,
    "cache_misses": misses,
    "hit_rate": (hits / total) if total else 0,
    "ttft_p50_ms": v("waypoint_ttft", "p(50)"),
    "ttft_p95_ms": v("waypoint_ttft", "p(95)"),
    "e2e_p50_ms": v("waypoint_e2e", "p(50)"),
    "e2e_p95_ms": v("waypoint_e2e", "p(95)"),
    "e2e_p99_ms": v("waypoint_e2e", "p(99)"),
    "success_rate": v("waypoint_success", "rate", 0),
}, indent=2))
PY

    stop_stack
    start_backends
  done

  echo "]" >>"$results"
  finalize "caching" "$results"
  stop_stack
}

# ---------- experiment: resilience ----------

experiment_resilience() {
  log "EXPERIMENT: resilience"
  build
  start_backends

  local cfg="$TMP_DIR/gw-resilience.yaml"
  write_gateway_config "$cfg" "$GATEWAY_PORT" "prefix_affinity_load_aware" "false" "false" "0.92" "false"
  start_gateway "$cfg" "$GATEWAY_PORT" "gw-resilience"

  local results="$TMP_DIR/resilience-results.json"
  echo "[" >"$results"
  local first=1

  # Scenarios map to SPEC §8: baseline, backend killed, backend slowed, backend
  # returning errors. Faults are injected through the mock backend's admin API,
  # which can express application-level failures a TCP proxy cannot.
  local scenarios=("baseline" "kill" "slow" "errors")

  for scenario in "${scenarios[@]}"; do
    log " scenario=$scenario"
    reset_backends
    sleep 1

    local fault_port=$BACKEND_BASE_PORT
    case "$scenario" in
      kill)
        # Hang mode: the backend accepts connections and never responds, which
        # exercises TTFT timeouts rather than connection refusal.
        curl -fsS -X POST "http://localhost:$fault_port/admin/faults" \
          -H 'Content-Type: application/json' \
          -d '{"hang": true}' >/dev/null
        ;;
      slow)
        curl -fsS -X POST "http://localhost:$fault_port/admin/faults" \
          -H 'Content-Type: application/json' \
          -d '{"extra_latency_ms": 2000}' >/dev/null
        ;;
      errors)
        curl -fsS -X POST "http://localhost:$fault_port/admin/faults" \
          -H 'Content-Type: application/json' \
          -d '{"error_rate": 0.3}' >/dev/null
        ;;
    esac

    local label="resilience_${scenario}"

    # Breaker state must be sampled *during* the run. The rolling window is
    # 10s, so reading /admin/backends after a 30s run shows a window that has
    # already rolled over and every breaker back to closed — which is exactly
    # what an earlier version of this benchmark reported, hiding a trip that
    # genuinely happened.
    local timeline="$TMP_DIR/${label}-breakers.jsonl"
    : >"$timeline"
    sample_breakers "$timeline" &
    local sampler_pid=$!

    run_k6 "$label" RPS=20 STREAM=true MAX_TOKENS=48

    kill "$sampler_pid" 2>/dev/null || true
    wait "$sampler_pid" 2>/dev/null || true

    # Clear faults before measuring recovery.
    curl -fsS -X POST "http://localhost:$fault_port/admin/faults" \
      -H 'Content-Type: application/json' -d '{}' >/dev/null 2>&1 || true

    local breakers stats
    breakers="$(curl -fsS "http://localhost:$GATEWAY_PORT/admin/backends" 2>/dev/null || echo '{}')"
    backend_stats_json >"$TMP_DIR/${label}-stats.json"

    [ $first -eq 0 ] && echo "," >>"$results"
    first=0
    python3 - "$RAW_DIR/$label.json" "$scenario" "$breakers" "$timeline" >>"$results" <<'PY'
import json, os, sys
raw_path, scenario, breakers_json = sys.argv[1], sys.argv[2], sys.argv[3]
timeline_path = sys.argv[4] if len(sys.argv) > 4 else ""
stats_path = sys.argv[5] if len(sys.argv) > 5 else ""
try:
    with open(stats_path) as sf:
        backend_stats = json.load(sf)
except Exception:
    backend_stats = []

try:
    with open(raw_path) as f:
        data = json.load(f)
except Exception as exc:
    print(json.dumps({"scenario": scenario, "error": str(exc)}))
    sys.exit()

m = data.get("metrics", {})
def v(n, f, d=None):
    return m.get(n, {}).get("values", {}).get(f, d)

try:
    admin = json.loads(breakers_json)
except Exception:
    admin = {}

# Reconstruct breaker behavior from the during-run samples.
trip_at, recover_at, ever_open = None, None, False
breaker_timeline = []
if timeline_path and os.path.exists(timeline_path):
    with open(timeline_path) as tf:
        for line in tf:
            line = line.strip()
            if not line:
                continue
            try:
                sample = json.loads(line)
            except Exception:
                continue
            breaker_timeline.append(sample)
            opened = [b for b, st in sample["states"].items() if st in ("open", "half_open")]
            if opened and trip_at is None:
                trip_at = sample["t"]
                ever_open = True
            if trip_at is not None and not opened and recover_at is None:
                recover_at = sample["t"]

print(json.dumps({
    "scenario": scenario,
    "requests": v("http_reqs", "count", 0),
    "availability": v("waypoint_success", "rate", 0),
    "breaker_tripped": ever_open,
    "breaker_trip_at_s": trip_at,
    "breaker_recovered_at_s": recover_at,
    "breaker_samples": len(breaker_timeline),
    "ttft_p50_ms": v("waypoint_ttft", "p(50)"),
    "ttft_p95_ms": v("waypoint_ttft", "p(95)"),
    "ttft_p99_ms": v("waypoint_ttft", "p(99)"),
    "e2e_p99_ms": v("waypoint_e2e", "p(99)"),
    "upstream_errors": v("waypoint_upstream_errors", "count", 0),
    "shed_503": v("waypoint_shed", "count", 0),
    "breakers": admin.get("breakers", {}),
    "retry_budget": admin.get("retry_budget", {}),
}, indent=2))
PY
  done

  echo "]" >>"$results"
  finalize "resilience" "$results"
  stop_stack
}

# ---------- experiment: horizontal scaling ----------

experiment_scaling() {
  log "EXPERIMENT: horizontal scaling and gateway overhead"
  if ! redis_available; then
    warn "Redis not reachable; scaling experiment needs it for shared rate limits"
    return
  fi

  build
  start_backends

  local results="$TMP_DIR/scaling-results.json"
  echo "[" >"$results"
  local first=1

  # ---- part 1: gateway overhead ----
  #
  # The honest cost of putting a gateway in front of inference. Measured by
  # running the identical workload straight at one backend, then through a
  # gateway configured with routing/caching/limits off, so the delta is the
  # proxy hop alone rather than anything the gateway chose to do.
  # Overhead must be measured UNSATURATED. At high RPS against one backend the
  # measurement is dominated by queueing inside the backend, not by the proxy
  # hop — an earlier run at 20 RPS reported the gateway as *faster* than direct,
  # which simply meant both paths were saturated and the comparison was noise.
  local OVERHEAD_RPS="${OVERHEAD_RPS:-4}"

  log " measuring baseline: direct to backend (no gateway), ${OVERHEAD_RPS} rps"
  reset_backends
  sleep 1
  run_k6_direct "overhead_direct" "http://localhost:$BACKEND_BASE_PORT" "$OVERHEAD_RPS"

  local cfg="$TMP_DIR/gw-overhead.yaml"
  write_gateway_config "$cfg" "$GATEWAY_PORT" "round_robin" "false" "false" "0.92" "false"
  # Pin to a single backend so the comparison isolates proxy cost rather than
  # measuring a different placement decision.
  python3 - "$cfg" "$BACKEND_BASE_PORT" <<'INNER'
import re, sys
path, port = sys.argv[1], sys.argv[2]
text = open(path).read()
lines = [l for l in text.splitlines() if not re.match(r'\s+- \{ name: mock-', l)]
out = []
for line in lines:
    out.append(line)
    if line.startswith("backends:"):
        out.append(f'  - {{ name: mock-1, url: "http://localhost:{port}", max_inflight_tokens: 640000 }}')
open(path, "w").write("\n".join(out) + "\n")
INNER
  start_gateway "$cfg" "$GATEWAY_PORT" "gw-overhead"
  reset_backends
  sleep 1

  log " measuring through gateway (routing/cache/limits off), ${OVERHEAD_RPS} rps"
  run_k6 "overhead_gateway" RPS="$OVERHEAD_RPS" STREAM=false MAX_TOKENS=32

  [ $first -eq 0 ] && echo "," >>"$results"
  first=0
  python3 - "$RAW_DIR/overhead_direct.json" "$RAW_DIR/overhead_gateway.json" >>"$results" <<'INNER'
import json, sys

def load(path):
    try:
        with open(path) as f:
            return json.load(f).get("metrics", {})
    except Exception:
        return {}

def v(m, n, f, d=None):
    return m.get(n, {}).get("values", {}).get(f, d)

direct, gw = load(sys.argv[1]), load(sys.argv[2])

def delta(a, b):
    if a is None or b is None:
        return None
    return round(b - a, 2)

print(json.dumps({
    "measurement": "gateway_overhead",
    "direct_p50_ms": v(direct, "waypoint_e2e", "p(50)"),
    "direct_p95_ms": v(direct, "waypoint_e2e", "p(95)"),
    "direct_p99_ms": v(direct, "waypoint_e2e", "p(99)"),
    "gateway_p50_ms": v(gw, "waypoint_e2e", "p(50)"),
    "gateway_p95_ms": v(gw, "waypoint_e2e", "p(95)"),
    "gateway_p99_ms": v(gw, "waypoint_e2e", "p(99)"),
    "added_p50_ms": delta(v(direct, "waypoint_e2e", "p(50)"), v(gw, "waypoint_e2e", "p(50)")),
    "added_p95_ms": delta(v(direct, "waypoint_e2e", "p(95)"), v(gw, "waypoint_e2e", "p(95)")),
    "added_p99_ms": delta(v(direct, "waypoint_e2e", "p(99)"), v(gw, "waypoint_e2e", "p(99)")),
    "note": ("Same workload, single backend, gateway features disabled, measured "
             "unsaturated so the delta reflects the proxy hop rather than queueing."),
}, indent=2))
INNER

  stop_stack

  # ---- part 2: shared rate limits across replicas ----
  #
  # The property worth proving is not that N gateways serve N times the
  # traffic — they cannot, the backends are the bottleneck — but that one
  # tenant's limit holds no matter how many replicas it is spread across. A
  # per-process limiter would allow N x the configured rate.
  local RPM_LIMIT=300

  for replicas in 1 2 4; do
    log " replicas=$replicas (shared RPM=$RPM_LIMIT)"
    start_backends

    local shared_prefix="scaling-$(date +%s)-$replicas"
    local urls=""
    for r in $(seq 1 "$replicas"); do
      local port=$((GATEWAY_PORT + r - 1))
      local rcfg="$TMP_DIR/gw-scale-$replicas-$r.yaml"
      write_gateway_config "$rcfg" "$port" "prefix_affinity_load_aware" "false" "false" "0.92" "true"
      python3 - "$rcfg" "$shared_prefix" "$RPM_LIMIT" <<'INNER'
import re, sys
path, prefix, rpm = sys.argv[1], sys.argv[2], sys.argv[3]
text = open(path).read()
text = re.sub(r'key_prefix: "[^"]*"', f'key_prefix: "{prefix}"', text)
text = re.sub(r'rpm: \d+', f'rpm: {rpm}', text)
open(path, "w").write(text)
INNER
      start_gateway "$rcfg" "$port" "gw-scale-$replicas-$r"
      [ -n "$urls" ] && urls+=","
      urls+="http://localhost:$port"
    done

    reset_backends
    sleep 1

    local label="scaling_${replicas}"
    # Load is spread across every replica, which is what makes the shared-limit
    # claim meaningful.
    run_k6_multi "$label" "$urls" 40

    [ $first -eq 0 ] && echo "," >>"$results"
    first=0
    python3 - "$RAW_DIR/$label.json" "$replicas" "$RPM_LIMIT" >>"$results" <<'INNER'
import json, sys
raw_path, replicas, rpm = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])

try:
    with open(raw_path) as f:
        data = json.load(f)
except Exception as exc:
    print(json.dumps({"replicas": replicas, "error": str(exc)}))
    sys.exit()

m = data.get("metrics", {})
def v(n, f, d=None):
    return m.get(n, {}).get("values", {}).get(f, d)

total = v("http_reqs", "count", 0) or 0
limited = v("waypoint_rate_limited", "count", 0) or 0
allowed = total - limited

print(json.dumps({
    "measurement": "shared_rate_limit",
    "replicas": replicas,
    "configured_rpm": rpm,
    "requests_sent": total,
    "allowed": allowed,
    "rate_limited": limited,
    "throughput_rps": v("http_reqs", "rate", 0),
    "ttft_p50_ms": v("waypoint_ttft", "p(50)"),
    "ttft_p99_ms": v("waypoint_ttft", "p(99)"),
}, indent=2))
INNER

    stop_stack
  done

  echo "]" >>"$results"
  finalize "scaling" "$results"
}

# run_k6_direct drives a backend with no gateway in the path.
run_k6_direct() {
  local label="$1" target="$2" rps="${3:-4}"
  log "  k6: $label (direct)"
  env \
    GATEWAY="$target" \
    API_KEY="unused" \
    DURATION="$DURATION" \
    RESULT_LABEL="$label" \
    RPS="$rps" STREAM=false MAX_TOKENS=32 \
    k6 run --quiet --no-usage-report bench/workload.js >"$TMP_DIR/$label.out" 2>&1 || true
  tail -10 "$TMP_DIR/$label.out" >&2
}

# run_k6_multi spreads load across several gateway replicas.
run_k6_multi() {
  local label="$1" urls="$2" rps="$3"
  log "  k6: $label (across replicas)"
  env \
    GATEWAY_URLS="$urls" \
    GATEWAY="${urls%%,*}" \
    API_KEY="$API_KEY" \
    DURATION="$DURATION" \
    RESULT_LABEL="$label" \
    RPS="$rps" STREAM=false MAX_TOKENS=32 \
    k6 run --quiet --no-usage-report bench/workload.js >"$TMP_DIR/$label.out" 2>&1 || true
  tail -10 "$TMP_DIR/$label.out" >&2
}

# ---------- result finalization ----------

# finalize wraps raw experiment output with the environment it was produced in.
# A benchmark number without its hardware and config is not reproducible.
finalize() {
  local name="$1" body="$2"
  local out="$RESULTS_DIR/$name.json"

  python3 - "$name" "$body" "$out" "$(env_json)" <<'PY'
import json, sys, datetime, os

name, body_path, out_path, env_json = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]

with open(body_path) as f:
    runs = json.load(f)

payload = {
    "experiment": name,
    "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "environment": json.loads(env_json),
    "workload": {
        "duration": os.environ.get("DURATION", "30s"),
        "system_prompts": int(os.environ.get("SYSTEM_PROMPTS", 16)),
        "prefix_capacity_blocks_per_backend": int(os.environ.get("PREFIX_CAPACITY_BLOCKS", 24)),
        "routing_prefix_tokens": int(os.environ.get("PREFIX_TOKENS", 128)),
        "backends": int(os.environ.get("BACKEND_COUNT", 3)),
        "note": ("KV cache capacity is deliberately scarce relative to the prompt "
                 "working set; with abundant capacity every backend holds every "
                 "prefix and routing policy cannot matter."),
    },
    "runs": runs,
}
with open(out_path, "w") as f:
    json.dump(payload, f, indent=2)
print(f"wrote {out_path}", file=sys.stderr)
PY
}

# ---------- entrypoint ----------

main() {
  local what="${1:-all}"
  case "$what" in
    routing)    experiment_routing ;;
    caching)    experiment_caching ;;
    resilience) experiment_resilience ;;
    scaling)    experiment_scaling ;;
    all)
      experiment_routing
      experiment_caching
      experiment_resilience
      experiment_scaling
      ;;
    *) die "unknown experiment: $what (routing|caching|resilience|scaling|all)" ;;
  esac

  log "done. results in bench/results/"
}

main "$@"
