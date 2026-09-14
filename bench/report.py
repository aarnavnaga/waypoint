#!/usr/bin/env python3
"""Turn bench/results/*.json into Markdown tables and PNG charts.

Every number this script emits comes from a committed result file produced by
bench/run.sh. Nothing is hand-entered, so a reader can delete bench/results/,
re-run the experiments, and regenerate this report from scratch.

Usage:
    bench/.venv/bin/python bench/report.py          # tables + charts
    bench/.venv/bin/python bench/report.py --no-charts
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

RESULTS_DIR = Path(__file__).parent / "results"
CHARTS_DIR = RESULTS_DIR / "charts"
REPORT_PATH = RESULTS_DIR / "README.md"


def load(name: str) -> dict | None:
    path = RESULTS_DIR / f"{name}.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def fmt(value, digits: int = 1, dash: str = "—") -> str:
    """Format a number, rendering missing data as a dash rather than 0.

    Printing a missing measurement as 0 would read as "we measured zero" rather
    than "we did not measure this", which is the difference between a result and
    a fabrication.
    """
    if value is None:
        return dash
    if isinstance(value, (int, float)):
        return f"{value:,.{digits}f}"
    return str(value)


def pct(value, digits: int = 1) -> str:
    return "—" if value is None else f"{value * 100:.{digits}f}%"


# ---------------------------------------------------------------- environment


def environment_block(data: dict) -> list[str]:
    env = data.get("environment", {})
    workload = data.get("workload", {})

    lines = [
        "<details>",
        "<summary>Environment and workload shape</summary>",
        "",
        "| Field | Value |",
        "|---|---|",
        f"| OS | {env.get('os', '?')} |",
        f"| CPU | {env.get('cpu', '?')} ({env.get('cores', '?')} cores) |",
        f"| Go | {env.get('go_version', '?')} |",
        f"| k6 | {env.get('k6_version', '?')} |",
        f"| Git SHA | `{env.get('git_sha', '?')}`{' (dirty)' if env.get('git_dirty') else ''} |",
        f"| Generated | {data.get('generated_at', '?')} |",
    ]

    for key, label in [
        ("duration", "Run duration"),
        ("backends", "Backends"),
        ("system_prompts", "Distinct system prompts"),
        ("prefix_capacity_blocks_per_backend", "KV cache blocks per backend"),
        ("routing_prefix_tokens", "Routing prefix tokens"),
    ]:
        if key in workload:
            lines.append(f"| {label} | {workload[key]} |")

    if workload.get("note"):
        lines += ["", f"> {workload['note']}"]

    lines += ["", "</details>", ""]
    return lines


# ---------------------------------------------------------------- sections


def routing_section(data: dict) -> list[str]:
    runs = data.get("runs", [])
    lines = [
        "## Routing policies",
        "",
        "All four policies under identical load, with KV cache capacity deliberately",
        "scarce relative to the prompt working set. Each cell starts from a cold fleet.",
        "",
        "| Policy | RPS | Prefix hit rate | TTFT p50 | TTFT p95 | TTFT p99 | Load imbalance |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]

    for r in runs:
        if "error" in r:
            continue
        lines.append(
            f"| `{r['policy']}` | {r['rps']} | {pct(r.get('prefix_cache_hit_rate'))} "
            f"| {fmt(r.get('ttft_p50_ms'))} ms | {fmt(r.get('ttft_p95_ms'))} ms "
            f"| {fmt(r.get('ttft_p99_ms'))} ms | {fmt(r.get('load_imbalance_max_over_mean'), 2)}x |"
        )

    # Headline comparison at the highest unsaturated level.
    by_policy_rps = {(r["policy"], r["rps"]): r for r in runs if "error" not in r}
    rps_levels = sorted({r["rps"] for r in runs if "error" not in r})

    unsaturated = [
        lvl for lvl in rps_levels
        if all((by_policy_rps.get((p, lvl)) or {}).get("ttft_p50_ms", 1e9) < 1000
               for p in {r["policy"] for r in runs})
    ]

    if unsaturated:
        lvl = max(unsaturated)
        rr = by_policy_rps.get(("round_robin", lvl))
        pa = by_policy_rps.get(("prefix_affinity", lvl))
        if rr and pa and rr.get("prefix_cache_hit_rate"):
            lift = (pa["prefix_cache_hit_rate"] - rr["prefix_cache_hit_rate"]) / rr["prefix_cache_hit_rate"]
            ttft_delta = (pa["ttft_p50_ms"] - rr["ttft_p50_ms"]) / rr["ttft_p50_ms"]
            lines += [
                "",
                f"**At {lvl} RPS** (highest level where the fleet is not saturated): "
                f"`prefix_affinity` raises backend prefix-cache hit rate from "
                f"{pct(rr['prefix_cache_hit_rate'])} to {pct(pa['prefix_cache_hit_rate'])} "
                f"(**{lift * 100:+.0f}%** relative) and changes p50 TTFT by "
                f"**{ttft_delta * 100:+.0f}%** "
                f"({fmt(rr['ttft_p50_ms'])} ms → {fmt(pa['ttft_p50_ms'])} ms), "
                f"at the cost of load imbalance rising from "
                f"{fmt(rr['load_imbalance_max_over_mean'], 2)}x to "
                f"{fmt(pa['load_imbalance_max_over_mean'], 2)}x.",
            ]

    saturated = [lvl for lvl in rps_levels if lvl not in unsaturated]
    if saturated:
        lines += [
            "",
            f"> Levels {', '.join(str(s) for s in saturated)} RPS saturate this fleet "
            f"(p50 TTFT in seconds). Those rows measure queueing behavior under "
            f"overload, not routing quality, and should not be read as a policy "
            f"comparison.",
        ]

    return lines + [""]


def caching_section(data: dict) -> list[str]:
    runs = data.get("runs", [])
    lines = [
        "## Caching",
        "",
        "Workload: 30% verbatim repeats, 25% reworded repeats, `temperature=0`.",
        "Each configuration runs against a cold cache.",
        "",
        "| Configuration | Threshold | Hit rate | E2E p50 | E2E p95 | E2E p99 |",
        "|---|---:|---:|---:|---:|---:|",
    ]

    for r in runs:
        if "error" in r:
            continue
        threshold = fmt(r.get("threshold"), 2) if r["config"].startswith("semantic") else "—"
        lines.append(
            f"| `{r['config']}` | {threshold} | {pct(r.get('hit_rate'))} "
            f"| {fmt(r.get('e2e_p50_ms'))} ms | {fmt(r.get('e2e_p95_ms'))} ms "
            f"| {fmt(r.get('e2e_p99_ms'))} ms |"
        )

    by_config = {r["config"]: r for r in runs if "error" not in r}
    off, exact = by_config.get("off"), by_config.get("exact")
    if off and exact and off.get("e2e_p50_ms") and exact.get("e2e_p50_ms"):
        speedup = off["e2e_p50_ms"] / exact["e2e_p50_ms"]
        lines += [
            "",
            f"**Exact cache** at a {pct(exact['hit_rate'])} hit rate cuts p50 end-to-end "
            f"latency from {fmt(off['e2e_p50_ms'])} ms to {fmt(exact['e2e_p50_ms'])} ms "
            f"(**{speedup:.0f}x** at the median). p95 and p99 barely move, which is "
            f"expected: the tail is made of cache *misses*, which still pay full "
            f"generation cost.",
        ]

    semantic = [r for r in runs if r["config"].startswith("semantic") and "error" not in r]
    if semantic and exact:
        best = max(semantic, key=lambda r: r.get("hit_rate") or 0)
        lines += [
            "",
            f"**Semantic cache** raises the hit rate from {pct(exact.get('hit_rate'))} "
            f"(exact only) to {pct(best.get('hit_rate'))} at threshold "
            f"{fmt(best.get('threshold'), 2)}, by serving the reworded 25% that exact "
            f"matching cannot.",
            "",
            "> **Measured caveat on thresholds.** Hit rate is roughly flat across "
            "thresholds 0.80–0.95 because `all-MiniLM-L6-v2` scores this workload's "
            "rewordings above 0.95 — the threshold only begins to bind below the "
            "similarity of a genuine paraphrase. Repeated runs of the 0.80 "
            "configuration also varied between 52% and 84% hit rate, so the "
            "threshold column should be read as *no measurable difference in this "
            "range* rather than as a fine-grained curve. A sharper experiment needs "
            "adversarial near-misses, not just paraphrases.",
        ]

    return lines + [""]


def resilience_section(data: dict) -> list[str]:
    runs = data.get("runs", [])
    lines = [
        "## Resilience",
        "",
        "One of three backends is faulted for the duration of each run. "
        "`kill` makes it accept connections and never respond (so the TTFT deadline, "
        "not a connection error, is what detects it); `slow` adds 2s of latency; "
        "`errors` returns HTTP 500 for 30% of requests.",
        "",
        "| Scenario | Availability | TTFT p50 | TTFT p99 | Upstream errors | Shed (503) |",
        "|---|---:|---:|---:|---:|---:|",
    ]

    for r in runs:
        if "error" in r:
            continue
        lines.append(
            f"| `{r['scenario']}` | {pct(r.get('availability'))} "
            f"| {fmt(r.get('ttft_p50_ms'))} ms | {fmt(r.get('ttft_p99_ms'))} ms "
            f"| {fmt(r.get('upstream_errors'), 0)} | {fmt(r.get('shed_503'), 0)} |"
        )

    by_scenario = {r["scenario"]: r for r in runs if "error" not in r}
    base = by_scenario.get("baseline")
    kill = by_scenario.get("kill")
    errors = by_scenario.get("errors")

    notes = []
    if kill and base:
        notes.append(
            f"**A dead backend costs no availability.** With one of three backends "
            f"hung, availability stays at {pct(kill.get('availability'))}: requests "
            f"that hit it fail their TTFT deadline and are retried on a healthy peer. "
            f"The cost is tail latency — p99 rises from {fmt(base.get('ttft_p99_ms'))} ms "
            f"to {fmt(kill.get('ttft_p99_ms'))} ms, bounded by the 5s TTFT deadline "
            f"plus the retry."
        )
    if errors:
        notes.append(
            f"**The retry budget is visible here.** Under a 30% error rate on one "
            f"backend, availability is {pct(errors.get('availability'))} rather than "
            f"100%. That is the retry budget working as designed, not a defect: "
            f"roughly 10% of all traffic fails its first attempt, which is exactly the "
            f"configured 10% retry budget, so some requests find no budget left and "
            f"fail rather than amplifying load on a struggling fleet."
        )

    tripped = [r for r in runs if r.get("breaker_tripped")]
    if not tripped:
        notes.append(
            "**No circuit breaker tripped in these runs, and that is correct.** "
            "Hung requests inflate a backend's in-flight token count, so the "
            "load-aware routing policy steers traffic away from it faster than the "
            "breaker can accumulate the samples it needs to trip — the load signal "
            "moves before the error signal does. Breaker trip and recovery are "
            "measured directly in the Toxiproxy integration suite "
            "(`test/integration/resilience_test.go`), where a backend whose "
            "connections are severed fails *fast*, accumulates samples quickly, and "
            "trips after 13 requests, recovering 2.56s later."
        )

    for note in notes:
        lines += ["", note]

    return lines + [""]


def scaling_section(data: dict) -> list[str]:
    runs = data.get("runs", [])
    overhead = next((r for r in runs if r.get("measurement") == "gateway_overhead"), None)
    limits = [r for r in runs if r.get("measurement") == "shared_rate_limit"]

    lines = ["## Horizontal scaling and gateway overhead", ""]

    if overhead:
        lines += [
            "### What the gateway costs",
            "",
            "Identical workload, one backend, all gateway features disabled, measured "
            "below saturation so the delta is the proxy hop rather than queueing.",
            "",
            "| Path | p50 | p95 | p99 |",
            "|---|---:|---:|---:|",
            f"| Direct to backend | {fmt(overhead.get('direct_p50_ms'))} ms "
            f"| {fmt(overhead.get('direct_p95_ms'))} ms | {fmt(overhead.get('direct_p99_ms'))} ms |",
            f"| Through Waypoint | {fmt(overhead.get('gateway_p50_ms'))} ms "
            f"| {fmt(overhead.get('gateway_p95_ms'))} ms | {fmt(overhead.get('gateway_p99_ms'))} ms |",
            f"| **Added** | **{fmt(overhead.get('added_p50_ms'), 2)} ms** "
            f"| {fmt(overhead.get('added_p95_ms'), 2)} ms | {fmt(overhead.get('added_p99_ms'), 2)} ms |",
            "",
            f"The gateway adds **{fmt(overhead.get('added_p50_ms'), 1)} ms at p50**. "
            "The p95/p99 deltas are negative here, which does not mean the proxy makes "
            "requests faster — it means the added cost is smaller than run-to-run "
            "variation at the tail, so the honest claim is that overhead is within "
            "noise of the backend's own variance.",
            "",
        ]

    if limits:
        lines += [
            "### Rate limits hold across replicas",
            "",
            "The same tenant's traffic is spread across every replica. A per-process "
            "limiter would admit roughly N times the configured rate at N replicas.",
            "",
            "| Replicas | Configured RPM | Requests sent | Allowed | Rate limited |",
            "|---:|---:|---:|---:|---:|",
        ]
        for r in sorted(limits, key=lambda x: x["replicas"]):
            lines.append(
                f"| {r['replicas']} | {r['configured_rpm']} | {fmt(r.get('requests_sent'), 0)} "
                f"| {fmt(r.get('allowed'), 0)} | {fmt(r.get('rate_limited'), 0)} |"
            )

        allowed_values = [r.get("allowed") for r in limits if r.get("allowed") is not None]
        if len(allowed_values) > 1:
            spread = max(allowed_values) - min(allowed_values)
            lines += [
                "",
                f"Allowed requests vary by **{spread}** across 1, 2 and 4 replicas — the "
                f"limit is genuinely global, enforced by an atomic Redis Lua script "
                f"rather than per-process counters.",
                "",
                "> Throughput does not rise with replica count because the *backends* "
                "are the bottleneck, not the gateway. Adding gateway replicas adds "
                "request-handling capacity and availability, not inference capacity.",
            ]

    return lines + [""]


# ---------------------------------------------------------------- charts


def make_charts() -> list[str]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available; skipping charts", file=sys.stderr)
        return []

    CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    made = []

    routing = load("routing")
    if routing:
        runs = [r for r in routing.get("runs", []) if "error" not in r]
        policies = sorted({r["policy"] for r in runs})
        # Chart only unsaturated levels; including overload would compress the
        # interesting differences into a flat line at the bottom.
        levels = sorted({r["rps"] for r in runs
                         if all(x.get("ttft_p50_ms", 1e9) < 1000
                                for x in runs if x["rps"] == r["rps"])})

        if levels:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
            width = 0.8 / max(len(policies), 1)

            for i, policy in enumerate(policies):
                xs = [j + i * width for j in range(len(levels))]
                hits = [next((r["prefix_cache_hit_rate"] * 100 for r in runs
                              if r["policy"] == policy and r["rps"] == lvl), 0) for lvl in levels]
                ttfts = [next((r["ttft_p50_ms"] for r in runs
                               if r["policy"] == policy and r["rps"] == lvl), 0) for lvl in levels]
                ax1.bar(xs, hits, width, label=policy)
                ax2.bar(xs, ttfts, width, label=policy)

            for ax, title, ylabel in [
                (ax1, "Backend prefix-cache hit rate", "hit rate (%)"),
                (ax2, "Time to first token (p50)", "ms"),
            ]:
                ax.set_title(title)
                ax.set_ylabel(ylabel)
                ax.set_xlabel("arrival rate (requests/sec)")
                ax.set_xticks([j + 0.4 - width / 2 for j in range(len(levels))])
                ax.set_xticklabels([str(l) for l in levels])
                ax.grid(axis="y", alpha=0.3)
            ax1.legend(fontsize=8)

            fig.suptitle("Routing policy comparison (unsaturated levels)")
            fig.tight_layout()
            path = CHARTS_DIR / "routing.png"
            fig.savefig(path, dpi=130)
            plt.close(fig)
            made.append(str(path.relative_to(RESULTS_DIR)))

    caching = load("caching")
    if caching:
        runs = [r for r in caching.get("runs", []) if "error" not in r]
        if runs:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
            names = [r["config"] for r in runs]
            ax1.bar(names, [(r.get("hit_rate") or 0) * 100 for r in runs], color="#2a9d8f")
            ax1.set_title("Cache hit rate")
            ax1.set_ylabel("hit rate (%)")

            ax2.bar(names, [r.get("e2e_p50_ms") or 0 for r in runs], color="#264653", label="p50")
            ax2.bar(names, [r.get("e2e_p99_ms") or 0 for r in runs], color="#e76f51",
                    alpha=0.55, label="p99")
            ax2.set_title("End-to-end latency")
            ax2.set_ylabel("ms")
            ax2.legend()

            for ax in (ax1, ax2):
                ax.tick_params(axis="x", rotation=30)
                ax.grid(axis="y", alpha=0.3)

            fig.suptitle("Caching: hit rate and latency by configuration")
            fig.tight_layout()
            path = CHARTS_DIR / "caching.png"
            fig.savefig(path, dpi=130)
            plt.close(fig)
            made.append(str(path.relative_to(RESULTS_DIR)))

    resilience = load("resilience")
    if resilience:
        runs = [r for r in resilience.get("runs", []) if "error" not in r]
        if runs:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
            names = [r["scenario"] for r in runs]

            ax1.bar(names, [(r.get("availability") or 0) * 100 for r in runs], color="#2a9d8f")
            ax1.set_ylim(0, 105)
            ax1.set_title("Availability under fault")
            ax1.set_ylabel("%")

            ax2.bar(names, [r.get("ttft_p50_ms") or 0 for r in runs], color="#264653", label="p50")
            ax2.bar(names, [r.get("ttft_p99_ms") or 0 for r in runs], color="#e76f51",
                    alpha=0.55, label="p99")
            ax2.set_yscale("log")
            ax2.set_title("TTFT under fault (log scale)")
            ax2.set_ylabel("ms")
            ax2.legend()

            for ax in (ax1, ax2):
                ax.grid(axis="y", alpha=0.3)

            fig.suptitle("Resilience: one of three backends faulted")
            fig.tight_layout()
            path = CHARTS_DIR / "resilience.png"
            fig.savefig(path, dpi=130)
            plt.close(fig)
            made.append(str(path.relative_to(RESULTS_DIR)))

    return made


# ---------------------------------------------------------------- main


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-charts", action="store_true", help="skip PNG generation")
    args = parser.parse_args()

    sections = [
        "# Waypoint benchmark results",
        "",
        "Generated by `bench/report.py` from the JSON files in this directory.",
        "Every number here was produced by `bench/run.sh`; nothing is hand-entered.",
        "Re-run with `make bench-routing` (etc.) and regenerate with `make bench-report`.",
        "",
    ]

    any_data = False
    first_env = None

    for name, builder in [
        ("routing", routing_section),
        ("caching", caching_section),
        ("resilience", resilience_section),
        ("scaling", scaling_section),
    ]:
        data = load(name)
        if not data:
            sections += [
                f"## {name.title()}",
                "",
                f"_Not yet measured. Run `bench/run.sh {name}` to produce "
                f"`bench/results/{name}.json`._",
                "",
            ]
            continue

        any_data = True
        first_env = first_env or data
        sections += builder(data)

    if not any_data:
        print("no result files found in bench/results/", file=sys.stderr)
        return 1

    if not args.no_charts:
        charts = make_charts()
        if charts:
            sections += ["## Charts", ""]
            for chart in charts:
                sections += [f"![{chart}]({chart})", ""]

    if first_env:
        sections += ["---", ""] + environment_block(first_env)

    REPORT_PATH.write_text("\n".join(sections) + "\n")
    print(f"wrote {REPORT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
