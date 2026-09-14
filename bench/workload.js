// Waypoint load generator.
//
// Two properties make this workload meaningful rather than synthetic noise:
//
//  1. OPEN-LOOP arrival. k6's `constant-arrival-rate` executor starts requests
//     at a fixed rate regardless of how long previous ones take. A closed-loop
//     generator (N virtual users looping) throttles itself when the system
//     slows, which hides exactly the queueing behavior a gateway benchmark is
//     supposed to expose — you measure a system that never actually gets
//     overloaded.
//
//  2. ZIPF-distributed prompt popularity. Real traffic is dominated by a few
//     very common system prompts, with a long tail of rare ones. A uniform
//     distribution makes prefix caching look useless (nothing repeats) and a
//     single-prompt workload makes it look magical (everything repeats).
//     Neither tells you what the policy does in production.

import http from 'k6/http';
import { check } from 'k6';
import { Trend, Counter, Rate } from 'k6/metrics';
import { SharedArray } from 'k6/data';
import exec from 'k6/execution';

// ---------- configuration (all overridable from the runner) ----------

const GATEWAY = __ENV.GATEWAY || 'http://localhost:18080';
// When several replicas are running, spread load across all of them: a shared
// rate limit is only meaningfully tested if requests actually arrive at
// different processes.
const GATEWAY_URLS = (__ENV.GATEWAY_URLS || GATEWAY).split(',').filter(Boolean);
const API_KEY = __ENV.API_KEY || 'dev-key';
const MODEL = __ENV.MODEL || 'mock-7b';
const RPS = Number(__ENV.RPS || 20);
const DURATION = __ENV.DURATION || '30s';
const STREAM = (__ENV.STREAM || 'true') === 'true';
const MAX_TOKENS = Number(__ENV.MAX_TOKENS || 64);

// Share of requests that repeat a previously asked question verbatim, which is
// what the exact cache can serve.
const DUPLICATE_RATE = Number(__ENV.DUPLICATE_RATE || 0);
// Share that are near-duplicates (reworded), which only the semantic cache hits.
const NEAR_DUPLICATE_RATE = Number(__ENV.NEAR_DUPLICATE_RATE || 0);
// Temperature 0 makes a request cacheable; anything else does not.
const TEMPERATURE = Number(__ENV.TEMPERATURE || 0);
// Max conversation turns; higher means more prefix reuse within a conversation.
const MAX_TURNS = Number(__ENV.MAX_TURNS || 4);

// ---------- custom metrics ----------

// k6's built-in http_req_duration measures the whole response. For streaming
// that includes generation, so it says nothing about time-to-first-token — the
// number users actually feel. TTFT is measured explicitly below.
const ttft = new Trend('waypoint_ttft', true);
const e2e = new Trend('waypoint_e2e', true);
const cacheHits = new Counter('waypoint_cache_hits');
const cacheMisses = new Counter('waypoint_cache_misses');
const rateLimited = new Counter('waypoint_rate_limited');
const shed = new Counter('waypoint_shed');
const upstreamErrors = new Counter('waypoint_upstream_errors');
const successRate = new Rate('waypoint_success');
// One counter with a backend tag, not a counter per backend: k6 requires every
// metric to be declared in the init context, so creating them lazily inside the
// default function throws "metrics must be declared in the init context" and
// silently kills the iteration.
const backendRequests = new Counter('waypoint_backend_requests');

export const options = {
  discardResponseBodies: false,
  // k6 reports only avg/min/med/max/p(90)/p(95) for trends by default, so p50
  // and p99 would come back as undefined and every percentile in the results
  // file would read "n/a".
  summaryTrendStats: ['avg', 'min', 'med', 'max', 'p(50)', 'p(90)', 'p(95)', 'p(99)'],
  scenarios: {
    openloop: {
      executor: 'constant-arrival-rate',
      rate: RPS,
      timeUnit: '1s',
      duration: DURATION,
      // Preallocate generously: if k6 runs out of VUs it silently degrades into
      // a closed-loop test, which would invalidate the whole measurement.
      preAllocatedVUs: Math.max(50, RPS * 8),
      maxVUs: Math.max(200, RPS * 30),
      gracefulStop: '30s',
    },
  },
  thresholds: {
    // Deliberately permissive: these runs are measurements, not pass/fail
    // gates, and a failed threshold would mask the numbers we came for.
    waypoint_success: ['rate >= 0.0'],
  },
};

// ---------- corpus ----------

const corpus = new SharedArray('corpus', function () {
  const systemPrompts = [];
  const K = Number(__ENV.SYSTEM_PROMPTS || 8);

  // Each system prompt is long enough to span many KV cache blocks, which is
  // what makes prefix reuse worth anything.
  const roles = [
    'a meticulous billing support agent for a telecom company',
    'a senior code reviewer focused on concurrency and memory safety',
    'a network engineer troubleshooting enterprise switch configurations',
    'a patient mathematics tutor explaining linear algebra step by step',
    'a technical writer producing release notes for infrastructure software',
    'a data analyst interpreting experiment results for a product team',
    'a security analyst triaging alerts from an intrusion detection system',
    'a logistics coordinator resolving freight exceptions and delays',
  ];

  for (let i = 0; i < K; i++) {
    const role = roles[i % roles.length];
    systemPrompts.push(
      `You are ${role}. Answer precisely and cite assumptions. ` +
      `Prefer concrete examples over abstractions. If a question is ambiguous, ` +
      `state the interpretation you chose before answering. Keep responses ` +
      `structured and avoid unnecessary preamble. This instruction set is ` +
      `revision ${i} and applies to all subsequent turns in this conversation. `.repeat(4)
    );
  }

  const questions = [
    'how do I reconcile a duplicate charge on the latest invoice',
    'what causes a deadlock between two mutexes acquired in different orders',
    'why would a trunk port drop VLAN tagged frames intermittently',
    'explain the intuition behind eigenvalues for a covariance matrix',
    'summarize the breaking changes for consumers of the public API',
    'is the observed lift statistically significant at this sample size',
    'what distinguishes a port scan from normal service discovery traffic',
    'how should I handle a shipment delayed at a customs checkpoint',
    'what is the difference between prefill and decode in LLM inference',
    'how does a circuit breaker differ from a simple retry limit',
    'when is consistent hashing preferable to a least connections policy',
    'what are the tradeoffs of caching responses from a language model',
  ];

  // Rewordings used for near-duplicate traffic: same meaning, different
  // surface form, so only the semantic cache can serve them.
  const rewordings = [
    (q) => `could you explain ${q}`,
    (q) => `${q}, briefly`,
    (q) => `I'd like to understand ${q}`,
    (q) => `please describe ${q}`,
  ];

  return [{ systemPrompts, questions, rewordings }];
})[0];

// zipfIndex draws an index from a Zipf distribution over n items.
//
// Rank r has probability proportional to 1/r^s. With s=1.1 the most popular
// system prompt takes roughly a third of traffic and the tail is genuinely
// long — close to what production LLM traffic looks like.
function zipfIndex(n, s) {
  let norm = 0;
  for (let i = 1; i <= n; i++) norm += 1 / Math.pow(i, s);

  const u = Math.random() * norm;
  let acc = 0;
  for (let i = 1; i <= n; i++) {
    acc += 1 / Math.pow(i, s);
    if (u <= acc) return i - 1;
  }
  return n - 1;
}

// A small per-VU history enables verbatim repeats without coordination.
const askedByVU = {};

function buildMessages() {
  const vu = exec.vu.idInTest;
  if (!askedByVU[vu]) askedByVU[vu] = [];
  const history = askedByVU[vu];

  const sysIdx = zipfIndex(corpus.systemPrompts.length, Number(__ENV.ZIPF_S || 1.1));
  const system = corpus.systemPrompts[sysIdx];

  let question;
  const roll = Math.random();

  if (roll < DUPLICATE_RATE && history.length > 0) {
    // Exact repeat: servable from the exact cache.
    question = history[Math.floor(Math.random() * history.length)];
  } else if (roll < DUPLICATE_RATE + NEAR_DUPLICATE_RATE && history.length > 0) {
    // Reworded repeat: only the semantic cache can serve this.
    const original = history[Math.floor(Math.random() * history.length)];
    const reword = corpus.rewordings[Math.floor(Math.random() * corpus.rewordings.length)];
    question = reword(original);
  } else {
    question = corpus.questions[Math.floor(Math.random() * corpus.questions.length)];
    history.push(question);
    if (history.length > 40) history.shift();
  }

  const messages = [{ role: 'system', content: system }];

  // Multi-turn conversations reuse the same prefix across turns, which is what
  // makes affinity routing sticky for a whole conversation rather than one
  // request.
  const turns = 1 + Math.floor(Math.random() * MAX_TURNS);
  for (let t = 0; t < turns - 1; t++) {
    messages.push({ role: 'user', content: corpus.questions[(sysIdx + t) % corpus.questions.length] });
    messages.push({ role: 'assistant', content: 'Understood. Here is a concise answer to that.' });
  }
  messages.push({ role: 'user', content: question });

  return messages;
}

export default function () {
  const payload = JSON.stringify({
    model: MODEL,
    stream: STREAM,
    temperature: TEMPERATURE,
    max_tokens: MAX_TOKENS,
    messages: buildMessages(),
  });

  const start = Date.now();
  const target = GATEWAY_URLS[exec.scenario.iterationInTest % GATEWAY_URLS.length];
  const res = http.post(`${target}/v1/chat/completions`, payload, {
    headers: {
      'Content-Type': 'application/json',
      Authorization: `Bearer ${API_KEY}`,
    },
    timeout: '120s',
    tags: { name: 'chat_completions' },
  });
  const finished = Date.now();

  const ok = res.status === 200;
  successRate.add(ok);

  if (res.status === 429) rateLimited.add(1);
  if (res.status === 503) shed.add(1);
  if (res.status === 502 || res.status === 504) upstreamErrors.add(1);

  if (!ok) return;

  e2e.add(finished - start);

  const cacheHeader = res.headers['X-Waypoint-Cache'] || '';
  if (cacheHeader === 'exact' || cacheHeader === 'semantic') {
    cacheHits.add(1, { layer: cacheHeader });
  } else if (cacheHeader === 'miss') {
    cacheMisses.add(1);
  }

  const backend = res.headers['X-Waypoint-Backend'] || 'unknown';
  backendRequests.add(1, { backend });

  // TTFT: time until the first content-bearing SSE chunk.
  //
  // k6 has no streaming HTTP client, so the body arrives complete and the
  // per-chunk arrival time is not observable here. What *is* observable is the
  // server's own timing, which the gateway reports; absent that, the closest
  // honest proxy is the time to the first response byte, which k6 exposes as
  // waiting time. Using total duration would conflate prefill with generation
  // and make the routing comparison meaningless.
  const waiting = res.timings.waiting;
  if (waiting > 0) ttft.add(waiting);

  check(res, {
    'status 200': (r) => r.status === 200,
    'has content': (r) => (r.body || '').length > 0,
  });
}

// handleSummary writes a machine-readable result next to k6's console output.
export function handleSummary(data) {
  const out = {};
  const label = __ENV.RESULT_LABEL || 'run';
  out[`bench/results/raw/${label}.json`] = JSON.stringify(data, null, 2);
  out.stdout = textSummary(data);
  return out;
}

function textSummary(data) {
  const m = data.metrics;
  const get = (name, field) => (m[name] && m[name].values ? m[name].values[field] : null);
  const fmt = (v) => (v === null || v === undefined ? 'n/a' : v.toFixed(2));

  return [
    '',
    `requests:      ${get('http_reqs', 'count') || 0}`,
    `success rate:  ${fmt((get('waypoint_success', 'rate') || 0) * 100)}%`,
    `TTFT p50/p95/p99 (ms): ${fmt(get('waypoint_ttft', 'p(50)'))} / ${fmt(get('waypoint_ttft', 'p(95)'))} / ${fmt(get('waypoint_ttft', 'p(99)'))}`,
    `E2E  p50/p95/p99 (ms): ${fmt(get('waypoint_e2e', 'p(50)'))} / ${fmt(get('waypoint_e2e', 'p(95)'))} / ${fmt(get('waypoint_e2e', 'p(99)'))}`,
    `cache hits:    ${get('waypoint_cache_hits', 'count') || 0}`,
    `rate limited:  ${get('waypoint_rate_limited', 'count') || 0}`,
    `shed (503):    ${get('waypoint_shed', 'count') || 0}`,
    '',
  ].join('\n');
}
