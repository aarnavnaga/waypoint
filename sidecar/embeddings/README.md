# Embeddings sidecar

CPU-only sentence-transformers behind FastAPI, used by Waypoint's semantic cache.

## Why a separate process

There is no good pure-Go sentence-transformers runtime, and embedding is CPU-bound
work that should not contend with a latency-sensitive proxy for scheduler time and
GC. Keeping it out-of-process also means the gateway can treat it as optional: if
the sidecar is slow or down, the semantic cache degrades to a miss and requests are
served live.

## API

- `GET /healthz` — 200 once the model is loaded, 503 while loading or on failure.
  The model loads in a background thread so the container answers health checks
  immediately instead of appearing dead during a multi-hundred-megabyte download.
- `POST /embed` — `{"texts": ["..."]}` → `{"embeddings": [[...]], "model": "...", "dimensions": 384}`

Embeddings are L2-normalized, which makes cosine similarity a plain dot product and
matches what the gateway and the Redis vector index expect.

## Run locally

```bash
pip install -r requirements.txt
uvicorn app:app --port 8000
```

First start downloads the model (~90MB), so `make up` gives this service a long
`start_period` in its healthcheck.
