"""Embeddings sidecar for Waypoint's semantic cache.

Runs sentence-transformers on CPU behind a small FastAPI surface. It lives in a
separate process rather than inside the gateway for two reasons: there is no
good pure-Go sentence-transformers runtime, and embedding is CPU-bound work that
should not contend with a latency-sensitive proxy's scheduler and GC.

The gateway treats this service as optional — if it is slow or down, the
semantic cache degrades to a miss and requests are served live.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from typing import List

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("embeddings")

MODEL_NAME = os.getenv("MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
MAX_BATCH = int(os.getenv("MAX_BATCH", "64"))
MAX_TEXT_CHARS = int(os.getenv("MAX_TEXT_CHARS", "8192"))

app = FastAPI(title="waypoint-embeddings", version="1.0.0")

# The model is loaded in a background thread so the container can start serving
# /healthz immediately and report "not ready" rather than failing its healthcheck
# while a few hundred megabytes download.
_model = None
_model_error: str | None = None
_load_started = time.monotonic()
_lock = threading.Lock()


def _load_model() -> None:
    global _model, _model_error
    try:
        from sentence_transformers import SentenceTransformer

        log.info("loading model %s", MODEL_NAME)
        started = time.monotonic()
        model = SentenceTransformer(MODEL_NAME, device="cpu")
        with _lock:
            _model = model
        log.info("model loaded in %.1fs", time.monotonic() - started)
    except Exception as exc:  # noqa: BLE001 - surfaced via /healthz
        log.exception("failed to load model")
        with _lock:
            _model_error = str(exc)


threading.Thread(target=_load_model, daemon=True).start()


class EmbedRequest(BaseModel):
    texts: List[str] = Field(..., min_length=1)


class EmbedResponse(BaseModel):
    embeddings: List[List[float]]
    model: str
    dimensions: int


@app.get("/healthz")
def healthz() -> dict:
    """Liveness plus model-load status."""
    with _lock:
        model, error = _model, _model_error

    if error:
        raise HTTPException(status_code=503, detail=f"model failed to load: {error}")
    if model is None:
        raise HTTPException(
            status_code=503,
            detail=f"model still loading ({time.monotonic() - _load_started:.0f}s)",
        )
    return {
        "status": "ok",
        "model": MODEL_NAME,
        "dimensions": model.get_sentence_embedding_dimension(),
    }


@app.post("/embed", response_model=EmbedResponse)
def embed(req: EmbedRequest) -> EmbedResponse:
    with _lock:
        model = _model
    if model is None:
        raise HTTPException(status_code=503, detail="model not loaded")

    if len(req.texts) > MAX_BATCH:
        raise HTTPException(
            status_code=413, detail=f"batch of {len(req.texts)} exceeds max {MAX_BATCH}"
        )

    # Truncate rather than reject: an over-long prompt should still be
    # embeddable, and the leading text carries most of the signal.
    texts = [t[:MAX_TEXT_CHARS] for t in req.texts]

    # normalize_embeddings makes cosine similarity a plain dot product, matching
    # what the gateway and Redis expect.
    vectors = model.encode(
        texts,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )

    return EmbedResponse(
        embeddings=[v.tolist() for v in vectors],
        model=MODEL_NAME,
        dimensions=int(vectors.shape[1]),
    )
