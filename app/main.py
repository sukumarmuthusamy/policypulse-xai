"""FastAPI application entrypoint for PolicyPulse."""

from __future__ import annotations

import json
import logging
import statistics
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel

from app.agents.orchestrator import run_agent
from app.agents.schemas import AgentRequest, AgentResponse
from app.agents.tools import get_policy_index_store
from app.config import get_settings
from app.observability.tracer import read_recent_traces
from scripts.build_index import build_policy_index

logger = logging.getLogger(__name__)


class HealthResponse(BaseModel):
    status: str


class MetadataResponse(BaseModel):
    model_provider: str
    model_name: str
    embedding_model: str
    deployment_target: str
    index_ready: bool
    chunk_count: int = 0
    index_built_at: str | None = None
    vector_count: int = 0
    index_error: str | None = None
    latency_last_ms: int | None = None
    latency_p50_ms: int | None = None
    recent_trace_count: int = 0


class UploadResponse(BaseModel):
    filename: str
    chunk_count: int
    vector_count: int
    index_built_at: str | None = None
    message: str


def _compute_latency_stats() -> tuple[int | None, int | None, int]:
    traces = read_recent_traces(limit=20)
    if not traces:
        return None, None, 0

    latencies = [trace.latency_ms for trace in traces]
    last_latency = latencies[-1]
    p50_latency = int(statistics.median(latencies))
    return last_latency, p50_latency, len(traces)


def _read_index_metadata() -> dict[str, Any]:
    settings = get_settings()
    if not settings.chunks_path.exists():
        return {}

    try:
        payload = json.loads(settings.chunks_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}

    return payload if isinstance(payload, dict) else {}


def _build_metadata_response(
    *,
    index_ready: bool,
    index_error: str | None = None,
) -> MetadataResponse:
    settings = get_settings()
    index_metadata = _read_index_metadata() if index_ready else {}
    last_latency, p50_latency, trace_count = _compute_latency_stats()

    vector_count = 0
    if index_ready:
        store = get_policy_index_store()
        if store.is_loaded and store._index is not None:
            vector_count = int(store._index.ntotal)

    return MetadataResponse(
        model_provider=settings.model_provider,
        model_name=settings.resolved_model_name,
        embedding_model=settings.resolved_embedding_model_name,
        deployment_target=settings.deployment_target,
        index_ready=index_ready,
        chunk_count=int(index_metadata.get("chunk_count", 0)),
        index_built_at=str(index_metadata["built_at"]) if index_metadata.get("built_at") else None,
        vector_count=vector_count,
        index_error=index_error,
        latency_last_ms=last_latency,
        latency_p50_ms=p50_latency,
        recent_trace_count=trace_count,
    )


def _sync_index_app_state(app: FastAPI) -> None:
    """Mirror warmed indexes from the policy store onto FastAPI app state."""
    store = get_policy_index_store()
    app.state.policy_index_store = store
    app.state.bm25_index = store.bm25_index


def _reload_index_state(app: FastAPI) -> dict[str, Any]:
    """Rebuild in-memory hybrid indexes from disk and sync FastAPI app flags."""
    store = get_policy_index_store()
    store.reload()
    app.state.index_ready = True
    app.state.index_error = None
    _sync_index_app_state(app)

    index_metadata = _read_index_metadata()
    vector_count = int(store._index.ntotal) if store._index is not None else 0
    return {
        "chunk_count": int(index_metadata.get("chunk_count", vector_count)),
        "vector_count": vector_count,
        "index_built_at": index_metadata.get("built_at"),
    }


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Warm-load the FAISS and BM25 indexes on startup for fast first retrieval."""
    index_ready = False
    index_error: str | None = None

    try:
        store = get_policy_index_store()
        store.load()
        index_ready = True
        _sync_index_app_state(app)
        logger.info(
            "Policy indexes loaded (%s vectors, bm25=%s).",
            store._index.ntotal if store._index is not None else 0,
            "ready" if store.bm25_index is not None else "missing",
        )
    except FileNotFoundError as error:
        index_error = str(error)
        logger.warning("Policy index not loaded at startup: %s", error)
    except Exception as error:
        index_error = str(error)
        logger.exception("Unexpected error while loading policy index: %s", error)

    app.state.index_ready = index_ready
    app.state.index_error = index_error
    if not index_ready:
        app.state.policy_index_store = None
        app.state.bm25_index = None
    yield


app = FastAPI(
    title="PolicyPulse",
    description="Enterprise Policy Copilot API",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Liveness probe for orchestrators and Cloud Run."""
    return HealthResponse(status="ok")


@app.get("/metadata", response_model=MetadataResponse)
def metadata() -> MetadataResponse:
    """Expose deployment, model, index, and latency telemetry for the UI sidebar."""
    index_ready = bool(getattr(app.state, "index_ready", False))
    index_error = getattr(app.state, "index_error", None)
    return _build_metadata_response(index_ready=index_ready, index_error=index_error)


@app.post("/agent", response_model=AgentResponse)
def agent(request: AgentRequest) -> AgentResponse:
    """Run the policy agent orchestrator for a user query."""
    if not getattr(app.state, "index_ready", False):
        raise HTTPException(
            status_code=503,
            detail=(
                "Policy index is not loaded. "
                "Add PDFs to data/policies/ and run `python scripts/build_index.py`."
            ),
        )

    return run_agent(request)


@app.post("/upload", response_model=UploadResponse)
async def upload_policy(file: UploadFile = File(...)) -> UploadResponse:
    """Save an uploaded PDF, rebuild the FAISS index, and hot-reload it in memory."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="Uploaded file must have a filename.")

    safe_name = Path(file.filename).name
    if not safe_name.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    settings = get_settings()
    settings.policies_dir.mkdir(parents=True, exist_ok=True)
    destination = settings.policies_dir / safe_name

    try:
        contents = await file.read()
        if not contents:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")
        destination.write_bytes(contents)

        build_summary = build_policy_index()
        index_state = _reload_index_state(app)
    except FileNotFoundError as error:
        app.state.index_ready = False
        app.state.index_error = str(error)
        raise HTTPException(status_code=400, detail=str(error)) from error
    except HTTPException:
        raise
    except Exception as error:
        app.state.index_ready = False
        app.state.index_error = str(error)
        logger.exception("Policy upload and index rebuild failed: %s", error)
        raise HTTPException(status_code=500, detail=f"Index rebuild failed: {error}") from error

    chunk_count = int(build_summary.get("chunk_count", index_state["chunk_count"]))
    vector_count = int(index_state["vector_count"])
    built_at = index_state.get("index_built_at")

    return UploadResponse(
        filename=safe_name,
        chunk_count=chunk_count,
        vector_count=vector_count,
        index_built_at=str(built_at) if built_at else None,
        message=f"Saved '{safe_name}' and rebuilt the policy index.",
    )
