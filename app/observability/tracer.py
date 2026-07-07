"""Structured JSONL tracing for agent executions."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field

from app.config import get_settings


class ToolCallTrace(BaseModel):
    """Record of a single tool invocation within an agent run."""

    name: str
    args: dict[str, Any] = Field(default_factory=dict)
    latency_ms: int = Field(ge=0)


class RetrievedChunkTrace(BaseModel):
    """A document chunk returned by the RAG retriever."""

    source: str
    page: int = Field(ge=0)
    text: str
    score: float


class ExecutionTrace(BaseModel):
    """Full structured trace for one /agent execution."""

    trace_id: str
    timestamp: str
    query: str
    raw_intent: str | None = None
    tool_calls: list[ToolCallTrace] = Field(default_factory=list)
    retrieved_chunks: list[RetrievedChunkTrace] = Field(default_factory=list)
    latency_ms: int = Field(ge=0)
    model_provider: str
    model_name: str


def new_trace_id() -> str:
    """Generate a unique trace identifier."""
    return str(uuid4())


def utc_now_iso() -> str:
    """Return the current UTC time in ISO-8601 format."""
    return datetime.now(timezone.utc).isoformat()


def write_trace(trace: ExecutionTrace, log_path: Path | None = None) -> Path:
    """Append one execution trace as a JSON line to the log file."""
    settings = get_settings()
    path = log_path or settings.resolved_log_path
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("a", encoding="utf-8") as log_file:
        log_file.write(trace.model_dump_json())
        log_file.write("\n")

    return path


def read_recent_traces(limit: int = 20, log_path: Path | None = None) -> list[ExecutionTrace]:
    """Read the most recent traces from the log file (newest last)."""
    if limit < 1:
        return []

    settings = get_settings()
    path = log_path or settings.resolved_log_path
    if not path.exists():
        return []

    traces: list[ExecutionTrace] = []
    with path.open("r", encoding="utf-8") as log_file:
        for line in log_file:
            line = line.strip()
            if not line:
                continue
            traces.append(ExecutionTrace.model_validate(json.loads(line)))

    return traces[-limit:]
