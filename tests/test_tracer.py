"""Unit tests for structured JSONL tracing."""

import json
from pathlib import Path

import pytest

from app.observability.tracer import (
    ExecutionTrace,
    RetrievedChunkTrace,
    ToolCallTrace,
    new_trace_id,
    read_recent_traces,
    utc_now_iso,
    write_trace,
)


def test_write_trace_appends_valid_jsonl(tmp_path: Path) -> None:
    log_path = tmp_path / "structured_logs.jsonl"
    trace = ExecutionTrace(
        trace_id=new_trace_id(),
        timestamp=utc_now_iso(),
        query="What is the remote work policy?",
        raw_intent="policy_lookup",
        tool_calls=[
            ToolCallTrace(
                name="search_policy_documents",
                args={"query": "remote work policy", "top_k": 4},
                latency_ms=42,
            )
        ],
        retrieved_chunks=[
            RetrievedChunkTrace(
                source="handbook.pdf",
                page=3,
                text="Employees may work remotely up to three days per week.",
                score=0.87,
            )
        ],
        latency_ms=1234,
        model_provider="gemini",
        model_name="gemini-2.5-flash",
    )

    written_path = write_trace(trace, log_path=log_path)

    assert written_path == log_path
    assert log_path.exists()

    lines = log_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1

    payload = json.loads(lines[0])
    assert payload["trace_id"] == trace.trace_id
    assert payload["query"] == trace.query
    assert payload["raw_intent"] == "policy_lookup"
    assert payload["tool_calls"][0]["name"] == "search_policy_documents"
    assert payload["tool_calls"][0]["latency_ms"] == 42
    assert payload["retrieved_chunks"][0]["source"] == "handbook.pdf"
    assert payload["retrieved_chunks"][0]["score"] == 0.87
    assert payload["latency_ms"] == 1234
    assert payload["model_provider"] == "gemini"
    assert payload["model_name"] == "gemini-2.5-flash"


def test_write_trace_appends_multiple_lines(tmp_path: Path) -> None:
    log_path = tmp_path / "structured_logs.jsonl"

    first = ExecutionTrace(
        trace_id="trace-1",
        timestamp=utc_now_iso(),
        query="first query",
        latency_ms=100,
        model_provider="openai",
        model_name="gpt-4o-mini",
    )
    second = ExecutionTrace(
        trace_id="trace-2",
        timestamp=utc_now_iso(),
        query="second query",
        latency_ms=200,
        model_provider="openai",
        model_name="gpt-4o-mini",
    )

    write_trace(first, log_path=log_path)
    write_trace(second, log_path=log_path)

    lines = log_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["trace_id"] == "trace-1"
    assert json.loads(lines[1])["trace_id"] == "trace-2"


def test_read_recent_traces_returns_newest_entries(tmp_path: Path) -> None:
    log_path = tmp_path / "structured_logs.jsonl"

    for index in range(3):
        write_trace(
            ExecutionTrace(
                trace_id=f"trace-{index}",
                timestamp=utc_now_iso(),
                query=f"query-{index}",
                latency_ms=index * 10,
                model_provider="gemini",
                model_name="gemini-2.5-flash",
            ),
            log_path=log_path,
        )

    recent = read_recent_traces(limit=2, log_path=log_path)

    assert len(recent) == 2
    assert recent[0].trace_id == "trace-1"
    assert recent[1].trace_id == "trace-2"


def test_read_recent_traces_empty_file(tmp_path: Path) -> None:
    log_path = tmp_path / "missing.jsonl"
    assert read_recent_traces(limit=5, log_path=log_path) == []


@pytest.mark.parametrize("limit", [0, -1])
def test_read_recent_traces_invalid_limit_returns_empty(
    tmp_path: Path, limit: int
) -> None:
    log_path = tmp_path / "structured_logs.jsonl"
    write_trace(
        ExecutionTrace(
            trace_id="trace-0",
            timestamp=utc_now_iso(),
            query="query",
            latency_ms=10,
            model_provider="gemini",
            model_name="gemini-2.5-flash",
        ),
        log_path=log_path,
    )

    assert read_recent_traces(limit=limit, log_path=log_path) == []
