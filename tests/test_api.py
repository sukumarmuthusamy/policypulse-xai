"""API route tests for the PolicyPulse FastAPI application."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from app.agents.schemas import AgentRequest, AgentResponse
from app.main import app
from app.observability.tracer import ExecutionTrace


@pytest.fixture
def mock_index_store(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    store = MagicMock()
    store.is_loaded = True
    store._index = MagicMock(ntotal=12)
    store._metadata = {
        "built_at": "2026-07-06T06:00:00+00:00",
        "chunk_count": 12,
    }
    store.bm25_index = MagicMock()
    store.load = MagicMock()
    store.reload = MagicMock()
    monkeypatch.setattr("app.main.get_policy_index_store", lambda: store)
    return store


@pytest.fixture
def client(mock_index_store: MagicMock, monkeypatch: pytest.MonkeyPatch, tmp_path):
    chunks_path = tmp_path / "chunks.json"
    chunks_path.write_text(
        '{"built_at": "2026-07-06T06:00:00+00:00", "chunk_count": 12, "chunks": []}',
        encoding="utf-8",
    )
    monkeypatch.setenv("CHUNKS_PATH", str(chunks_path))
    monkeypatch.setenv("LOG_PATH", str(tmp_path / "structured_logs.jsonl"))

    from app.config import get_settings

    get_settings.cache_clear()

    with TestClient(app) as test_client:
        yield test_client

    get_settings.cache_clear()


def test_health_returns_ok(client: TestClient) -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_metadata_returns_telemetry(client: TestClient) -> None:
    response = client.get("/metadata")

    assert response.status_code == 200
    payload = response.json()
    assert payload["index_ready"] is True
    assert payload["chunk_count"] == 12
    assert payload["vector_count"] == 12
    assert payload["index_built_at"] == "2026-07-06T06:00:00+00:00"
    assert "model_provider" in payload
    assert "model_name" in payload
    assert "deployment_target" in payload


def test_agent_returns_mocked_response(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    trace = ExecutionTrace(
        trace_id="trace-test-1",
        timestamp="2026-07-06T06:00:00+00:00",
        query="What is the remote work policy?",
        raw_intent="policy_lookup",
        latency_ms=1500,
        model_provider="gemini",
        model_name="gemini-2.5-flash",
    )
    mocked_response = AgentResponse(
        answer="Remote work is allowed up to three days per week.",
        trace=trace,
    )

    def fake_run_agent(request: AgentRequest) -> AgentResponse:
        assert request.query == "What is the remote work policy?"
        return mocked_response

    monkeypatch.setattr("app.main.run_agent", fake_run_agent)

    response = client.post(
        "/agent",
        json={"query": "What is the remote work policy?"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["answer"] == "Remote work is allowed up to three days per week."
    assert payload["trace"]["trace_id"] == "trace-test-1"
    assert payload["trace"]["raw_intent"] == "policy_lookup"


def test_agent_rejects_empty_query(client: TestClient) -> None:
    response = client.post("/agent", json={"query": ""})

    assert response.status_code == 422


def test_agent_returns_503_when_index_not_ready(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    store = MagicMock()
    store.load = MagicMock(side_effect=FileNotFoundError("FAISS index not found"))
    monkeypatch.setattr("app.main.get_policy_index_store", lambda: store)
    monkeypatch.setenv("CHUNKS_PATH", str(tmp_path / "chunks.json"))
    monkeypatch.setenv("LOG_PATH", str(tmp_path / "structured_logs.jsonl"))

    from app.config import get_settings

    get_settings.cache_clear()

    with TestClient(app) as test_client:
        metadata = test_client.get("/metadata").json()
        assert metadata["index_ready"] is False

        response = test_client.post("/agent", json={"query": "What is PTO policy?"})

    get_settings.cache_clear()

    assert response.status_code == 503
    assert "Policy index is not loaded" in response.json()["detail"]


def test_upload_rebuilds_index_and_reloads_state(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    mock_index_store: MagicMock,
    tmp_path,
) -> None:
    policies_dir = tmp_path / "policies"
    policies_dir.mkdir()
    monkeypatch.setenv("POLICIES_DIR", str(policies_dir))

    from app.config import get_settings

    get_settings.cache_clear()

    chunks_path = tmp_path / "chunks.json"
    chunks_path.write_text(
        '{"built_at": "2026-07-06T07:00:00+00:00", "chunk_count": 20, "chunks": []}',
        encoding="utf-8",
    )
    monkeypatch.setenv("CHUNKS_PATH", str(chunks_path))
    get_settings.cache_clear()

    mock_index_store._index = MagicMock(ntotal=20)
    monkeypatch.setattr("app.main.build_policy_index", lambda: {"chunk_count": 20})

    pdf_bytes = b"%PDF-1.4 test content"
    response = client.post(
        "/upload",
        files={"file": ("new_policy.pdf", pdf_bytes, "application/pdf")},
    )

    get_settings.cache_clear()

    assert response.status_code == 200
    payload = response.json()
    assert payload["filename"] == "new_policy.pdf"
    assert payload["chunk_count"] == 20
    assert payload["vector_count"] == 20
    assert (policies_dir / "new_policy.pdf").exists()
    mock_index_store.reload.assert_called_once()


def test_upload_rejects_non_pdf(client: TestClient) -> None:
    response = client.post(
        "/upload",
        files={"file": ("notes.txt", b"hello", "text/plain")},
    )

    assert response.status_code == 400
    assert "Only PDF files are supported" in response.json()["detail"]
