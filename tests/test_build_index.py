"""Tests for PDF chunking and FAISS index persistence."""

from __future__ import annotations

from pathlib import Path

import faiss
import numpy as np
import pytest
from pypdf import PdfWriter

from scripts.build_index import (
    build_bm25_index,
    build_faiss_index,
    build_policy_index,
    chunk_text,
    collect_policy_chunks,
    load_bm25_index,
    load_index,
    save_bm25_index,
    save_index,
)
from app.rag.hybrid import tokenize


class FakeEmbeddingService:
    """Deterministic embeddings for offline tests."""

    def __init__(self, settings=None, batch_size: int | None = None, dimension: int = 8) -> None:
        self.dimension = dimension
        self.provider = "test"
        self.model_name = "fake-embedder"

    def embed_texts(self, texts: list[str], *, task_type: str = "retrieval_document") -> list[list[float]]:
        vectors: list[list[float]] = []
        for text in texts:
            seed = sum(ord(char) for char in text)
            vectors.append(
                [((seed + index) % 97) / 97.0 for index in range(self.dimension)]
            )
        return vectors


def test_chunk_text_overlap() -> None:
    text = "A" * 900 + "B" * 900
    chunks = chunk_text(text, chunk_size=800, overlap=150)

    assert len(chunks) >= 2
    assert all(len(chunk) <= 800 for chunk in chunks)
    assert chunks[0].startswith("A")
    assert chunks[-1].endswith("B")


def test_collect_policy_chunks_from_blank_pdf(tmp_path: Path) -> None:
    pdf_path = tmp_path / "remote_work.pdf"
    writer = PdfWriter()
    writer.add_blank_page(width=612, height=792)
    with pdf_path.open("wb") as pdf_file:
        writer.write(pdf_file)

    chunks = collect_policy_chunks(tmp_path)
    assert chunks == []


def test_save_and_load_index_roundtrip(tmp_path: Path) -> None:
    chunks = [
        {"chunk_id": 0, "source_file": "policy.pdf", "page": 1, "text": "Remote work is allowed."},
        {"chunk_id": 1, "source_file": "policy.pdf", "page": 2, "text": "Vacation accrues monthly."},
    ]
    embeddings = FakeEmbeddingService().embed_texts([chunk["text"] for chunk in chunks])
    index = build_faiss_index(embeddings)

    index_path = tmp_path / "faiss.index"
    chunks_path = tmp_path / "chunks.json"
    save_index(
        index,
        chunks,
        index_path=index_path,
        chunks_path=chunks_path,
        metadata={"embedding_provider": "test", "embedding_model": "fake"},
    )

    loaded_index, loaded_chunks, payload = load_index(index_path, chunks_path)

    assert index_path.exists()
    assert chunks_path.exists()
    assert loaded_index.ntotal == 2
    assert len(loaded_chunks) == 2
    assert payload["chunk_count"] == 2
    assert payload["embedding_provider"] == "test"


def test_save_and_load_bm25_roundtrip(tmp_path: Path) -> None:
    chunks = [
        {"chunk_id": 0, "source_file": "pay.pdf", "page": 1, "text": "Pay per hour is $80-$105."},
        {"chunk_id": 1, "source_file": "pay.pdf", "page": 2, "text": "Employers of record handle reimbursements."},
    ]
    bm25_path = tmp_path / "bm25.pkl"
    save_bm25_index(build_bm25_index(chunks), bm25_path)

    loaded = load_bm25_index(bm25_path)
    scores = loaded.get_scores(tokenize("pay per hour"))
    assert int(np.argmax(scores)) == 0


def test_build_policy_index_with_fake_embedder(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from app.config import Settings

    policies_dir = tmp_path / "policies"
    storage_dir = tmp_path / "storage"
    policies_dir.mkdir()
    storage_dir.mkdir()

    settings = Settings(
        policies_dir=policies_dir,
        faiss_index_path=storage_dir / "faiss.index",
        bm25_index_path=storage_dir / "bm25.pkl",
        chunks_path=storage_dir / "chunks.json",
        model_provider="gemini",
        gemini_api_key="test-key",
    )

    fake_chunks = [
        {
            "chunk_id": 0,
            "source_file": "handbook.pdf",
            "page": 1,
            "text": "Employees may work remotely up to three days per week.",
        }
    ]

    monkeypatch.setattr("scripts.build_index.collect_policy_chunks", lambda policies_dir: fake_chunks)
    monkeypatch.setattr("scripts.build_index.EmbeddingService", FakeEmbeddingService)

    summary = build_policy_index(settings=settings)

    assert summary["chunk_count"] == 1
    assert Path(summary["index_path"]).exists()
    assert Path(summary["bm25_index_path"]).exists()
    assert Path(summary["chunks_path"]).exists()

    loaded_index, loaded_chunks, _ = load_index(settings.faiss_index_path, settings.chunks_path)
    assert loaded_index.ntotal == 1
    assert loaded_chunks[0]["text"].startswith("Employees")


def test_faiss_search_returns_scores() -> None:
    chunks = [
        {"chunk_id": 0, "source_file": "a.pdf", "page": 1, "text": "remote work policy"},
        {"chunk_id": 1, "source_file": "a.pdf", "page": 2, "text": "expense reimbursement"},
    ]
    fake = FakeEmbeddingService()
    embeddings = fake.embed_texts([chunk["text"] for chunk in chunks])
    index = build_faiss_index(embeddings)

    query_vector = np.array([fake.embed_texts(["remote work"])[0]], dtype=np.float32)
    faiss.normalize_L2(query_vector)

    scores, indices = index.search(query_vector, k=1)
    assert indices[0][0] in {0, 1}
    assert scores[0][0] > 0
