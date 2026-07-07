"""Tests for hybrid policy retrieval in agent tools."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest
from rank_bm25 import BM25Okapi

from app.agents.tools import PolicyIndexStore
from app.rag.hybrid import tokenize
from scripts.build_index import build_bm25_index


class FakeEmbedder:
    def embed_text(self, text: str, *, task_type: str = "retrieval_query") -> list[float]:
        if "pay" in text.lower() or "hour" in text.lower():
            return [1.0, 0.0, 0.0]
        return [0.0, 1.0, 0.0]


def test_hybrid_search_prefers_keyword_match_for_short_pay_query() -> None:
    chunks = [
        {
            "chunk_id": 0,
            "source_file": "pay.pdf",
            "page": 1,
            "text": "Contractor pay per hour ranges from $80-$105 depending on experience.",
        },
        {
            "chunk_id": 1,
            "source_file": "pay.pdf",
            "page": 2,
            "text": "Employers of record manage reimbursements and expense approvals.",
        },
        {
            "chunk_id": 2,
            "source_file": "pay.pdf",
            "page": 3,
            "text": "General reimbursement policy for travel and meals.",
        },
        {
            "chunk_id": 3,
            "source_file": "pay.pdf",
            "page": 4,
            "text": "Employers of record compliance requirements and reporting.",
        },
    ]

    store = PolicyIndexStore()
    store._chunks = chunks
    store._metadata = {"chunk_count": len(chunks)}
    store._bm25 = build_bm25_index(chunks)

    vectors = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.9, 0.1, 0.0],
            [0.8, 0.2, 0.0],
            [0.7, 0.3, 0.0],
        ],
        dtype=np.float32,
    )
    index = MagicMock()
    index.ntotal = 4
    index.search = MagicMock(
        return_value=(
            np.array([[0.95, 0.90, 0.85, 0.80]], dtype=np.float32),
            np.array([[1, 3, 2, 0]], dtype=np.int64),
        )
    )
    store._index = index

    results = store.search("whats the pay per hour?", top_k=5, embedder=FakeEmbedder())
    top_sources = [chunk.text for chunk in results]

    assert any("$80-$105" in text for text in top_sources)
    assert results[0].score <= 1.0
    assert results[0].score > 0
