"""Tests for hybrid retrieval helpers."""

from __future__ import annotations

from app.rag.hybrid import normalize_rrf_scores, reciprocal_rank_fusion, tokenize


def test_tokenize_extracts_alphanumeric_terms() -> None:
    tokens = tokenize("What's the pay per hour? $80-$105")
    assert "pay" in tokens
    assert "hour" in tokens
    assert "80" in tokens
    assert "105" in tokens


def test_reciprocal_rank_fusion_deduplicates_and_promotes_overlap() -> None:
  # Chunk 2 appears in both lists and should outrank single-list-only items.
    fused = reciprocal_rank_fusion(
        [
            [0, 1, 2],
            [2, 3, 4],
        ],
        final_k=3,
    )

    fused_ids = [chunk_id for chunk_id, _ in fused]
    assert fused_ids[0] == 2
    assert len(fused_ids) == 3
    assert len(set(fused_ids)) == 3


def test_normalize_rrf_scores_scales_to_unit_interval() -> None:
    normalized = normalize_rrf_scores({1: 0.5, 2: 1.0, 3: 0.25})
    assert normalized[2] == 1.0
    assert normalized[1] == 0.5
    assert normalized[3] == 0.25
