"""Hybrid retrieval helpers: tokenization and Reciprocal Rank Fusion."""

from __future__ import annotations

import re

RRF_K = 60
HYBRID_DENSE_K = 4
HYBRID_SPARSE_K = 4
HYBRID_FINAL_K = 5


def tokenize(text: str) -> list[str]:
    """Lowercase alphanumeric tokenization for BM25."""
    return re.findall(r"[a-z0-9]+", text.lower())


def reciprocal_rank_fusion(
    ranked_lists: list[list[int]],
    *,
    rrf_k: int = RRF_K,
    final_k: int = HYBRID_FINAL_K,
) -> list[tuple[int, float]]:
    """Merge ranked chunk-id lists with Reciprocal Rank Fusion."""
    fused_scores: dict[int, float] = {}

    for ranked_ids in ranked_lists:
        for rank, chunk_id in enumerate(ranked_ids, start=1):
            fused_scores[chunk_id] = fused_scores.get(chunk_id, 0.0) + (1.0 / (rrf_k + rank))

    ranked = sorted(fused_scores.items(), key=lambda item: item[1], reverse=True)
    return ranked[:final_k]


def normalize_rrf_scores(rrf_scores: dict[int, float]) -> dict[int, float]:
    """Normalize RRF scores to 0-1 for UI confidence display."""
    if not rrf_scores:
        return {}

    max_score = max(rrf_scores.values())
    if max_score <= 0:
        return {chunk_id: 0.0 for chunk_id in rrf_scores}

    return {chunk_id: score / max_score for chunk_id, score in rrf_scores.items()}
