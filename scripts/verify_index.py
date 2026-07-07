"""Verify a built FAISS index can be loaded and searched locally."""

from __future__ import annotations

import sys
from pathlib import Path

import faiss
import numpy as np

from app.config import get_settings
from app.services.embedding_factory import get_embedding_service
from scripts.build_index import load_index


def main() -> int:
    settings = get_settings()

    try:
        index, chunks, payload = load_index(settings.faiss_index_path, settings.chunks_path)
    except FileNotFoundError as error:
        print(f"Verification failed: {error}", file=sys.stderr)
        print("Run `python scripts/build_index.py` after adding PDFs to data/policies/.")
        return 1

    print("Index verification passed.")
    print(f"  Chunks: {payload.get('chunk_count', len(chunks))}")
    print(f"  Built at: {payload.get('built_at', 'unknown')}")
    print(f"  Provider: {payload.get('embedding_provider', 'unknown')}")
    print(f"  Model: {payload.get('embedding_model', 'unknown')}")
    print(f"  Vectors: {index.ntotal}")

    if not chunks:
        print("  Warning: chunk metadata is empty.")
        return 0

    sample_query = "remote work policy"
    embedder = get_embedding_service()
    query_vector = np.array([embedder.embed_text(sample_query)], dtype=np.float32)
    faiss.normalize_L2(query_vector)

    scores, indices = index.search(query_vector, k=min(3, index.ntotal))
    print(f"  Sample query: {sample_query!r}")

    for rank, (score, chunk_index) in enumerate(zip(scores[0], indices[0], strict=False), start=1):
        if chunk_index < 0:
            continue
        chunk = chunks[chunk_index]
        preview = str(chunk.get("text", ""))[:120]
        print(
            f"    #{rank} score={score:.4f} "
            f"source={chunk.get('source_file')} page={chunk.get('page')} "
            f"text={preview!r}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
