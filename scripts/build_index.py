"""Build a local FAISS index from policy PDFs in data/policies/."""

from __future__ import annotations

import json
import pickle
import sys
from datetime import datetime, timezone
from pathlib import Path

import faiss
import numpy as np
from pypdf import PdfReader
from rank_bm25 import BM25Okapi

from app.config import Settings, get_settings
from app.rag.hybrid import tokenize
from app.services.embedding_factory import EmbeddingService

CHUNK_SIZE = 800
CHUNK_OVERLAP = 150


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into overlapping character windows."""
    cleaned = " ".join(text.split())
    if not cleaned:
        return []

    if len(cleaned) <= chunk_size:
        return [cleaned]

    chunks: list[str] = []
    start = 0
    stride = max(chunk_size - overlap, 1)

    while start < len(cleaned):
        chunk = cleaned[start : start + chunk_size].strip()
        if chunk:
            chunks.append(chunk)
        if start + chunk_size >= len(cleaned):
            break
        start += stride

    return chunks


def extract_chunks_from_pdf(pdf_path: Path) -> list[dict[str, object]]:
    """Extract page-level text chunks from a PDF file."""
    reader = PdfReader(str(pdf_path))
    records: list[dict[str, object]] = []

    for page_number, page in enumerate(reader.pages, start=1):
        page_text = page.extract_text() or ""
        for chunk in chunk_text(page_text):
            records.append(
                {
                    "source_file": pdf_path.name,
                    "page": page_number,
                    "text": chunk,
                }
            )

    return records


def collect_policy_chunks(policies_dir: Path) -> list[dict[str, object]]:
    """Read and chunk every PDF in the policies directory."""
    pdf_files = sorted(policies_dir.glob("*.pdf"))
    if not pdf_files:
        raise FileNotFoundError(f"No PDF files found in {policies_dir}")

    chunks: list[dict[str, object]] = []
    for pdf_path in pdf_files:
        chunks.extend(extract_chunks_from_pdf(pdf_path))

    for chunk_id, chunk in enumerate(chunks):
        chunk["chunk_id"] = chunk_id

    return chunks


def build_faiss_index(embeddings: list[list[float]]) -> faiss.IndexFlatIP:
    """Create a cosine-similarity FAISS index from embedding vectors."""
    if not embeddings:
        raise ValueError("Cannot build a FAISS index without embeddings.")

    vectors = np.array(embeddings, dtype=np.float32)
    faiss.normalize_L2(vectors)

    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    return index


def save_index(
    index: faiss.Index,
    chunks: list[dict[str, object]],
    *,
    index_path: Path,
    chunks_path: Path,
    metadata: dict[str, object] | None = None,
) -> None:
    """Persist the FAISS index and chunk metadata to disk."""
    index_path.parent.mkdir(parents=True, exist_ok=True)
    chunks_path.parent.mkdir(parents=True, exist_ok=True)

    faiss.write_index(index, str(index_path))

    payload = {
        "built_at": datetime.now(timezone.utc).isoformat(),
        "chunk_count": len(chunks),
        "chunks": chunks,
    }
    if metadata:
        payload.update(metadata)

    chunks_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_index(
    index_path: Path,
    chunks_path: Path,
) -> tuple[faiss.Index, list[dict[str, object]], dict[str, object]]:
    """Load a persisted FAISS index and chunk metadata."""
    if not index_path.exists():
        raise FileNotFoundError(f"FAISS index not found: {index_path}")
    if not chunks_path.exists():
        raise FileNotFoundError(f"Chunk metadata not found: {chunks_path}")

    index = faiss.read_index(str(index_path))
    payload = json.loads(chunks_path.read_text(encoding="utf-8"))
    chunks = payload.get("chunks", payload)
    if not isinstance(chunks, list):
        raise ValueError("Chunk metadata must contain a list under 'chunks'.")

    return index, chunks, payload


def build_bm25_index(chunks: list[dict[str, object]]) -> BM25Okapi:
    """Create a BM25 sparse index from chunked policy text."""
    if not chunks:
        raise ValueError("Cannot build a BM25 index without chunks.")

    corpus = [tokenize(str(chunk["text"])) for chunk in chunks]
    return BM25Okapi(corpus)


def save_bm25_index(bm25_index: BM25Okapi, bm25_path: Path) -> None:
    """Persist the BM25 index to disk."""
    bm25_path.parent.mkdir(parents=True, exist_ok=True)
    bm25_path.write_bytes(pickle.dumps(bm25_index))


def load_bm25_index(bm25_path: Path) -> BM25Okapi:
    """Load a persisted BM25 index from disk."""
    if not bm25_path.exists():
        raise FileNotFoundError(f"BM25 index not found: {bm25_path}")

    return pickle.loads(bm25_path.read_bytes())


def build_policy_index(
    settings: Settings | None = None,
    embedding_service: EmbeddingService | None = None,
) -> dict[str, object]:
    """End-to-end index build from PDFs to storage artifacts."""
    settings = settings or get_settings()
    embedding_service = embedding_service or EmbeddingService(settings=settings)

    chunks = collect_policy_chunks(settings.policies_dir)
    texts = [str(chunk["text"]) for chunk in chunks]
    embeddings = embedding_service.embed_texts(texts, task_type="retrieval_document")
    index = build_faiss_index(embeddings)
    bm25_index = build_bm25_index(chunks)

    metadata = {
        "embedding_provider": settings.model_provider,
        "embedding_model": settings.resolved_embedding_model_name,
        "vector_dimension": len(embeddings[0]),
        "retrieval_mode": "hybrid_rrf",
    }
    save_index(
        index,
        chunks,
        index_path=settings.faiss_index_path,
        chunks_path=settings.chunks_path,
        metadata=metadata,
    )
    save_bm25_index(bm25_index, settings.bm25_index_path)

    return {
        "chunk_count": len(chunks),
        "index_path": str(settings.faiss_index_path),
        "bm25_index_path": str(settings.bm25_index_path),
        "chunks_path": str(settings.chunks_path),
        **metadata,
    }


def main() -> int:
    """CLI entrypoint for building the policy index."""
    try:
        summary = build_policy_index()
    except (FileNotFoundError, ValueError) as error:
        print(f"Index build failed: {error}", file=sys.stderr)
        return 1

    print("Policy index built successfully.")
    print(f"  Chunks: {summary['chunk_count']}")
    print(f"  Provider: {summary['embedding_provider']}")
    print(f"  Model: {summary['embedding_model']}")
    print(f"  Dimension: {summary['vector_dimension']}")
    print(f"  Index: {summary['index_path']}")
    print(f"  BM25: {summary['bm25_index_path']}")
    print(f"  Metadata: {summary['chunks_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
