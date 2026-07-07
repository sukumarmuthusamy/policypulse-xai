"""Agent tools backed by hybrid FAISS + BM25 policy retrieval."""

from __future__ import annotations

from functools import lru_cache
from typing import Any, Callable

import faiss
import numpy as np
from rank_bm25 import BM25Okapi

from app.agents.schemas import ToolDefinition, ToolExecutionResult
from app.config import Settings, get_settings
from app.observability.tracer import RetrievedChunkTrace
from app.rag.hybrid import (
    HYBRID_DENSE_K,
    HYBRID_FINAL_K,
    HYBRID_SPARSE_K,
    normalize_rrf_scores,
    reciprocal_rank_fusion,
    tokenize,
)
from app.services.embedding_factory import EmbeddingService, get_embedding_service
from scripts.build_index import load_bm25_index, load_index

DEFAULT_TOP_K = HYBRID_FINAL_K

SYSTEM_TOOL_DEFINITIONS: list[ToolDefinition] = [
    ToolDefinition(
        name="retrieve_policy_context",
        description=(
            "Search indexed company policy PDFs for passages relevant to the user question. "
            "Use this before answering factual policy questions."
        ),
        parameters={
            "type": "object",
            "description": "Arguments for searching indexed policy documents.",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Natural-language search query derived from the user question.",
                },
                "top_k": {
                    "type": "integer",
                    "description": (
                        "Maximum number of policy passages to return after hybrid fusion. "
                        f"Use {DEFAULT_TOP_K} when the caller does not specify a value."
                    ),
                },
            },
            "required": ["query"],
        },
    )
]


class PolicyIndexStore:
    """In-memory FAISS + BM25 indexes for hybrid retrieval."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self._index: faiss.Index | None = None
        self._bm25: BM25Okapi | None = None
        self._chunks: list[dict[str, object]] | None = None
        self._metadata: dict[str, object] | None = None

    @property
    def is_loaded(self) -> bool:
        return self._index is not None and self._chunks is not None

    @property
    def bm25_index(self) -> BM25Okapi | None:
        return self._bm25

    def load(self) -> None:
        if not self.is_loaded:
            index, chunks, metadata = load_index(
                self.settings.faiss_index_path,
                self.settings.chunks_path,
            )
            self._index = index
            self._chunks = chunks
            self._metadata = metadata

        if self._bm25 is None:
            try:
                self._bm25 = load_bm25_index(self.settings.bm25_index_path)
            except FileNotFoundError:
                self._bm25 = None

    def reload(self) -> None:
        """Drop cached index data and load the latest artifacts from disk."""
        self._index = None
        self._bm25 = None
        self._chunks = None
        self._metadata = None
        self.load()

    def _search_dense_ranked(self, query: str, *, top_k: int, embedder: EmbeddingService) -> list[int]:
        assert self._index is not None

        query_vector = np.array([embedder.embed_text(query, task_type="retrieval_query")], dtype=np.float32)
        faiss.normalize_L2(query_vector)

        k = min(max(top_k, 1), self._index.ntotal)
        _, indices = self._index.search(query_vector, k=k)
        return [int(chunk_index) for chunk_index in indices[0] if chunk_index >= 0]

    def _search_sparse_ranked(self, query: str, *, top_k: int) -> list[int]:
        assert self._bm25 is not None

        query_tokens = tokenize(query)
        if not query_tokens:
            return []

        scores = self._bm25.get_scores(query_tokens)
        if len(scores) == 0:
            return []

        ranked_indices = np.argsort(scores)[::-1]
        results: list[int] = []
        for chunk_index in ranked_indices:
            if len(results) >= top_k:
                break
            if scores[chunk_index] > 0:
                results.append(int(chunk_index))
        return results

    def _build_chunk_traces(
        self,
        ranked_chunk_ids: list[tuple[int, float]],
    ) -> list[RetrievedChunkTrace]:
        assert self._chunks is not None

        return [
            RetrievedChunkTrace(
                source=str(self._chunks[chunk_id].get("source_file", "unknown")),
                page=int(self._chunks[chunk_id].get("page", 0)),
                text=str(self._chunks[chunk_id].get("text", "")),
                score=float(score),
            )
            for chunk_id, score in ranked_chunk_ids
        ]

    def search(
        self,
        query: str,
        *,
        top_k: int = DEFAULT_TOP_K,
        embedder: EmbeddingService | None = None,
    ) -> list[RetrievedChunkTrace]:
        if not self.is_loaded:
            self.load()

        assert self._index is not None
        assert self._chunks is not None

        if self._index.ntotal == 0:
            return []

        embedder = embedder or get_embedding_service()
        final_k = min(max(top_k, 1), len(self._chunks))

        if self._bm25 is None:
            dense_only = self._search_dense_ranked(query, top_k=final_k, embedder=embedder)
            dense_scores = {
                chunk_id: float(final_k - rank)
                for rank, chunk_id in enumerate(dense_only, start=1)
            }
            normalized = normalize_rrf_scores(dense_scores)
            ranked = [(chunk_id, normalized[chunk_id]) for chunk_id in dense_only]
            return self._build_chunk_traces(ranked)

        dense_ranked = self._search_dense_ranked(query, top_k=HYBRID_DENSE_K, embedder=embedder)
        sparse_ranked = self._search_sparse_ranked(query, top_k=HYBRID_SPARSE_K)
        fused = reciprocal_rank_fusion(
            [dense_ranked, sparse_ranked],
            final_k=final_k,
        )
        normalized = normalize_rrf_scores(dict(fused))
        ranked = [(chunk_id, normalized[chunk_id]) for chunk_id, _ in fused]
        return self._build_chunk_traces(ranked)


@lru_cache
def get_policy_index_store() -> PolicyIndexStore:
    """Return a process-cached policy index store."""
    return PolicyIndexStore()


def _format_retrieved_chunks(chunks: list[RetrievedChunkTrace]) -> str:
    if not chunks:
        return "No relevant policy passages were found in the indexed documents."

    sections: list[str] = []
    for rank, chunk in enumerate(chunks, start=1):
        match_percent = int(round(max(0.0, min(chunk.score, 1.0)) * 100))
        sections.append(
            "\n".join(
                [
                    (
                        f"[Passage {rank}] Source: {chunk.source} | Page: {chunk.page} | "
                        f"Match: {match_percent}%"
                    ),
                    chunk.text,
                ]
            )
        )
    return "\n\n".join(sections)


def retrieve_policy_context(query: str, top_k: int = DEFAULT_TOP_K) -> str:
    """Retrieve hybrid-ranked policy passages and format them for the LLM."""
    chunks = get_policy_index_store().search(query, top_k=top_k)
    return _format_retrieved_chunks(chunks)


def execute_tool(name: str, arguments: dict[str, Any]) -> ToolExecutionResult:
    """Dispatch a tool call by name and return structured output for the orchestrator."""
    if name == "retrieve_policy_context":
        query = str(arguments.get("query", "")).strip()
        if not query:
            return ToolExecutionResult(output="Error: retrieve_policy_context requires a non-empty query.")

        top_k = int(arguments.get("top_k", DEFAULT_TOP_K))
        chunks = get_policy_index_store().search(query, top_k=top_k)
        return ToolExecutionResult(
            output=_format_retrieved_chunks(chunks),
            retrieved_chunks=chunks,
        )

    return ToolExecutionResult(output=f"Error: unknown tool '{name}'.")


def get_openai_tool_schemas() -> list[dict[str, Any]]:
    """Serialize tool definitions for the OpenAI Chat Completions API."""
    return [
        {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters,
            },
        }
        for tool in SYSTEM_TOOL_DEFINITIONS
    ]


def _sanitize_gemini_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Strip OpenAPI fields that Gemini protobuf Schema does not accept."""
    allowed_keys = {"type", "properties", "required", "description", "enum", "items"}
    sanitized: dict[str, Any] = {}

    for key, value in schema.items():
        if key not in allowed_keys:
            continue
        if key == "properties" and isinstance(value, dict):
            sanitized[key] = {
                property_name: _sanitize_gemini_schema(property_schema)
                if isinstance(property_schema, dict)
                else property_schema
                for property_name, property_schema in value.items()
            }
        elif key == "items" and isinstance(value, dict):
            sanitized[key] = _sanitize_gemini_schema(value)
        else:
            sanitized[key] = value

    return sanitized


def get_gemini_tool_declarations() -> list[dict[str, Any]]:
    """Serialize tool definitions for Gemini function calling."""
    return [
        {
            "name": tool.name,
            "description": tool.description,
            "parameters": _sanitize_gemini_schema(tool.parameters),
        }
        for tool in SYSTEM_TOOL_DEFINITIONS
    ]


TOOL_HANDLERS: dict[str, Callable[..., str]] = {
    "retrieve_policy_context": retrieve_policy_context,
}
