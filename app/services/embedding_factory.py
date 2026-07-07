"""Provider-agnostic embedding service for RAG indexing and retrieval."""

from __future__ import annotations

from typing import TYPE_CHECKING

from app.config import Settings, get_settings

if TYPE_CHECKING:
    from openai import OpenAI


class EmbeddingService:
    """Generate text embeddings via Gemini or OpenAI based on environment config."""

    DEFAULT_BATCH_SIZE = 64

    def __init__(self, settings: Settings | None = None, batch_size: int | None = None) -> None:
        self.settings = settings or get_settings()
        self.provider = self.settings.model_provider
        self.model_name = self.settings.resolved_embedding_model_name
        self.batch_size = batch_size or self.DEFAULT_BATCH_SIZE
        self._openai_client: OpenAI | None = None

        if self.provider == "gemini":
            self._configure_gemini()
        elif self.provider == "openai":
            self._configure_openai()
        else:
            raise ValueError(f"Unsupported MODEL_PROVIDER: {self.provider}")

    def _configure_gemini(self) -> None:
        if not self.settings.gemini_api_key:
            raise ValueError("GEMINI_API_KEY is required when MODEL_PROVIDER=gemini")

        import google.generativeai as genai

        genai.configure(api_key=self.settings.gemini_api_key)

    def _configure_openai(self) -> None:
        if not self.settings.openai_api_key:
            raise ValueError("OPENAI_API_KEY is required when MODEL_PROVIDER=openai")

    @property
    def openai_client(self) -> OpenAI:
        if self._openai_client is None:
            from openai import OpenAI

            self._openai_client = OpenAI(api_key=self.settings.openai_api_key)
        return self._openai_client

    def embed_texts(self, texts: list[str], *, task_type: str = "retrieval_document") -> list[list[float]]:
        """Embed a list of text chunks, processing in batches."""
        if not texts:
            return []

        embeddings: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            if self.provider == "gemini":
                embeddings.extend(self._embed_batch_gemini(batch, task_type=task_type))
            else:
                embeddings.extend(self._embed_batch_openai(batch))
        return embeddings

    def embed_text(self, text: str, *, task_type: str = "retrieval_query") -> list[float]:
        """Embed a single text string."""
        return self.embed_texts([text], task_type=task_type)[0]

    def _embed_batch_gemini(self, texts: list[str], *, task_type: str) -> list[list[float]]:
        import google.generativeai as genai

        model = f"models/{self.model_name}"
        result = genai.embed_content(model=model, content=texts, task_type=task_type)
        return self._normalize_gemini_embeddings(result, expected_count=len(texts))

    def _embed_batch_openai(self, texts: list[str]) -> list[list[float]]:
        response = self.openai_client.embeddings.create(
            model=self.model_name,
            input=texts,
        )
        return [list(item.embedding) for item in response.data]

    @staticmethod
    def _normalize_gemini_embeddings(result: object, *, expected_count: int) -> list[list[float]]:
        """Normalize Gemini embed_content responses for single or batched inputs."""
        if isinstance(result, dict):
            if "embedding" in result:
                embedding = result["embedding"]
                if embedding and isinstance(embedding[0], (int, float)):
                    return [list(embedding)]
                return [list(vector) for vector in embedding]

            if "embeddings" in result:
                return [list(item["values"]) for item in result["embeddings"]]

        embedding_attr = getattr(result, "embedding", None)
        if embedding_attr is not None:
            if embedding_attr and isinstance(embedding_attr[0], (int, float)):
                return [list(embedding_attr)]
            return [list(vector) for vector in embedding_attr]

        embeddings_attr = getattr(result, "embeddings", None)
        if embeddings_attr is not None:
            return [list(item.values) for item in embeddings_attr]

        raise ValueError(
            f"Unexpected Gemini embedding response shape (expected {expected_count} vectors)."
        )


def get_embedding_service(settings: Settings | None = None) -> EmbeddingService:
    """Factory helper for a configured embedding service."""
    return EmbeddingService(settings=settings)
