"""Application configuration loaded from environment variables."""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

ModelProvider = Literal["gemini", "openai"]
DeploymentTarget = Literal["local", "docker", "cloud-run"]

_DEFAULT_CHAT_MODELS: dict[ModelProvider, str] = {
    "gemini": "gemini-2.5-flash",
    "openai": "gpt-4o-mini",
}

_DEFAULT_EMBEDDING_MODELS: dict[ModelProvider, str] = {
    "gemini": "gemini-embedding-001",
    "openai": "text-embedding-3-small",
}


class Settings(BaseSettings):
    """Central configuration for PolicyPulse."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    model_provider: ModelProvider = "gemini"
    model_name: str | None = None

    gemini_api_key: str = ""
    openai_api_key: str = ""

    deployment_target: DeploymentTarget = "local"
    backend_url: str = "http://localhost:8000"

    faiss_index_path: Path = Field(default=Path("storage/faiss.index"))
    bm25_index_path: Path = Field(default=Path("storage/bm25.pkl"))
    chunks_path: Path = Field(default=Path("storage/chunks.json"))
    log_path: Path = Field(default=Path("structured_logs.jsonl"))
    policies_dir: Path = Field(default=Path("data/policies"))

    max_tool_iterations: int = Field(default=5, ge=1, le=20)

    @field_validator("model_provider", mode="before")
    @classmethod
    def normalize_provider(cls, value: str) -> str:
        return value.strip().lower()

    @field_validator("deployment_target", mode="before")
    @classmethod
    def normalize_deployment_target(cls, value: str) -> str:
        normalized = value.strip().lower().replace("_", "-")
        if normalized == "cloudrun":
            normalized = "cloud-run"
        if normalized not in {"local", "docker", "cloud-run"}:
            raise ValueError("deployment_target must be one of: local, docker, cloud-run")
        return normalized

    @property
    def resolved_model_name(self) -> str:
        if self.model_name:
            return self.model_name
        return _DEFAULT_CHAT_MODELS[self.model_provider]

    @property
    def resolved_embedding_model_name(self) -> str:
        return _DEFAULT_EMBEDDING_MODELS[self.model_provider]

    @property
    def resolved_log_path(self) -> Path:
        """Return a writable trace path for the active deployment target."""
        if self.log_path.is_absolute():
            return self.log_path
        if self.deployment_target == "cloud-run":
            return Path("/tmp") / self.log_path
        return self.log_path


@lru_cache
def get_settings() -> Settings:
    """Return a cached Settings instance."""
    return Settings()
