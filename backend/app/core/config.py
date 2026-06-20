"""Application configuration via Pydantic BaseSettings.

All values are read from environment variables (or a .env file during local dev).
Import the singleton via: ``from app.core.config import get_settings``
"""

from __future__ import annotations

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Centralised, validated configuration for AskMyDocs.

    Attributes:
        DATABASE_URL: Async-compatible PostgreSQL URL (asyncpg driver).
        GROQ_API_KEY: Secret key for the Groq inference API.
        GROQ_MODEL: Groq model identifier (default: llama3-70b-8192).
        EMBEDDING_MODEL: HuggingFace model for dense embeddings.
        RERANKER_MODEL: HuggingFace cross-encoder for reranking.
        CHUNK_SIZE: Maximum tokens per document chunk.
        CHUNK_OVERLAP: Overlapping tokens between consecutive chunks.
        TOP_K_RETRIEVAL: Candidates fetched from each search arm.
        TOP_N_RERANK: Final context chunks passed to the LLM.
        RRF_K: Constant k in the RRF formula (default 60).
        CORS_ORIGINS: Comma-separated allowed origins (Vercel URL).
        RAGAS_FAITHFULNESS_THRESHOLD: CI gate minimum faithfulness score.
        RAGAS_RELEVANCE_THRESHOLD: CI gate minimum answer relevance score.
        RAGAS_CONTEXT_PRECISION_THRESHOLD: CI gate minimum context precision.
        RAGAS_CONTEXT_RECALL_THRESHOLD: CI gate minimum context recall.
        DEBUG: Enable SQLAlchemy echo and verbose logging.
        LOG_LEVEL: Python logging level string.
        APP_VERSION: Semantic version exposed in /health.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Database ──────────────────────────────────────────────────────────
    DATABASE_URL: str = Field(
        ...,
        description="postgresql+asyncpg://user:pass@host/db?sslmode=require",
    )

    # ── LLM ───────────────────────────────────────────────────────────────
    GROQ_API_KEY: str = Field(..., description="Groq API secret key")
    GROQ_MODEL: str = Field(default="llama-3.3-70b-versatile")

    # ── Embedding & Reranking ─────────────────────────────────────────────
    EMBEDDING_MODEL: str = Field(default="BAAI/bge-small-en-v1.5")
    RERANKER_MODEL: str = Field(default="cross-encoder/ms-marco-MiniLM-L-6-v2")

    # ── Chunking ──────────────────────────────────────────────────────────
    # Parent chunks provide broad context to the LLM; child chunks are smaller,
    # precision-focused units that are embedded and searched. Retrieved children
    # are expanded back to their parent for generation (parent-child retrieval).
    CHUNK_SIZE: int = Field(default=512, ge=64, le=2048)
    CHUNK_OVERLAP: int = Field(default=64, ge=0, le=256)
    CHILD_CHUNK_SIZE: int = Field(default=256, ge=32, le=1024)
    CHILD_CHUNK_OVERLAP: int = Field(default=32, ge=0, le=256)

    # ── Retrieval ─────────────────────────────────────────────────────────
    TOP_K_RETRIEVAL: int = Field(default=20, ge=5, le=100)
    TOP_N_RERANK: int = Field(default=3, ge=1, le=10)
    RRF_K: int = Field(default=60, description="Constant k for Reciprocal Rank Fusion")

    # ── Ragas Evaluation Thresholds ───────────────────────────────────────
    RAGAS_FAITHFULNESS_THRESHOLD: float = Field(default=0.70, ge=0.0, le=1.0)
    RAGAS_RELEVANCE_THRESHOLD: float = Field(default=0.70, ge=0.0, le=1.0)
    RAGAS_CONTEXT_PRECISION_THRESHOLD: float = Field(default=0.65, ge=0.0, le=1.0)
    RAGAS_CONTEXT_RECALL_THRESHOLD: float = Field(default=0.65, ge=0.0, le=1.0)

    # ── CORS ──────────────────────────────────────────────────────────────
    CORS_ORIGINS: str = Field(
        default="http://localhost:5173",
        description="Comma-separated allowed CORS origins",
    )

    # ── Server ────────────────────────────────────────────────────────────
    DEBUG: bool = Field(default=False)
    LOG_LEVEL: str = Field(default="INFO")
    APP_VERSION: str = Field(default="1.0.0")

    @property
    def cors_origins_list(self) -> list[str]:
        """Return CORS origins as a parsed list of stripped strings."""
        return [o.strip() for o in self.CORS_ORIGINS.split(",") if o.strip()]


@lru_cache
def get_settings() -> Settings:
    """Return a cached singleton of the application settings.

    Using ``@lru_cache`` ensures the .env file is parsed only once for the
    lifetime of the process.
    """
    return Settings()
