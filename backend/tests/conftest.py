"""Pytest configuration and shared fixtures for the AskMyDocs test suite.

Uses an in-memory SQLite database (via aiosqlite) for fast, isolated unit tests
that do not require a live PostgreSQL + pgvector instance.

Fixtures:
  - event_loop: Module-scoped async event loop.
  - mock_settings: Overrides settings with safe test values.
  - db_session: Provides a clean AsyncSession backed by SQLite per test.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
from unittest.mock import AsyncMock, patch

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.core.config import Settings


# ── Event Loop ────────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def event_loop():
    """Provide a module-scoped event loop for all async tests."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


# ── Settings Override ─────────────────────────────────────────────────────────

TEST_SETTINGS = {
    "DATABASE_URL": "sqlite+aiosqlite:///:memory:",
    "GROQ_API_KEY": "gsk_test_key",
    "GROQ_MODEL": "llama3-8b-8192",
    "EMBEDDING_MODEL": "BAAI/bge-small-en-v1.5",
    "RERANKER_MODEL": "cross-encoder/ms-marco-MiniLM-L-6-v2",
    "CHUNK_SIZE": "256",
    "CHUNK_OVERLAP": "32",
    "TOP_K_RETRIEVAL": "5",
    "TOP_N_RERANK": "2",
    "CORS_ORIGINS": "http://localhost:5173",
    "DEBUG": "false",
    "LOG_LEVEL": "ERROR",
}


@pytest.fixture(scope="session")
def mock_settings() -> Settings:
    """Return a Settings instance with safe test values."""
    return Settings(**TEST_SETTINGS)


# ── Database Fixture ──────────────────────────────────────────────────────────

@pytest_asyncio.fixture
async def db_session() -> AsyncGenerator[AsyncSession, None]:
    """Provide an isolated async SQLite session per test.

    Creates all tables fresh for each test and rolls back any uncommitted
    state on teardown.
    """
    from app.models.document import Base

    engine = create_async_engine("sqlite+aiosqlite:///:memory:", echo=False)

    # SQLite does not support pgvector — patch Vector column type
    # to use a plain Text column for test compatibility
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    factory = async_sessionmaker(engine, expire_on_commit=False)
    async with factory() as session:
        yield session

    await engine.dispose()


# ── Mock Embedder Fixture ─────────────────────────────────────────────────────

@pytest.fixture
def mock_embedder():
    """Return a mock embedder that produces deterministic zero vectors."""
    with patch("app.services.embedder.embed_text", new_callable=AsyncMock) as m:
        m.return_value = [0.0] * 384
        with patch("app.services.embedder.embed_batch", new_callable=AsyncMock) as mb:
            mb.return_value = [[0.0] * 384]
            yield m, mb


# ── Mock LLM Fixture ──────────────────────────────────────────────────────────

@pytest.fixture
def mock_llm():
    """Return a mock LLM that returns a canned cited answer."""
    with patch("app.services.llm.generate_answer", new_callable=AsyncMock) as m:
        m.return_value = "The answer is documented in [S1] and further detailed in [S2]."
        yield m
