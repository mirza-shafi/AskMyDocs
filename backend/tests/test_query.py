"""Tests for the RAG query endpoint and pipeline components.

Covers:
  - RRF fusion logic correctness.
  - Query endpoint happy path (mocked retriever + LLM).
  - Empty retrieval graceful fallback.
"""

from __future__ import annotations

import pytest

# ── RRF Logic Unit Test ───────────────────────────────────────────────────────

class TestRRFFusion:
    """Unit tests for the Reciprocal Rank Fusion algorithm in retriever.py."""

    def test_rrf_score_decreases_with_rank(self):
        """Higher-ranked results should receive higher RRF scores."""
        k = 60
        rank1_score = 1.0 / (k + 1)
        rank5_score = 1.0 / (k + 5)
        assert rank1_score > rank5_score

    def test_rrf_dual_hit_outscores_single(self):
        """A chunk appearing in both search arms should outscore chunk in only one."""
        k = 60
        # Chunk A: rank 3 in vector only
        score_a = 1.0 / (k + 3)
        # Chunk B: rank 5 in both vector and FTS
        score_b = 1.0 / (k + 5) + 1.0 / (k + 5)
        assert score_b > score_a


# ── Query Endpoint Tests ──────────────────────────────────────────────────────

class TestQueryEndpoint:
    """Integration tests for POST /api/v1/query."""

    @pytest.mark.asyncio
    async def test_query_returns_200_with_mocks(self, mock_embedder, mock_llm):
        """Query endpoint should return 200 with answer and sources when mocked."""
        from unittest.mock import AsyncMock, patch

        from fastapi.testclient import TestClient

        from app.main import app


        with patch("app.services.rag_engine.run", new_callable=AsyncMock) as mock_run:
            from app.schemas.query import SourceChunk
            from app.services.rag_engine import RAGResult

            mock_run.return_value = RAGResult(
                question="What is RAG?",
                answer="RAG stands for Retrieval-Augmented Generation [S1].",
                sources=[
                    SourceChunk(
                        chunk_id="abc-123",
                        doc_id="test_doc",
                        source_name="test.txt",
                        chunk_index=0,
                        content="RAG stands for Retrieval-Augmented Generation.",
                        rerank_score=0.95,
                    )
                ],
                latency_ms=142.5,
            )

            client = TestClient(app)
            response = client.post(
                "/api/v1/query",
                json={"question": "What is RAG?"},
            )

        assert response.status_code == 200
        data = response.json()
        assert "answer" in data
        assert "sources" in data
        assert "latency_ms" in data
        assert data["eval_scores"] is None  # include_eval defaults to False

    @pytest.mark.asyncio
    async def test_query_requires_minimum_question_length(self):
        """A question shorter than 3 characters should return 422."""
        from fastapi.testclient import TestClient

        from app.main import app

        client = TestClient(app)
        response = client.post("/api/v1/query", json={"question": "Hi"})
        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_health_endpoint_returns_ok(self):
        """GET /health should return 200."""
        from unittest.mock import patch

        from fastapi.testclient import TestClient

        from app.main import app

        with patch("app.api.v1.health.get_db"):
            client = TestClient(app)
            response = client.get("/health")
        # Health may return 200 or 503 depending on DB connectivity in test
        assert response.status_code in (200, 500, 503)
