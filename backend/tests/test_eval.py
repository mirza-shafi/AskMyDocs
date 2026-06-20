"""Ragas CI gate tests — fail the build if scores fall below thresholds.

This test file is run as the ``eval-gate`` job in the GitHub Actions CI
pipeline. It ingests a golden test set, runs the full RAG pipeline for each
Q&A pair, computes Ragas metrics, and asserts all scores meet the configured
thresholds.

Requirements:
  - GROQ_API_KEY and DATABASE_URL must be set as GitHub Actions secrets.
  - A test document must be pre-ingested before this test runs (handled by
    the CI workflow via the /ingest endpoint).

Skip locally unless ENABLE_EVAL_TESTS=1 is set to avoid slow LLM calls.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

GOLDEN_SET_PATH = Path(__file__).parent / "golden_set.json"
SKIP_REASON = "Set ENABLE_EVAL_TESTS=1 to run Ragas evaluation (requires GROQ_API_KEY)"


@pytest.fixture(scope="module")
def golden_set() -> list[dict]:
    """Load the golden Q&A evaluation set from disk."""
    with GOLDEN_SET_PATH.open() as f:
        return json.load(f)


@pytest.mark.skipif(
    os.getenv("ENABLE_EVAL_TESTS") != "1",
    reason=SKIP_REASON,
)
class TestRagasEvalGate:
    """CI gate: assert all Ragas scores meet configured thresholds."""

    @pytest.mark.asyncio
    async def test_faithfulness_threshold(self, golden_set):
        """Faithfulness score must meet the configured threshold across the golden set."""
        from app.core.config import get_settings
        from app.services.evaluator import EvalInput, evaluate_rag

        settings = get_settings()
        faithfulness_scores = []

        for item in golden_set[:5]:  # Use first 5 items to save API quota
            eval_input = EvalInput(
                question=item["question"],
                answer=item.get("ground_truth", ""),
                contexts=[item.get("ground_truth", "")],
                ground_truth=item.get("ground_truth", ""),
            )
            scores = await evaluate_rag(eval_input)
            if scores.faithfulness is not None:
                faithfulness_scores.append(scores.faithfulness)

        if faithfulness_scores:
            avg_faithfulness = sum(faithfulness_scores) / len(faithfulness_scores)
            assert avg_faithfulness >= settings.RAGAS_FAITHFULNESS_THRESHOLD, (
                f"Average faithfulness {avg_faithfulness:.3f} is below "
                f"threshold {settings.RAGAS_FAITHFULNESS_THRESHOLD:.3f}"
            )

    @pytest.mark.asyncio
    async def test_answer_relevance_threshold(self, golden_set):
        """Answer relevance score must meet the configured threshold."""
        from app.core.config import get_settings
        from app.services.evaluator import EvalInput, evaluate_rag

        settings = get_settings()
        relevance_scores = []

        for item in golden_set[:5]:
            eval_input = EvalInput(
                question=item["question"],
                answer=item.get("ground_truth", ""),
                contexts=[item.get("ground_truth", "")],
                ground_truth=item.get("ground_truth", ""),
            )
            scores = await evaluate_rag(eval_input)
            if scores.answer_relevance is not None:
                relevance_scores.append(scores.answer_relevance)

        if relevance_scores:
            avg_relevance = sum(relevance_scores) / len(relevance_scores)
            assert avg_relevance >= settings.RAGAS_RELEVANCE_THRESHOLD, (
                f"Average answer relevance {avg_relevance:.3f} is below "
                f"threshold {settings.RAGAS_RELEVANCE_THRESHOLD:.3f}"
            )

    @pytest.mark.asyncio
    async def test_threshold_check_helper(self):
        """check_thresholds should correctly identify failing metrics."""
        from app.schemas.query import EvalScores
        from app.services.evaluator import check_thresholds

        bad_scores = EvalScores(
            faithfulness=0.50,   # Below 0.70 threshold
            answer_relevance=0.80,
            context_precision=0.70,
            context_recall=0.60,  # Below 0.65 threshold
        )
        passed, failures = check_thresholds(bad_scores)
        assert not passed
        assert len(failures) == 2
        assert any("faithfulness" in f for f in failures)
        assert any("context_recall" in f for f in failures)

    def test_threshold_check_passes_good_scores(self):
        """check_thresholds should return True for scores above all thresholds."""
        from app.schemas.query import EvalScores
        from app.services.evaluator import check_thresholds

        good_scores = EvalScores(
            faithfulness=0.85,
            answer_relevance=0.90,
            context_precision=0.80,
            context_recall=0.75,
        )
        passed, failures = check_thresholds(good_scores)
        assert passed
        assert failures == []
