"""Ragas evaluation module for measuring RAG system quality.

Computes four Ragas metrics on a (question, answer, contexts, ground_truth) tuple:
  - Faithfulness: fraction of claims in the answer grounded in context.
  - Answer Relevance: how well the answer addresses the question.
  - Context Precision: precision of the retrieved context chunks.
  - Context Recall: recall of relevant information from ground truth.

Note: Ragas metrics require an LLM judge; this module reuses the Groq API
key already configured in Settings via the ``langchain-groq`` adapter.
"""

from __future__ import annotations

from dataclasses import dataclass

import structlog
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
)

from app.core.config import get_settings
from app.schemas.query import EvalScores

logger = structlog.get_logger(__name__)


@dataclass(slots=True)
class EvalInput:
    """Input data for a single Ragas evaluation.

    Attributes:
        question: The user question.
        answer: The LLM-generated answer to evaluate.
        contexts: List of context strings used to generate the answer.
        ground_truth: Reference answer for context recall computation.
    """

    question: str
    answer: str
    contexts: list[str]
    ground_truth: str = ""


async def evaluate_rag(sample: EvalInput) -> EvalScores:
    """Run Ragas metrics on a single RAG output sample.

    Args:
        sample: The RAG input/output pair to evaluate.

    Returns:
        ``EvalScores`` with faithfulness, answer_relevance, context_precision,
        and context_recall. Any metric that fails returns ``None``.
    """
    settings = get_settings()

    data = Dataset.from_dict(
        {
            "question": [sample.question],
            "answer": [sample.answer],
            "contexts": [sample.contexts],
            "ground_truth": [sample.ground_truth or sample.question],
        }
    )

    # Build Groq-backed LangChain LLM for Ragas judge
    try:
        from langchain_groq import ChatGroq  # type: ignore[import-untyped]
        llm_judge = ChatGroq(
            api_key=settings.GROQ_API_KEY,
            model_name="llama3-8b-8192",  # Smaller model for eval to save quota
            temperature=0,
        )
    except ImportError as exc:
        logger.warning(
            "langchain-groq not installed — skipping Ragas eval",
            error=str(exc),
        )
        return EvalScores()

    logger.info("Running Ragas evaluation", question_preview=sample.question[:60])

    try:
        result = evaluate(
            dataset=data,
            metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
            llm=llm_judge,
            raise_exceptions=False,
        )
        scores_df = result.to_pandas()
        row = scores_df.iloc[0]

        scores = EvalScores(
            faithfulness=_safe_float(row.get("faithfulness")),
            answer_relevance=_safe_float(row.get("answer_relevancy")),
            context_precision=_safe_float(row.get("context_precision")),
            context_recall=_safe_float(row.get("context_recall")),
        )
        logger.info("Ragas evaluation complete", scores=scores)
        return scores

    except Exception as exc:
        logger.error("Ragas evaluation failed", error=str(exc))
        return EvalScores()


def check_thresholds(scores: EvalScores) -> tuple[bool, list[str]]:
    """Validate Ragas scores against configured CI thresholds.

    Args:
        scores: The computed Ragas metrics.

    Returns:
        Tuple of (passed: bool, failures: list[str]) where ``failures``
        contains human-readable descriptions of any threshold violations.
    """
    settings = get_settings()
    failures: list[str] = []

    checks = [
        ("faithfulness", scores.faithfulness, settings.RAGAS_FAITHFULNESS_THRESHOLD),
        ("answer_relevance", scores.answer_relevance, settings.RAGAS_RELEVANCE_THRESHOLD),
        (
            "context_precision",
            scores.context_precision,
            settings.RAGAS_CONTEXT_PRECISION_THRESHOLD,
        ),
        (
            "context_recall",
            scores.context_recall,
            settings.RAGAS_CONTEXT_RECALL_THRESHOLD,
        ),
    ]

    for metric, value, threshold in checks:
        if value is None:
            continue  # Skip metrics that could not be computed
        if value < threshold:
            failures.append(
                f"{metric}: {value:.3f} < threshold {threshold:.3f}"
            )

    return len(failures) == 0, failures


def _safe_float(value: object) -> float | None:
    """Coerce a value to float, returning None on failure."""
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
