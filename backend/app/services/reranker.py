"""Cross-encoder reranking service using sentence-transformers.

The ``CrossEncoder`` model rescores (query, passage) pairs with a much higher
precision than bi-encoder similarity, at the cost of processing each pair
individually. Loaded once at startup as a module-level singleton.
"""

from __future__ import annotations

import asyncio

import structlog
from sentence_transformers import CrossEncoder

from app.core.config import get_settings

logger = structlog.get_logger(__name__)

_reranker: CrossEncoder | None = None


def initialise() -> None:
    """Load the cross-encoder model into memory.

    Should be called once from the FastAPI lifespan ``startup`` handler.
    Subsequent calls are no-ops.
    """
    global _reranker  # noqa: PLW0603
    if _reranker is not None:
        return

    settings = get_settings()
    model_name = settings.RERANKER_MODEL
    logger.info("Loading reranker model", model=model_name)
    _reranker = CrossEncoder(model_name, max_length=512)
    logger.info("Reranker model ready", model=model_name)


def _get_reranker() -> CrossEncoder:
    if _reranker is None:
        raise RuntimeError(
            "Reranker model is not initialised. "
            "Call reranker.initialise() during application startup."
        )
    return _reranker


async def rerank(
    query: str,
    passages: list[str],
    top_n: int,
) -> list[tuple[int, float]]:
    """Score and rank passages for relevance to the query.

    Args:
        query: The user question.
        passages: Candidate passages retrieved by the hybrid retriever.
        top_n: Number of top-scoring passages to return.

    Returns:
        List of ``(original_index, score)`` tuples sorted descending by score,
        containing at most ``top_n`` items.
    """
    if not passages:
        return []

    model = _get_reranker()
    pairs = [[query, passage] for passage in passages]

    scores: list[float] = await asyncio.to_thread(
        lambda: model.predict(pairs).tolist()  # type: ignore[arg-type]
    )

    ranked = sorted(enumerate(scores), key=lambda t: t[1], reverse=True)
    top = ranked[:top_n]

    logger.debug(
        "Reranking complete",
        candidates=len(passages),
        top_n=top_n,
        top_score=top[0][1] if top else None,
    )
    return top
