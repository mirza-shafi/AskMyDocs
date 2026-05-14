"""RAG pipeline orchestrator: embeds → retrieves → reranks → generates.

This module is the single entry point for the full retrieval-augmented
generation pipeline. It composes the embedder, retriever, reranker, and LLM
services into a single ``run`` coroutine.

Pipeline:
  1. Embed the user query (384-dim vector).
  2. Hybrid retrieval (vector cosine + full-text) with RRF fusion → top-K candidates.
  3. Cross-encoder reranking → top-N context chunks.
  4. Groq LLM generation with citation-enforcing system prompt.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import structlog
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.schemas.query import SourceChunk
from app.services import embedder, llm, reranker, retriever

logger = structlog.get_logger(__name__)


@dataclass(slots=True)
class RAGResult:
    """The structured output of the RAG pipeline.

    Attributes:
        question: Original user question.
        answer: LLM-generated answer with inline [Sn] citations.
        sources: Reranked context chunks used as LLM input.
        latency_ms: Total wall-clock time for the pipeline in milliseconds.
    """

    question: str
    answer: str
    sources: list[SourceChunk]
    latency_ms: float


async def run(
    db: AsyncSession,
    question: str,
    doc_id_filter: str | None = None,
) -> RAGResult:
    """Execute the full RAG pipeline for a user question.

    Args:
        db: Active async database session.
        question: The natural-language question to answer.
        doc_id_filter: Optional document ID to restrict retrieval scope.

    Returns:
        A ``RAGResult`` containing the answer and source attribution.
    """
    settings = get_settings()
    t_start = time.perf_counter()

    # ── Step 1: Embed the query ───────────────────────────────────────────
    logger.info("RAG pipeline started", question_preview=question[:80])
    query_vector = await embedder.embed_text(question)

    # ── Step 2: Hybrid retrieval + RRF ────────────────────────────────────
    candidates = await retriever.hybrid_search(
        db=db,
        query_vector=query_vector,
        query_text=question,
        top_k=settings.TOP_K_RETRIEVAL,
        rrf_k=settings.RRF_K,
        doc_id_filter=doc_id_filter,
    )

    if not candidates:
        logger.warning("No candidates retrieved", question=question)
        return RAGResult(
            question=question,
            answer="I don't know based on the provided documents.",
            sources=[],
            latency_ms=_elapsed_ms(t_start),
        )

    # ── Step 3: Cross-encoder reranking ───────────────────────────────────
    passages = [c.content for c in candidates]
    ranked = await reranker.rerank(
        query=question,
        passages=passages,
        top_n=settings.TOP_N_RERANK,
    )

    top_chunks = [candidates[idx] for idx, _ in ranked]
    top_scores = {idx: score for idx, score in ranked}

    # ── Step 4: Build context and generate ────────────────────────────────
    context_blocks = [chunk.content for chunk in top_chunks]
    answer = await llm.generate_answer(question=question, context_blocks=context_blocks)

    # ── Step 5: Assemble source attribution ───────────────────────────────
    sources = [
        SourceChunk(
            chunk_id=str(candidates[idx].id),
            doc_id=candidates[idx].doc_id,
            source_name=candidates[idx].source_name,
            chunk_index=candidates[idx].chunk_index,
            content=candidates[idx].content,
            rerank_score=round(score, 4),
        )
        for idx, score in ranked
    ]

    elapsed = _elapsed_ms(t_start)
    logger.info(
        "RAG pipeline complete",
        latency_ms=round(elapsed, 1),
        sources_used=len(sources),
    )

    return RAGResult(
        question=question,
        answer=answer,
        sources=sources,
        latency_ms=round(elapsed, 1),
    )


def _elapsed_ms(t_start: float) -> float:
    """Return elapsed milliseconds since ``t_start``."""
    return (time.perf_counter() - t_start) * 1000
