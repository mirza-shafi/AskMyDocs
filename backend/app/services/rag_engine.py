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

    logger.info("RAG pipeline started", question_preview=question[:80])

    # ── Overview / summary questions ───────────────────────────────────────
    # "Explain this document", "summarize", etc. retrieve poorly via semantic
    # similarity (the query is vague). Instead, feed the LLM a broad, evenly
    # sampled set of the document's parent chunks so it can summarize.
    if _is_overview_question(question):
        sampled = await retriever.sample_parents(
            db=db,
            doc_id_filter=doc_id_filter,
            limit=max(settings.TOP_N_RERANK + 3, 6),
        )
        if sampled:
            context_blocks = [c.content for c in sampled]
            llm_question = (
                f"{question}\n\n(Give a clear overview of what this document is "
                "about: its purpose, the kind of content it contains, and its "
                "main topics. Base the overview only on the context provided.)"
            )
            answer = await llm.generate_answer(
                question=llm_question, context_blocks=context_blocks
            )
            sources = [
                SourceChunk(
                    chunk_id=str(c.id),
                    doc_id=c.doc_id,
                    source_name=c.source_name,
                    chunk_index=c.chunk_index,
                    content=c.content,
                    rerank_score=None,  # representative sample, not relevance-ranked
                )
                for c in sampled
            ]
            elapsed = _elapsed_ms(t_start)
            logger.info(
                "RAG overview complete",
                latency_ms=round(elapsed, 1),
                sampled_parents=len(sampled),
            )
            return RAGResult(
                question=question,
                answer=answer,
                sources=sources,
                latency_ms=round(elapsed, 1),
            )

    # ── Step 1: Embed the query ───────────────────────────────────────────
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

    # ── Step 3: Cross-encoder reranking (on precise child chunks) ──────────
    passages = [c.content for c in candidates]
    ranked = await reranker.rerank(
        query=question,
        passages=passages,
        top_n=settings.TOP_N_RERANK,
    )

    # ── Step 4: Parent-child expansion ─────────────────────────────────────
    # Children give retrieval precision; their parents give the LLM broader
    # context. Resolve each reranked child to its parent, de-duplicating
    # parents while preserving rerank order so [Sn] maps 1:1 with context.
    ranked_children = [(candidates[idx], score) for idx, score in ranked]
    parent_ids = [c.parent_id for c, _ in ranked_children if c.parent_id is not None]
    parents = await retriever.fetch_parents(db, parent_ids)

    context_blocks: list[str] = []
    sources: list[SourceChunk] = []
    seen: set = set()
    for child, score in ranked_children:
        parent = parents.get(child.parent_id) if child.parent_id else None
        ctx_chunk = parent if parent is not None else child
        # Skip blank context (e.g. image-only PDFs that yielded no text).
        if not ctx_chunk.content.strip():
            continue
        if ctx_chunk.id in seen:
            continue
        seen.add(ctx_chunk.id)
        context_blocks.append(ctx_chunk.content)
        sources.append(
            SourceChunk(
                chunk_id=str(ctx_chunk.id),
                doc_id=ctx_chunk.doc_id,
                source_name=ctx_chunk.source_name,
                chunk_index=ctx_chunk.chunk_index,
                content=ctx_chunk.content,
                rerank_score=round(score, 4),
            )
        )

    # ── Step 5: Generate the cited answer from parent context ──────────────
    answer = await llm.generate_answer(question=question, context_blocks=context_blocks)

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


# Phrases that signal a document-level overview/summary request rather than a
# specific factual question.
_OVERVIEW_PATTERNS = (
    "explain this",
    "explain the document",
    "explain the pdf",
    "explain the file",
    "what is this",
    "what's this",
    "whats this",
    "what is in this",
    "what's in this",
    "whats in this",
    "what is the document",
    "what is this document",
    "what is this pdf",
    "what is this file",
    "what does this document",
    "what does this pdf",
    "tell me about this",
    "about this document",
    "about this pdf",
    "about this file",
    "summarize",
    "summary",
    "overview",
    "describe this",
    "give me a rundown",
    "tldr",
    "tl;dr",
)


def _is_overview_question(question: str) -> bool:
    """Return True if the question asks for a document overview/summary."""
    q = question.lower()
    return any(pattern in q for pattern in _OVERVIEW_PATTERNS)
