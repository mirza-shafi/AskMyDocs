"""Hybrid retrieval combining vector (semantic) and full-text keyword search.

Algorithm:
  1. Vector search — cosine similarity over pgvector HNSW index (top-k).
  2. Full-text search — PostgreSQL ``plainto_tsquery`` over GIN tsvector index (top-k).
  3. Reciprocal Rank Fusion (RRF) — merge both ranked lists into a unified score.
     RRF(d) = Σ 1 / (k + rank_i)   where k=60 (default, from literature).

The fused ranking provides robustly higher recall than either search arm alone.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass

import structlog
from sqlalchemy import delete, func, select, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.models.document import DocumentChunk

logger = structlog.get_logger(__name__)


@dataclass(slots=True)
class RetrievedChunk:
    """A candidate chunk returned by the retriever.

    Attributes:
        id: UUID of the ``DocumentChunk`` row.
        doc_id: Parent document identifier.
        source_name: Original filename.
        chunk_index: Position within the document.
        content: Raw text of the chunk.
        rrf_score: Reciprocal Rank Fusion score (higher = more relevant).
    """

    id: uuid.UUID
    doc_id: str
    source_name: str
    chunk_index: int
    content: str
    rrf_score: float


async def hybrid_search(
    db: AsyncSession,
    query_vector: list[float],
    query_text: str,
    top_k: int,
    rrf_k: int,
    doc_id_filter: str | None = None,
) -> list[RetrievedChunk]:
    """Perform hybrid retrieval and fuse results via RRF.

    Args:
        db: Active async database session.
        query_vector: 384-dim dense embedding of the user query.
        query_text: Raw query text for full-text search.
        top_k: Number of candidates fetched from each search arm.
        rrf_k: RRF constant (default 60 as per the original paper).
        doc_id_filter: If provided, restrict search to a single document.

    Returns:
        List of ``RetrievedChunk`` objects sorted by RRF score (descending).
    """
    settings = get_settings()

    # ── Build optional WHERE clause ───────────────────────────────────────
    base_filter = (
        DocumentChunk.doc_id == doc_id_filter
        if doc_id_filter
        else text("TRUE")
    )

    # ── 1. Vector search (cosine distance — lower is better) ──────────────
    vector_q = (
        select(
            DocumentChunk.id,
            DocumentChunk.doc_id,
            DocumentChunk.source_name,
            DocumentChunk.chunk_index,
            DocumentChunk.content,
        )
        .where(DocumentChunk.embedding.is_not(None))
        .where(base_filter)
        .order_by(DocumentChunk.embedding.cosine_distance(query_vector))
        .limit(top_k)
    )

    # ── 2. Full-text search (ts_rank — higher is better) ──────────────────
    ts_query = func.plainto_tsquery("english", query_text)
    fts_q = (
        select(
            DocumentChunk.id,
            DocumentChunk.doc_id,
            DocumentChunk.source_name,
            DocumentChunk.chunk_index,
            DocumentChunk.content,
        )
        .where(DocumentChunk.fts_vector.op("@@")(ts_query))
        .where(base_filter)
        .order_by(func.ts_rank(DocumentChunk.fts_vector, ts_query).desc())
        .limit(top_k)
    )

    vector_rows = (await db.execute(vector_q)).fetchall()
    fts_rows = (await db.execute(fts_q)).fetchall()

    logger.debug(
        "Raw retrieval counts",
        vector_hits=len(vector_rows),
        fts_hits=len(fts_rows),
    )

    # ── 3. Reciprocal Rank Fusion ─────────────────────────────────────────
    # Accumulate RRF scores per chunk id
    rrf_scores: dict[uuid.UUID, float] = {}
    chunk_data: dict[uuid.UUID, tuple] = {}

    for rank, row in enumerate(vector_rows, start=1):
        rrf_scores[row.id] = rrf_scores.get(row.id, 0.0) + 1.0 / (rrf_k + rank)
        chunk_data[row.id] = row

    for rank, row in enumerate(fts_rows, start=1):
        rrf_scores[row.id] = rrf_scores.get(row.id, 0.0) + 1.0 / (rrf_k + rank)
        chunk_data[row.id] = row

    # Sort by fused score (descending) and return top results
    sorted_ids = sorted(rrf_scores, key=lambda cid: rrf_scores[cid], reverse=True)

    results = [
        RetrievedChunk(
            id=cid,
            doc_id=chunk_data[cid].doc_id,
            source_name=chunk_data[cid].source_name,
            chunk_index=chunk_data[cid].chunk_index,
            content=chunk_data[cid].content,
            rrf_score=rrf_scores[cid],
        )
        for cid in sorted_ids
    ]

    logger.info("Hybrid retrieval complete", total_candidates=len(results))
    return results


async def delete_document_chunks(db: AsyncSession, doc_id: str) -> int:
    """Delete all chunks belonging to a document.

    Args:
        db: Active async database session.
        doc_id: Document identifier whose chunks should be removed.

    Returns:
        Number of rows deleted.
    """
    result = await db.execute(
        delete(DocumentChunk).where(DocumentChunk.doc_id == doc_id)
    )
    return result.rowcount


async def list_documents(db: AsyncSession) -> list[dict]:
    """Return a summary of all ingested documents.

    Args:
        db: Active async database session.

    Returns:
        List of dicts with ``doc_id``, ``source_name``, and ``chunk_count``.
    """
    rows = await db.execute(
        select(
            DocumentChunk.doc_id,
            DocumentChunk.source_name,
            func.count(DocumentChunk.id).label("chunk_count"),
        ).group_by(DocumentChunk.doc_id, DocumentChunk.source_name)
    )
    return [
        {"doc_id": r.doc_id, "source_name": r.source_name, "chunk_count": r.chunk_count}
        for r in rows.fetchall()
    ]
