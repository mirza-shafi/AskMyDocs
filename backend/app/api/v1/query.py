"""RAG query endpoint.

Accepts a user question, runs the full RAG pipeline (embed → retrieve →
rerank → generate), and returns a cited answer with source attribution.

Optionally computes Ragas evaluation metrics when ``include_eval=True``
(adds ~3-5 seconds of latency due to additional LLM judge calls).
"""

from __future__ import annotations

import structlog
from fastapi import APIRouter

from app.api.deps import DBSession
from app.schemas.query import QueryRequest, QueryResponse
from app.services import evaluator, rag_engine
from app.services.evaluator import EvalInput

logger = structlog.get_logger(__name__)

router = APIRouter(tags=["Query"])


@router.post(
    "/query",
    response_model=QueryResponse,
    summary="Ask a question and get a cited answer",
)
async def query_documents(
    body: QueryRequest,
    db: DBSession,
) -> QueryResponse:
    """Run the full RAG pipeline for a user question.

    Steps:
      1. Embed the question (384-dim).
      2. Hybrid retrieval (vector + FTS) with RRF fusion.
      3. Cross-encoder reranking to select top-N context chunks.
      4. Groq Llama-3 generation with citation-enforcing system prompt.
      5. Optionally run Ragas evaluation metrics.

    Args:
        body: Validated ``QueryRequest`` payload.
        db: Injected async database session.

    Returns:
        ``QueryResponse`` with answer, source citations, latency, and optional
        Ragas scores.
    """
    logger.info(
        "Query received",
        question_preview=body.question[:80],
        doc_id_filter=body.doc_id,
        include_eval=body.include_eval,
    )

    result = await rag_engine.run(
        db=db,
        question=body.question,
        doc_id_filter=body.doc_id,
    )

    eval_scores = None
    if body.include_eval:
        eval_input = EvalInput(
            question=body.question,
            answer=result.answer,
            contexts=[s.content for s in result.sources],
        )
        eval_scores = await evaluator.evaluate_rag(eval_input)

    return QueryResponse(
        question=result.question,
        answer=result.answer,
        sources=result.sources,
        latency_ms=result.latency_ms,
        eval_scores=eval_scores,
    )
