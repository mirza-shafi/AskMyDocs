"""Pydantic V2 schemas for the RAG query API."""

from __future__ import annotations

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    """Input payload for the RAG query endpoint.

    Attributes:
        question: The natural-language question to answer.
        doc_id: Optional filter — restrict retrieval to a single document.
        include_eval: If True, compute and return Ragas metrics (slow).
    """

    question: str = Field(
        ...,
        min_length=3,
        max_length=1000,
        description="The question to answer from the ingested documents",
        examples=["What are the main findings of the report?"],
    )
    doc_id: str | None = Field(
        default=None,
        description="Restrict search to a specific document ID (optional)",
    )
    include_eval: bool = Field(
        default=False,
        description="Compute Ragas metrics for this query (adds ~3s latency)",
    )


class SourceChunk(BaseModel):
    """A single retrieved context chunk returned alongside the answer.

    Attributes:
        chunk_id: UUID of the DocumentChunk row.
        doc_id: Parent document identifier.
        source_name: Original filename (used in citations).
        chunk_index: Position within the document.
        content: Raw text of the chunk.
        rerank_score: Cross-encoder relevance score (higher is better).
    """

    chunk_id: str
    doc_id: str
    source_name: str
    chunk_index: int
    content: str
    rerank_score: float


class EvalScores(BaseModel):
    """Ragas evaluation metrics for a single query/response pair.

    Attributes:
        faithfulness: Fraction of claims grounded in context (0–1).
        answer_relevance: How well the answer addresses the question (0–1).
        context_precision: Precision of the retrieved context (0–1).
        context_recall: Recall of the retrieved context (0–1).
    """

    faithfulness: float | None = None
    answer_relevance: float | None = None
    context_precision: float | None = None
    context_recall: float | None = None


class QueryResponse(BaseModel):
    """Full response from the RAG query endpoint.

    Attributes:
        question: The original question (echoed back).
        answer: LLM-generated answer with inline [SOURCE_ID] citations.
        sources: Top-N reranked chunks used as context.
        latency_ms: Total server-side processing time in milliseconds.
        eval_scores: Ragas metrics (populated only when include_eval=True).
    """

    question: str
    answer: str
    sources: list[SourceChunk]
    latency_ms: float
    eval_scores: EvalScores | None = None
