"""Pydantic V2 schemas for the document ingestion API."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class IngestStatus(str, Enum):
    """Possible states of a background ingestion job."""

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class IngestResponse(BaseModel):
    """Response returned immediately when a file is submitted for ingestion.

    Attributes:
        job_id: Unique identifier for polling the ingestion status.
        doc_id: Stable document identifier derived from the filename.
        status: Initial job status (always ``pending``).
        message: Human-readable confirmation message.
    """

    job_id: str = Field(..., description="Poll this ID at GET /api/v1/ingest/{job_id}")
    doc_id: str = Field(..., description="Stable document identifier")
    status: IngestStatus = IngestStatus.PENDING
    message: str = Field(default="File received and queued for processing")


class IngestStatusResponse(BaseModel):
    """Response for polling an ingestion job's progress.

    Attributes:
        job_id: The job being polled.
        doc_id: Document identifier associated with this job.
        status: Current status of the background task.
        chunks_created: Number of chunks stored (set after completion).
        error: Error detail if the job failed.
    """

    job_id: str
    doc_id: str
    status: IngestStatus
    chunks_created: int | None = None
    error: str | None = None


class DocumentListItem(BaseModel):
    """A summary item for listing ingested documents.

    Attributes:
        doc_id: Unique document identifier.
        source_name: Original filename.
        chunk_count: Total number of stored chunks.
    """

    doc_id: str
    source_name: str
    chunk_count: int


class DocumentListResponse(BaseModel):
    """Paginated list of all ingested documents.

    Attributes:
        documents: List of document summaries.
        total: Total number of distinct documents.
    """

    documents: list[DocumentListItem]
    total: int


class DeleteResponse(BaseModel):
    """Confirmation of a document deletion.

    Attributes:
        doc_id: The document that was deleted.
        chunks_deleted: Number of chunk rows removed.
        message: Human-readable confirmation.
    """

    doc_id: str
    chunks_deleted: int
    message: str = "Document and all associated chunks deleted successfully"
