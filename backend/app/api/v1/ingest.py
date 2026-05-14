"""Document ingestion endpoints.

Handles file upload → background chunking/embedding → status polling.
Raw PDF/TXT files are discarded after chunking; only chunk text and
embeddings are persisted in PostgreSQL.

Endpoints:
  POST /api/v1/ingest           — upload file, start background job
  GET  /api/v1/ingest/{job_id} — poll job status
  GET  /api/v1/docs             — list all ingested documents
  DELETE /api/v1/docs/{doc_id} — delete document and all its chunks
"""

from __future__ import annotations

import asyncio
import hashlib
import uuid
from typing import Any

import structlog
from fastapi import APIRouter, BackgroundTasks, UploadFile
from sqlalchemy import func
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import DBSession
from app.models.document import DocumentChunk
from app.schemas.ingest import (
    DeleteResponse,
    DocumentListResponse,
    IngestResponse,
    IngestStatus,
    IngestStatusResponse,
)
from app.services import embedder
from app.services.chunker import chunk_document
from app.services.retriever import delete_document_chunks, list_documents

logger = structlog.get_logger(__name__)
router = APIRouter(tags=["Ingestion"])

# ── In-memory job store ───────────────────────────────────────────────────────
# For a single-worker Render deployment this is sufficient.
# Replace with Redis for multi-worker or persistence across restarts.
_jobs: dict[str, dict[str, Any]] = {}
_jobs_lock = asyncio.Lock()


# ── Background task ───────────────────────────────────────────────────────────

async def _process_document(
    job_id: str,
    doc_id: str,
    source_name: str,
    file_bytes: bytes,
    db_session_factory: Any,
) -> None:
    """Background task: chunk, embed, and store a document.

    Args:
        job_id: Tracking identifier for this ingestion job.
        doc_id: Stable document identifier derived from the filename.
        source_name: Original filename (shown in citations).
        file_bytes: Raw file content already read into memory.
        db_session_factory: Callable that yields an async DB session.
    """
    async with _jobs_lock:
        _jobs[job_id]["status"] = IngestStatus.PROCESSING

    chunks_created = 0
    try:
        # Step 1: Chunk the document
        text_chunks = chunk_document(file_bytes, source_name)

        if not text_chunks:
            raise ValueError("Document produced no text chunks — may be empty or image-only PDF")

        # Step 2: Batch embed all chunks
        texts = [c.content for c in text_chunks]
        vectors = await embedder.embed_batch(texts)

        # Step 3: Persist chunks to the database
        async with db_session_factory() as session:
            for chunk, vector in zip(text_chunks, vectors):
                # Build tsvector from content using PostgreSQL's to_tsvector
                db_chunk = DocumentChunk(
                    doc_id=doc_id,
                    source_name=source_name,
                    chunk_index=chunk.chunk_index,
                    content=chunk.content,
                    embedding=vector,
                    fts_vector=func.to_tsvector("english", chunk.content),
                    metadata_=chunk.metadata,
                )
                session.add(db_chunk)
                chunks_created += 1
            await session.commit()

        logger.info(
            "Ingestion complete",
            job_id=job_id,
            doc_id=doc_id,
            chunks=chunks_created,
        )

        async with _jobs_lock:
            _jobs[job_id]["status"] = IngestStatus.COMPLETED
            _jobs[job_id]["chunks_created"] = chunks_created

    except Exception as exc:
        logger.error("Ingestion failed", job_id=job_id, error=str(exc))
        async with _jobs_lock:
            _jobs[job_id]["status"] = IngestStatus.FAILED
            _jobs[job_id]["error"] = str(exc)


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post(
    "/ingest",
    response_model=IngestResponse,
    status_code=202,
    summary="Upload a PDF or TXT file for ingestion",
)
async def ingest_document(
    file: UploadFile,
    background_tasks: BackgroundTasks,
    db: DBSession,
) -> IngestResponse:
    """Accept a PDF or TXT file, queue it for processing, and return a job ID.

    The file is read into memory, then chunked, embedded, and stored by a
    background task. Poll ``GET /api/v1/ingest/{job_id}`` for status.

    Args:
        file: The uploaded file (PDF or TXT only).
        background_tasks: FastAPI background task queue.
        db: Async database session (used indirectly via factory).

    Returns:
        ``IngestResponse`` with job_id and doc_id for status polling.
    """
    filename = file.filename or "unknown.txt"
    file_bytes = await file.read()

    # Derive a stable doc_id from the filename (slug + short hash)
    name_hash = hashlib.md5(filename.encode()).hexdigest()[:8]
    base_name = filename.rsplit(".", 1)[0].lower().replace(" ", "_")
    doc_id = f"{base_name}_{name_hash}"
    job_id = str(uuid.uuid4())

    # Register job
    async with _jobs_lock:
        _jobs[job_id] = {
            "status": IngestStatus.PENDING,
            "doc_id": doc_id,
            "chunks_created": None,
            "error": None,
        }

    # Import session factory here to avoid circular imports at module load
    from app.db.session import AsyncSessionLocal

    background_tasks.add_task(
        _process_document,
        job_id,
        doc_id,
        filename,
        file_bytes,
        AsyncSessionLocal,
    )

    logger.info("Ingestion job queued", job_id=job_id, doc_id=doc_id, filename=filename)

    return IngestResponse(job_id=job_id, doc_id=doc_id)


@router.get(
    "/ingest/{job_id}",
    response_model=IngestStatusResponse,
    summary="Poll ingestion job status",
)
async def get_ingest_status(job_id: str) -> IngestStatusResponse:
    """Return the current status of an ingestion background job.

    Args:
        job_id: The job identifier returned by ``POST /ingest``.

    Returns:
        ``IngestStatusResponse`` with status, chunks_created, and any error.
    """
    from app.core.exceptions import IngestJobNotFoundError

    async with _jobs_lock:
        job = _jobs.get(job_id)

    if job is None:
        raise IngestJobNotFoundError(job_id)

    return IngestStatusResponse(
        job_id=job_id,
        doc_id=job["doc_id"],
        status=job["status"],
        chunks_created=job.get("chunks_created"),
        error=job.get("error"),
    )


@router.get(
    "/docs",
    response_model=DocumentListResponse,
    summary="List all ingested documents",
)
async def list_all_documents(db: DBSession) -> DocumentListResponse:
    """Return a summary of every document currently stored in the database.

    Args:
        db: Async database session.

    Returns:
        ``DocumentListResponse`` with doc summaries and total count.
    """
    docs = await list_documents(db)
    return DocumentListResponse(documents=docs, total=len(docs))


@router.delete(
    "/docs/{doc_id}",
    response_model=DeleteResponse,
    summary="Delete a document and all its chunks",
)
async def delete_document(doc_id: str, db: DBSession) -> DeleteResponse:
    """Remove a document and all associated chunks from the database.

    Args:
        doc_id: The document identifier to delete.
        db: Async database session.

    Returns:
        ``DeleteResponse`` confirming the number of rows deleted.
    """
    from app.core.exceptions import DocumentNotFoundError

    count = await delete_document_chunks(db, doc_id)
    if count == 0:
        raise DocumentNotFoundError(doc_id)

    logger.info("Document deleted", doc_id=doc_id, chunks_deleted=count)
    return DeleteResponse(doc_id=doc_id, chunks_deleted=count)
