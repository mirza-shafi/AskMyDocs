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
from fastapi import APIRouter, BackgroundTasks, HTTPException, UploadFile
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
from app.services.chunker import chunk_document_hierarchical
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
        # Step 1: Structure-aware hierarchical chunking (parents + children)
        parents = chunk_document_hierarchical(file_bytes, source_name)

        if not parents:
            raise ValueError("Document produced no text chunks — may be empty or image-only PDF")

        # Step 2: Batch-embed every child across all parents in a single call.
        # Parents are NOT embedded — they only provide context to the LLM.
        all_child_texts = [c.content for p in parents for c in p.children]
        if not all_child_texts:
            raise ValueError("Document produced no child chunks to embed")
        child_vectors = await embedder.embed_batch(all_child_texts)

        # Step 3: Persist parents (context) and children (searchable) to the DB.
        async with db_session_factory() as session:
            # Idempotent re-ingest: clear any existing chunks for this doc_id
            # first, so re-uploading the same file replaces rather than
            # duplicates its chunks (doc_id is derived from the filename).
            removed = await delete_document_chunks(session, doc_id)
            if removed:
                logger.info(
                    "Replacing existing document chunks",
                    doc_id=doc_id,
                    removed=removed,
                )
            vi = 0
            parents_created = 0
            for parent in parents:
                parent_uuid = uuid.uuid4()
                # Parent / context chunk: no embedding, no FTS — never retrieved
                # directly, only fetched to expand a matched child.
                session.add(
                    DocumentChunk(
                        id=parent_uuid,
                        doc_id=doc_id,
                        parent_id=None,
                        source_name=source_name,
                        chunk_index=parent.chunk_index,
                        content=parent.content,
                        embedding=None,
                        fts_vector=None,
                        metadata_=parent.metadata,
                    )
                )
                parents_created += 1

                for child in parent.children:
                    vector = child_vectors[vi]
                    vi += 1
                    # Child chunk: embedded + FTS-indexed, linked to its parent.
                    session.add(
                        DocumentChunk(
                            doc_id=doc_id,
                            parent_id=parent_uuid,
                            source_name=source_name,
                            chunk_index=child.chunk_index,
                            content=child.content,
                            embedding=vector,
                            fts_vector=func.to_tsvector("english", child.content),
                            metadata_=child.metadata,
                        )
                    )
                    chunks_created += 1
            await session.commit()

        logger.info(
            "Ingestion complete",
            job_id=job_id,
            doc_id=doc_id,
            parents=parents_created,
            children=chunks_created,
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

    # Validate file type up-front so unsupported uploads fail fast with 422
    # instead of being accepted (202) and then failing silently in the worker.
    from app.core.exceptions import UnsupportedFileTypeError

    ext = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if ext not in {".pdf", ".txt"}:
        raise UnsupportedFileTypeError(filename)

    file_bytes = await file.read()

    # Reject oversized uploads (protect worker memory).
    max_bytes = 25 * 1024 * 1024  # 25 MB
    if len(file_bytes) > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"File too large ({len(file_bytes) // (1024 * 1024)} MB). Max is 25 MB.",
        )
    if not file_bytes:
        raise HTTPException(status_code=422, detail="Uploaded file is empty.")

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
