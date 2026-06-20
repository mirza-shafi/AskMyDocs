"""Custom domain exceptions and global FastAPI exception handlers.

Register all handlers by calling ``register_exception_handlers(app)``
inside the FastAPI app factory.
"""

from __future__ import annotations

import uuid

import structlog
from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse

logger = structlog.get_logger(__name__)


# ── Domain Exceptions ─────────────────────────────────────────────────────────


class DocumentNotFoundError(Exception):
    """Raised when a requested document does not exist in the database.

    Args:
        doc_id: The document identifier that was not found.
    """

    def __init__(self, doc_id: str) -> None:
        self.doc_id = doc_id
        super().__init__(f"Document '{doc_id}' not found")


class IngestJobNotFoundError(Exception):
    """Raised when an ingestion job ID cannot be located.

    Args:
        job_id: The job identifier that was not found.
    """

    def __init__(self, job_id: str) -> None:
        self.job_id = job_id
        super().__init__(f"Ingestion job '{job_id}' not found")


class EmbeddingError(Exception):
    """Raised when the embedding model fails to produce a vector."""


class LLMError(Exception):
    """Raised when the Groq LLM API call fails or returns an error."""


class UnsupportedFileTypeError(Exception):
    """Raised when an uploaded file has an unsupported MIME type or extension.

    Args:
        filename: The name of the rejected file.
    """

    def __init__(self, filename: str) -> None:
        self.filename = filename
        super().__init__(f"Unsupported file type: '{filename}'. Use PDF or TXT.")


# ── Handler Registration ──────────────────────────────────────────────────────


def register_exception_handlers(app: FastAPI) -> None:
    """Attach all global exception handlers to the FastAPI application.

    Args:
        app: The FastAPI application instance.
    """

    @app.exception_handler(DocumentNotFoundError)
    async def _document_not_found(
        request: Request, exc: DocumentNotFoundError
    ) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_404_NOT_FOUND,
            content={"error": str(exc), "doc_id": exc.doc_id},
        )

    @app.exception_handler(IngestJobNotFoundError)
    async def _ingest_job_not_found(
        request: Request, exc: IngestJobNotFoundError
    ) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_404_NOT_FOUND,
            content={"error": str(exc), "job_id": exc.job_id},
        )

    @app.exception_handler(UnsupportedFileTypeError)
    async def _unsupported_file_type(
        request: Request, exc: UnsupportedFileTypeError
    ) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={"error": str(exc), "filename": exc.filename},
        )

    @app.exception_handler(Exception)
    async def _global_handler(request: Request, exc: Exception) -> JSONResponse:
        request_id = str(uuid.uuid4())
        logger.error(
            "Unhandled exception",
            request_id=request_id,
            method=request.method,
            path=request.url.path,
            exc_info=exc,
        )
        # The base-Exception (500) handler runs in Starlette's outermost
        # ServerErrorMiddleware, OUTSIDE CORSMiddleware, so error responses
        # would otherwise lack CORS headers and the browser would see an
        # opaque "network error" instead of the JSON body. Re-add them here.
        from app.core.config import get_settings

        headers: dict[str, str] = {}
        origin = request.headers.get("origin")
        if origin and origin in get_settings().cors_origins_list:
            headers["Access-Control-Allow-Origin"] = origin
            headers["Access-Control-Allow-Credentials"] = "true"
            headers["Vary"] = "Origin"

        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"error": "Internal server error", "request_id": request_id},
            headers=headers,
        )
