"""FastAPI application factory with lifespan, middleware, and exception handlers.

Architecture:
  - Lifespan context manager: warm up ML models at startup.
  - RequestLoggingMiddleware: structured JSON log per request.
  - CORSMiddleware: allow configured Vercel origins.
  - GZipMiddleware: compress large response bodies.
  - Global exception handlers: consistent JSON error responses.
"""

from __future__ import annotations

import time
import uuid
from contextlib import asynccontextmanager
from typing import AsyncGenerator

import structlog
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from starlette.middleware.base import BaseHTTPMiddleware

from app.api.v1.router import router as v1_router
from app.core.config import get_settings
from app.core.exceptions import register_exception_handlers
from app.core.logging import configure_logging
from app.services import embedder, reranker

logger = structlog.get_logger(__name__)


# ── Lifespan ──────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Manage application startup and shutdown lifecycle.

    Startup:
      - Configure structured logging.
      - Pre-load embedding and reranker models into memory.
        (Avoids cold-start latency on the first real request.)

    Shutdown:
      - Log graceful shutdown message.
    """
    settings = get_settings()
    configure_logging(settings.LOG_LEVEL)
    logger.info("AskMyDocs starting up", version=settings.APP_VERSION)

    # Warm up ML models synchronously at startup
    embedder.initialise()
    reranker.initialise()

    logger.info("All models loaded — server is ready")
    yield

    logger.info("AskMyDocs shutting down")


# ── Request Logging Middleware ─────────────────────────────────────────────────

class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Log every HTTP request with method, path, status, and duration."""

    async def dispatch(self, request: Request, call_next) -> Response:  # type: ignore[override]
        request_id = str(uuid.uuid4())
        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(request_id=request_id)

        t_start = time.perf_counter()
        response: Response = await call_next(request)
        duration_ms = round((time.perf_counter() - t_start) * 1000, 1)

        logger.info(
            "HTTP request",
            method=request.method,
            path=request.url.path,
            status=response.status_code,
            duration_ms=duration_ms,
        )

        response.headers["X-Request-ID"] = request_id
        return response


# ── Application Factory ────────────────────────────────────────────────────────

def create_app() -> FastAPI:
    """Create and configure the FastAPI application instance.

    Returns:
        Configured ``FastAPI`` application ready to serve.
    """
    settings = get_settings()

    app = FastAPI(
        title="AskMyDocs API",
        description=(
            "Production-grade RAG API: Hybrid Search (pgvector + FTS) + "
            "Cross-Encoder Reranking + Groq Llama-3 with citation enforcement."
        ),
        version=settings.APP_VERSION,
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    # ── Middleware (order matters: outermost applied last) ─────────────────
    app.add_middleware(GZipMiddleware, minimum_size=1000)
    app.add_middleware(RequestLoggingMiddleware)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins_list,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Exception Handlers ────────────────────────────────────────────────
    register_exception_handlers(app)

    # ── Routers ───────────────────────────────────────────────────────────
    # Health at root (GET /health) — no versioned prefix for Render probe
    app.include_router(v1_router, prefix="/api/v1")
    # Also expose health at root for Render's default health check path
    from app.api.v1.health import router as health_router
    app.include_router(health_router)

    return app


app = create_app()
