"""Health-check endpoint.

Returns server status and a live database connectivity check. Used by
Render's health-check probe to determine if the service is ready.
"""

from __future__ import annotations

import structlog
from fastapi import APIRouter
from sqlalchemy import text

from app.api.deps import AppSettings, DBSession
from app.core.config import get_settings

logger = structlog.get_logger(__name__)

router = APIRouter(tags=["Health"])


@router.get(
    "/health",
    summary="Liveness and readiness probe",
    response_description="Service status and version",
)
async def health_check(db: DBSession, settings: AppSettings) -> dict:
    """Return service health status including a live database ping.

    Args:
        db: Injected async database session.
        settings: Injected application settings.

    Returns:
        JSON object with status, version, and database connectivity.
    """
    db_ok = False
    try:
        await db.execute(text("SELECT 1"))
        db_ok = True
    except Exception as exc:
        logger.error("Database ping failed", error=str(exc))

    return {
        "status": "ok" if db_ok else "degraded",
        "version": settings.APP_VERSION,
        "database": "connected" if db_ok else "unreachable",
    }
