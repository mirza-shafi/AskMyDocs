"""API v1 router — aggregates all v1 endpoint routers."""

from __future__ import annotations

from fastapi import APIRouter

from app.api.v1 import health, ingest, query

router = APIRouter()

# Health is mounted at root level (no /api/v1 prefix) for Render probe compatibility
router.include_router(health.router)

# Ingestion and query endpoints
router.include_router(ingest.router)
router.include_router(query.router)
