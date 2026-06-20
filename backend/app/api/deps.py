"""FastAPI dependency providers.

Use these with ``Depends()`` in route handlers to inject typed, validated
resources without coupling routes to infrastructure concerns.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import Settings, get_settings
from app.db.session import get_db

# ── Typed aliases for cleaner route signatures ────────────────────────────────

DBSession = Annotated[AsyncSession, Depends(get_db)]
AppSettings = Annotated[Settings, Depends(get_settings)]
