"""Async database engine and session factory.

Usage (via FastAPI Depends):
    from app.db.session import get_db
    async def endpoint(db: AsyncSession = Depends(get_db)): ...

The engine is module-level so it is created once per worker process.
Connection pool is sized conservatively for Neon.tech free tier (20 conn max).

SSL note: asyncpg does not accept ?ssl= or ?sslmode= as URL query params via
SQLAlchemy. SSL is passed explicitly via connect_args={"ssl": ssl_context}.
"""

from __future__ import annotations

import re
import ssl
from collections.abc import AsyncGenerator

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from app.core.config import get_settings

_settings = get_settings()


def _clean_url(url: str) -> tuple[str, dict]:
    """Strip ?ssl=... / ?sslmode=... from URL; return (clean_url, connect_args)."""
    connect_args: dict = {}
    if re.search(r"[?&](ssl|sslmode)=", url):
        url = re.sub(r"[?&]ssl(mode)?=[^&]*", "", url).rstrip("?&")
        connect_args["ssl"] = ssl.create_default_context()
    return url, connect_args


_clean_db_url, _connect_args = _clean_url(_settings.DATABASE_URL)

_engine_kwargs: dict = {
    "echo": _settings.DEBUG,
}

if _clean_db_url.startswith("sqlite"):
    from sqlalchemy.pool import StaticPool
    _engine_kwargs["poolclass"] = StaticPool
    _engine_kwargs["connect_args"] = {"check_same_thread": False, **_connect_args}
else:
    _engine_kwargs["connect_args"] = _connect_args
    _engine_kwargs["pool_size"] = 5        # Conservative for Neon free tier
    _engine_kwargs["max_overflow"] = 10
    _engine_kwargs["pool_pre_ping"] = True # Recover from idle connection drops
    _engine_kwargs["pool_recycle"] = 1800  # Recycle connections after 30 min

# One engine per worker process — reused across all requests.
engine: AsyncEngine = create_async_engine(_clean_db_url, **_engine_kwargs)

# Session factory — expire_on_commit=False keeps objects usable after commit.
AsyncSessionLocal: async_sessionmaker[AsyncSession] = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autocommit=False,
    autoflush=False,
)


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """Yield a transactional async database session.

    Commits on success, rolls back on any exception, and always closes the
    session. Intended for use as a FastAPI dependency.

    Yields:
        AsyncSession: An active SQLAlchemy async session.
    """
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()
