"""Async database engine and session factory.

Usage (via FastAPI Depends):
    from app.db.session import get_db
    async def endpoint(db: AsyncSession = Depends(get_db)): ...

The engine is module-level so it is created once per worker process.
Connection pool is sized conservatively for Neon.tech free tier (20 conn max).
"""

from __future__ import annotations

from collections.abc import AsyncGenerator

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from app.core.config import get_settings

_settings = get_settings()

# One engine per worker process — reused across all requests.
engine: AsyncEngine = create_async_engine(
    _settings.DATABASE_URL,
    pool_size=5,        # Conservative for Neon free tier
    max_overflow=10,
    pool_pre_ping=True, # Recover from idle connection drops
    pool_recycle=1800,  # Recycle connections after 30 min
    echo=_settings.DEBUG,
)

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
