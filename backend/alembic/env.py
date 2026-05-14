"""Alembic async migration environment.

Supports both offline (--sql) and online (async engine) migration modes.
DATABASE_URL is read from the application's Settings to avoid duplication.
"""

from __future__ import annotations

import asyncio
from logging.config import fileConfig

from alembic import context
from sqlalchemy.ext.asyncio import create_async_engine

from app.core.config import get_settings
from app.models.document import Base  # noqa: F401 — registers all ORM models

# ── Alembic config object ─────────────────────────────────────────────────────
config = context.config
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# The metadata object drives autogenerate ("alembic revision --autogenerate")
target_metadata = Base.metadata

_settings = get_settings()


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode (outputs raw SQL, no DB connection).

    Useful for generating SQL scripts to review or apply manually.
    """
    context.configure(
        url=_settings.DATABASE_URL,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection) -> None:  # type: ignore[no-untyped-def]
    """Execute pending migrations using the given synchronous connection."""
    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        compare_type=True,
    )
    with context.begin_transaction():
        context.run_migrations()


async def run_migrations_online() -> None:
    """Run migrations in 'online' mode using an async engine."""
    connectable = create_async_engine(_settings.DATABASE_URL, future=True)
    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)
    await connectable.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    asyncio.run(run_migrations_online())
