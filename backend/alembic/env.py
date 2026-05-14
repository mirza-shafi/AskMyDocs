"""Alembic async migration environment.

Supports both offline (--sql) and online (async engine) migration modes.
DATABASE_URL is read from the application's Settings to avoid duplication.

SSL note: asyncpg does not accept ?ssl= or ?sslmode= as URL query parameters
when used via SQLAlchemy. SSL must be passed via connect_args={"ssl": True}.
This env.py strips those params from the URL automatically.
"""

from __future__ import annotations

import asyncio
import re
import ssl
from logging.config import fileConfig

from alembic import context
from sqlalchemy.ext.asyncio import create_async_engine

from app.core.config import get_settings
from app.models.document import Base  # noqa: F401 — registers all ORM models

# ── Alembic config object ─────────────────────────────────────────────────────
config = context.config
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata
_settings = get_settings()


def _clean_url(url: str) -> tuple[str, dict]:
    """Strip SSL query params from URL and return (clean_url, connect_args).

    asyncpg ignores ?ssl= and ?sslmode= when passed in the URL string via
    SQLAlchemy. Instead, SSL must be passed as connect_args={"ssl": ctx}.

    Args:
        url: Raw DATABASE_URL from settings.

    Returns:
        Tuple of (url_without_ssl_params, connect_args_dict).
    """
    connect_args: dict = {}
    needs_ssl = bool(re.search(r"[?&](ssl|sslmode)=", url))

    if needs_ssl:
        # Strip ?ssl=... and &sslmode=... from the URL
        clean = re.sub(r"[?&]ssl(mode)?=[^&]*", "", url).rstrip("?&")
        ctx = ssl.create_default_context()
        connect_args["ssl"] = ctx
        return clean, connect_args

    return url, connect_args


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode (outputs raw SQL, no DB connection)."""
    clean_url, _ = _clean_url(_settings.DATABASE_URL)
    context.configure(
        url=clean_url,
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
    clean_url, connect_args = _clean_url(_settings.DATABASE_URL)
    connectable = create_async_engine(
        clean_url,
        connect_args=connect_args,
        future=True,
    )
    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)
    await connectable.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    asyncio.run(run_migrations_online())

