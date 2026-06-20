"""add parent_id for parent-child retrieval

Revision ID: b1c2d3e4f5a6
Revises: 8a3295d8fe99
Create Date: 2026-06-21 00:30:00.000000+00:00
"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "b1c2d3e4f5a6"
down_revision: Union[str, None] = "8a3295d8fe99"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Add nullable parent_id column + index for parent-child retrieval."""
    op.add_column(
        "document_chunks",
        sa.Column(
            "parent_id",
            sa.UUID(),
            nullable=True,
            comment="Parent chunk id (NULL for parent/context chunks; set for children)",
        ),
    )
    op.create_index(
        op.f("ix_document_chunks_parent_id"),
        "document_chunks",
        ["parent_id"],
        unique=False,
    )


def downgrade() -> None:
    """Drop parent_id column and its index."""
    op.drop_index(op.f("ix_document_chunks_parent_id"), table_name="document_chunks")
    op.drop_column("document_chunks", "parent_id")
