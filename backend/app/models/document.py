"""SQLAlchemy ORM model for document chunks with pgvector and full-text search.

The ``DocumentChunk`` table stores semantically chunked text from ingested
documents. Each row carries:
  - A dense 384-dim embedding for approximate nearest-neighbour search.
  - A PostgreSQL ``tsvector`` column for BM25-style keyword search.
  - JSONB metadata for extensibility (page number, section heading, etc.).
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from pgvector.sqlalchemy import Vector
from sqlalchemy import DateTime, Index, Integer, String, Text, func
from sqlalchemy.dialects.postgresql import JSONB, TSVECTOR, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    """Declarative base shared by all ORM models."""


class DocumentChunk(Base):
    """A single semantic chunk of an ingested document.

    Attributes:
        id: UUID primary key, auto-generated.
        doc_id: Caller-supplied identifier for the parent document (slug/hash).
        source_name: Original filename shown to users in citations.
        chunk_index: Zero-based position within the parent document.
        content: Raw text of this chunk.
        embedding: 384-dimensional dense vector (BAAI/bge-small-en-v1.5).
        fts_vector: PostgreSQL tsvector for GIN-indexed keyword search.
        metadata_: Arbitrary key/value pairs (page, section, author, …).
        created_at: UTC wall-clock time of ingestion.
    """

    __tablename__ = "document_chunks"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        comment="Chunk primary key (UUID v4)",
    )
    doc_id: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        index=True,
        comment="Parent document identifier",
    )
    source_name: Mapped[str] = mapped_column(
        String(512),
        nullable=False,
        comment="Original filename shown in citations",
    )
    chunk_index: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        comment="Zero-based chunk position within the document",
    )
    content: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        comment="Raw text of the chunk",
    )
    embedding: Mapped[list[float] | None] = mapped_column(
        Vector(384),
        nullable=True,
        comment="384-dim dense vector from BAAI/bge-small-en-v1.5",
    )
    fts_vector: Mapped[Any] = mapped_column(
        TSVECTOR,
        nullable=True,
        comment="PostgreSQL tsvector for GIN full-text search",
    )
    metadata_: Mapped[dict[str, Any]] = mapped_column(
        "metadata",
        JSONB,
        nullable=False,
        default=dict,
        comment="Extensible JSON metadata (page, section, …)",
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    __table_args__ = (
        # GIN index for tsvector keyword search
        Index("ix_doc_chunks_fts", "fts_vector", postgresql_using="gin"),
        # HNSW index for fast approximate nearest-neighbour vector search
        Index(
            "ix_doc_chunks_embedding",
            "embedding",
            postgresql_using="hnsw",
            postgresql_with={"m": 16, "ef_construction": 64},
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
    )

    def __repr__(self) -> str:
        return (
            f"<DocumentChunk id={self.id} "
            f"doc_id={self.doc_id!r} chunk={self.chunk_index}>"
        )
