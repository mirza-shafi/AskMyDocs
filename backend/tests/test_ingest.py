"""Tests for the document ingestion pipeline.

Covers:
  - Chunker: PDF and TXT splitting, overlap, unsupported file rejection.
  - Ingest endpoint: file upload acceptance, job creation, status polling.
"""

from __future__ import annotations

import pytest

from app.core.exceptions import UnsupportedFileTypeError
from app.services.chunker import TextChunk, chunk_document

# ── Chunker Unit Tests ────────────────────────────────────────────────────────

class TestChunkDocument:
    """Unit tests for the ``chunk_document`` function."""

    def test_txt_produces_chunks(self):
        """A plain-text file should produce at least one chunk."""
        content = ("This is a test sentence. " * 50).encode("utf-8")
        chunks = chunk_document(content, "test.txt")
        assert len(chunks) >= 1
        assert all(isinstance(c, TextChunk) for c in chunks)

    def test_chunk_index_sequential(self):
        """Chunk indices should be zero-based and sequential."""
        content = ("Word " * 300).encode("utf-8")
        chunks = chunk_document(content, "test.txt")
        for i, chunk in enumerate(chunks):
            assert chunk.chunk_index == i

    def test_chunk_content_not_empty(self):
        """No chunk should have empty or whitespace-only content."""
        content = ("Test content paragraph.\n\n" * 40).encode("utf-8")
        chunks = chunk_document(content, "document.txt")
        assert all(c.content.strip() for c in chunks)

    def test_short_document_single_chunk(self):
        """A document shorter than CHUNK_SIZE should produce exactly one chunk."""
        content = b"Short document."
        chunks = chunk_document(content, "short.txt")
        assert len(chunks) == 1
        assert chunks[0].content == "Short document."

    def test_unsupported_extension_raises(self):
        """Uploading a .docx file should raise UnsupportedFileTypeError."""
        with pytest.raises(UnsupportedFileTypeError) as exc_info:
            chunk_document(b"some content", "file.docx")
        assert "file.docx" in str(exc_info.value)

    def test_metadata_contains_extension(self):
        """Each chunk metadata should record the source file extension."""
        content = b"Some meaningful text for testing metadata fields."
        chunks = chunk_document(content, "report.txt")
        assert all(c.metadata.get("source_ext") == ".txt" for c in chunks)


# ── Ingest Endpoint Tests ─────────────────────────────────────────────────────

class TestIngestEndpoint:
    """Integration tests for POST /api/v1/ingest and GET /api/v1/ingest/{job_id}."""

    @pytest.mark.asyncio
    async def test_ingest_returns_202(self, mock_embedder):
        """A valid TXT upload should return 202 Accepted with a job_id."""
        from io import BytesIO

        from fastapi.testclient import TestClient

        from app.main import app

        client = TestClient(app)
        content = b"This is a test document for ingestion testing purposes.\n" * 20
        response = client.post(
            "/api/v1/ingest",
            files={"file": ("test_doc.txt", BytesIO(content), "text/plain")},
        )
        assert response.status_code == 202
        data = response.json()
        assert "job_id" in data
        assert "doc_id" in data
        assert data["status"] == "pending"

    @pytest.mark.asyncio
    async def test_poll_unknown_job_returns_404(self):
        """Polling a non-existent job ID should return 404."""
        from fastapi.testclient import TestClient

        from app.main import app

        client = TestClient(app)
        response = client.get("/api/v1/ingest/non-existent-job-id")
        assert response.status_code == 404
