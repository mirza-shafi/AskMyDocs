"""Document chunker: converts PDF/TXT files into overlapping text chunks.

Supports:
  - PDF files (via pdfplumber, preserves layout better than PyPDF2)
  - Plain text files (.txt)

The chunking strategy is a simple sliding-window character splitter that
respects sentence/paragraph boundaries where possible by snapping to the
nearest whitespace boundary.
"""

from __future__ import annotations

import io
import re
from dataclasses import dataclass

import pdfplumber
import structlog

from app.core.config import get_settings
from app.core.exceptions import UnsupportedFileTypeError

logger = structlog.get_logger(__name__)

SUPPORTED_EXTENSIONS = {".pdf", ".txt"}


@dataclass(frozen=True, slots=True)
class TextChunk:
    """A single text chunk produced by the chunker.

    Attributes:
        content: Raw text of the chunk.
        chunk_index: Zero-based position within the source document.
        metadata: Extra context (page number, etc.).
    """

    content: str
    chunk_index: int
    metadata: dict


def _extract_text_from_pdf(file_bytes: bytes) -> str:
    """Extract plain text from a PDF binary using pdfplumber.

    Args:
        file_bytes: Raw PDF file content.

    Returns:
        Concatenated text from all pages, separated by newlines.
    """
    pages: list[str] = []
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        for page in pdf.pages:
            text = page.extract_text() or ""
            if text.strip():
                pages.append(text.strip())
    return "\n\n".join(pages)


def _split_text(
    text: str,
    chunk_size: int,
    chunk_overlap: int,
) -> list[str]:
    """Split text into overlapping chunks using a sliding window.

    Snaps chunk boundaries to the nearest whitespace to avoid cutting
    words mid-token.

    Args:
        text: The full document text.
        chunk_size: Target character length per chunk.
        chunk_overlap: Number of characters shared between consecutive chunks.

    Returns:
        List of non-empty text chunks.
    """
    # Normalise whitespace — collapse multiple blank lines into one
    text = re.sub(r"\n{3,}", "\n\n", text).strip()

    if len(text) <= chunk_size:
        return [text]

    chunks: list[str] = []
    start = 0
    text_len = len(text)

    while start < text_len:
        end = min(start + chunk_size, text_len)

        # Snap end to nearest whitespace (don't cut mid-word)
        if end < text_len:
            snap = text.rfind(" ", start, end)
            # Only snap if the resulting chunk is larger than the overlap!
            # Otherwise we'll enter an infinite loop when advancing start.
            if snap > start + chunk_overlap:
                end = snap

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        # Advance start with overlap, forcing strictly forward progress
        next_start = end - chunk_overlap
        if next_start <= start:
            next_start = start + 1
            
        start = next_start

    return chunks


def chunk_document(
    file_bytes: bytes,
    filename: str,
) -> list[TextChunk]:
    """Parse a PDF or TXT file and split it into overlapping text chunks.

    Args:
        file_bytes: Raw file content as bytes.
        filename: Original filename (used to detect MIME type).

    Returns:
        Ordered list of ``TextChunk`` objects.

    Raises:
        UnsupportedFileTypeError: If the file extension is not in
            ``SUPPORTED_EXTENSIONS``.
    """
    settings = get_settings()
    ext = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""

    if ext not in SUPPORTED_EXTENSIONS:
        raise UnsupportedFileTypeError(filename)

    logger.info("Starting document chunking", filename=filename, ext=ext)

    if ext == ".pdf":
        raw_text = _extract_text_from_pdf(file_bytes)
    else:
        raw_text = file_bytes.decode("utf-8", errors="replace")

    raw_chunks = _split_text(raw_text, settings.CHUNK_SIZE, settings.CHUNK_OVERLAP)

    result = [
        TextChunk(
            content=chunk,
            chunk_index=i,
            metadata={"source_ext": ext, "chunk_chars": len(chunk)},
        )
        for i, chunk in enumerate(raw_chunks)
    ]

    logger.info(
        "Chunking complete",
        filename=filename,
        total_chars=len(raw_text),
        chunks=len(result),
    )
    return result
