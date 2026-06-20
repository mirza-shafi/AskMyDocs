"""Document chunker: converts PDF/TXT files into overlapping text chunks.

Supports:
  - PDF files (via pdfplumber, preserves layout better than PyPDF2)
  - Plain text files (.txt)

The chunking strategy is structure-aware: it recursively splits text along
the strongest natural boundary available (paragraph → line → sentence →
clause → word → character), then greedily packs those units into
overlapping chunks. This keeps coherent units — table rows, list items and
whole sentences — intact instead of cutting through them mid-token.
"""

from __future__ import annotations

import io
import re
from dataclasses import dataclass, field

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


# Structural separators in descending priority. Splitting prefers the
# highest-level boundary that exists in the text so that semantically
# coherent units stay intact:
#   "\n\n"  paragraph / section break
#   "\n"    line break (keeps table rows and list items whole)
#   ". "/"? "/"! "/"; "  sentence boundaries
#   ", "    clause boundary
#   " "     word boundary (last resort before hard character split)
#   ""      hard character split (only for unbreakable runs)
_SEPARATORS: tuple[str, ...] = (
    "\n\n",
    "\n",
    ". ",
    "? ",
    "! ",
    "; ",
    ", ",
    " ",
    "",
)


def _recursive_split(
    text: str,
    chunk_size: int,
    separators: tuple[str, ...],
) -> list[str]:
    """Recursively break text into atomic fragments no larger than ``chunk_size``.

    Tries the highest-priority separator present in ``text``; any resulting
    fragment still longer than ``chunk_size`` is split again with the next
    separator down. The chosen separator is re-attached to each fragment so
    the original text is preserved when fragments are later re-joined.

    Args:
        text: The text to fragment.
        chunk_size: Maximum character length of a fragment.
        separators: Ordered candidate separators (highest priority first).

    Returns:
        List of fragments, each ``<= chunk_size`` unless a single token is
        itself longer than ``chunk_size``.
    """
    # Pick the first separator that actually occurs in the text.
    separator = separators[-1]
    remaining = separators[len(separators) - 1 :]
    for i, sep in enumerate(separators):
        if sep == "":
            separator = sep
            remaining = ()
            break
        if sep in text:
            separator = sep
            remaining = separators[i + 1 :]
            break

    # Hard character split — the unbreakable base case.
    if separator == "":
        return [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]

    pieces = text.split(separator)
    fragments: list[str] = []
    for idx, piece in enumerate(pieces):
        # Re-attach the separator to every fragment except the last so that
        # joining fragments reconstructs the source text faithfully.
        fragment = piece + (separator if idx < len(pieces) - 1 else "")
        if not fragment:
            continue
        if len(fragment) <= chunk_size:
            fragments.append(fragment)
        elif remaining:
            fragments.extend(_recursive_split(fragment, chunk_size, remaining))
        else:
            fragments.extend(
                fragment[i : i + chunk_size]
                for i in range(0, len(fragment), chunk_size)
            )
    return fragments


def _merge_fragments(
    fragments: list[str],
    chunk_size: int,
    chunk_overlap: int,
) -> list[str]:
    """Greedily pack structural fragments into chunks with overlap.

    Fragments are concatenated until adding the next one would exceed
    ``chunk_size``. When a chunk is emitted, a tail of recent fragments
    totalling up to ``chunk_overlap`` characters is carried into the next
    chunk to preserve context across boundaries.

    Args:
        fragments: Atomic fragments from ``_recursive_split`` (each <= chunk_size).
        chunk_size: Maximum character length per chunk.
        chunk_overlap: Target overlap (chars) carried between consecutive chunks.

    Returns:
        Ordered list of non-empty chunk strings.
    """
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for frag in fragments:
        frag_len = len(frag)

        if current and current_len + frag_len > chunk_size:
            chunks.append("".join(current).strip())

            # Retain a trailing window of fragments (<= chunk_overlap chars)
            # as overlap for the next chunk; drop the rest from the front.
            while current and current_len > chunk_overlap:
                if len(current) == 1:
                    # A single fragment larger than the overlap budget — no
                    # overlap is carried to avoid duplicating a whole fragment.
                    current = []
                    current_len = 0
                    break
                removed = current.pop(0)
                current_len -= len(removed)

        current.append(frag)
        current_len += frag_len

    if current:
        chunks.append("".join(current).strip())

    return [c for c in chunks if c]


# Matches C0 control characters that PostgreSQL text columns cannot store
# (notably NUL / U+0000), while preserving tab (\t), newline (\n) and CR (\r).
_CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")


def _sanitize_text(text: str) -> str:
    """Strip NUL bytes and disallowed control characters from extracted text.

    PostgreSQL rejects U+0000 in ``text``/``varchar`` columns; some PDFs embed
    stray nulls (common with non-Latin scripts), which otherwise crash the
    INSERT with ``CharacterNotInRepertoireError``.

    Args:
        text: Raw extracted text.

    Returns:
        Text with NUL and other C0 control characters removed.
    """
    return _CONTROL_CHARS_RE.sub("", text)


def _split_text(
    text: str,
    chunk_size: int,
    chunk_overlap: int,
) -> list[str]:
    """Split text into overlapping, structure-aware chunks.

    Strategy (recursive, structure-aware):
      1. Normalise excess blank lines.
      2. Recursively fragment the text along the strongest structural
         boundary available (paragraph → line → sentence → clause → word →
         character), so coherent units (table rows, sentences) stay intact.
      3. Greedily merge fragments into chunks up to ``chunk_size`` with a
         ``chunk_overlap`` context window carried between chunks.

    Args:
        text: The full document text.
        chunk_size: Target character length per chunk.
        chunk_overlap: Number of characters shared between consecutive chunks.

    Returns:
        List of non-empty text chunks.
    """
    # Remove NUL bytes and other C0 control chars (except tab/newline/CR).
    # PostgreSQL text columns reject U+0000, and some PDFs (e.g. non-Latin
    # scripts) embed stray nulls during extraction.
    text = _sanitize_text(text)
    # Normalise whitespace — collapse 3+ consecutive newlines into one blank line.
    text = re.sub(r"\n{3,}", "\n\n", text).strip()

    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]

    fragments = _recursive_split(text, chunk_size, _SEPARATORS)
    return _merge_fragments(fragments, chunk_size, chunk_overlap)


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


# ── Parent-child (hierarchical) chunking ──────────────────────────────────────


@dataclass(frozen=True, slots=True)
class ChildChunk:
    """A small, precision-focused chunk that gets embedded and searched.

    Attributes:
        content: Raw text of the child chunk.
        chunk_index: Zero-based position of the child within its parent.
        metadata: Extra context (source extension, char count, …).
    """

    content: str
    chunk_index: int
    metadata: dict


@dataclass(frozen=True, slots=True)
class ParentChunk:
    """A larger context chunk handed to the LLM, holding its child chunks.

    Children are embedded/searched for retrieval precision; when any child is
    retrieved, this parent's full ``content`` is supplied to the LLM so the
    model sees broader, coherent context (parent-child retrieval).

    Attributes:
        content: Raw text of the parent (context) chunk.
        chunk_index: Zero-based position of the parent within the document.
        metadata: Extra context (source extension, char count, child count, …).
        children: Ordered child chunks derived from this parent.
    """

    content: str
    chunk_index: int
    metadata: dict
    children: list[ChildChunk] = field(default_factory=list)


def _read_raw_text(file_bytes: bytes, filename: str) -> tuple[str, str]:
    """Extract raw text from a supported file and return ``(text, ext)``.

    Args:
        file_bytes: Raw file content as bytes.
        filename: Original filename (used to detect file type).

    Returns:
        Tuple of (extracted_text, normalised_extension).

    Raises:
        UnsupportedFileTypeError: If the extension is not supported.
    """
    ext = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if ext not in SUPPORTED_EXTENSIONS:
        raise UnsupportedFileTypeError(filename)

    if ext == ".pdf":
        raw_text = _extract_text_from_pdf(file_bytes)
    else:
        raw_text = file_bytes.decode("utf-8", errors="replace")
    return raw_text, ext


def chunk_document_hierarchical(
    file_bytes: bytes,
    filename: str,
) -> list[ParentChunk]:
    """Parse a file into structure-aware parent chunks, each split into children.

    Both levels use the structure-aware recursive splitter:
      - Parents use ``CHUNK_SIZE`` / ``CHUNK_OVERLAP`` (broad LLM context).
      - Children use ``CHILD_CHUNK_SIZE`` / ``CHILD_CHUNK_OVERLAP`` (precise
        units that are embedded and searched).

    Args:
        file_bytes: Raw file content as bytes.
        filename: Original filename (used to detect MIME type).

    Returns:
        Ordered list of ``ParentChunk`` objects, each carrying its children.

    Raises:
        UnsupportedFileTypeError: If the file extension is unsupported.
    """
    settings = get_settings()
    raw_text, ext = _read_raw_text(file_bytes, filename)

    logger.info("Starting hierarchical chunking", filename=filename, ext=ext)

    parent_texts = _split_text(raw_text, settings.CHUNK_SIZE, settings.CHUNK_OVERLAP)

    parents: list[ParentChunk] = []
    total_children = 0
    for p_idx, parent_text in enumerate(parent_texts):
        child_texts = _split_text(
            parent_text,
            settings.CHILD_CHUNK_SIZE,
            settings.CHILD_CHUNK_OVERLAP,
        )
        children = [
            ChildChunk(
                content=child_text,
                chunk_index=c_idx,
                metadata={"source_ext": ext, "chunk_chars": len(child_text)},
            )
            for c_idx, child_text in enumerate(child_texts)
        ]
        total_children += len(children)
        parents.append(
            ParentChunk(
                content=parent_text,
                chunk_index=p_idx,
                metadata={
                    "source_ext": ext,
                    "chunk_chars": len(parent_text),
                    "child_count": len(children),
                },
                children=children,
            )
        )

    logger.info(
        "Hierarchical chunking complete",
        filename=filename,
        total_chars=len(raw_text),
        parents=len(parents),
        children=total_children,
    )
    return parents
