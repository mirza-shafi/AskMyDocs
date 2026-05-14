"""Local HuggingFace embedding service using sentence-transformers.

The model is loaded once at application startup (inside the FastAPI lifespan)
and held as a module-level singleton. This avoids ~2s model-load latency on
the first request and allows pre-warming on Render.

Encoding runs in a thread-pool executor (``asyncio.to_thread``) to avoid
blocking the async event loop during inference.
"""

from __future__ import annotations

import asyncio

import structlog
from sentence_transformers import SentenceTransformer

from app.core.config import get_settings
from app.core.exceptions import EmbeddingError

logger = structlog.get_logger(__name__)

# Module-level singleton — populated by ``initialise()`` at startup.
_model: SentenceTransformer | None = None


def initialise() -> None:
    """Load the embedding model into memory.

    Should be called once from the FastAPI lifespan ``startup`` handler.
    Subsequent calls are no-ops.
    """
    global _model  # noqa: PLW0603
    if _model is not None:
        return

    settings = get_settings()
    model_name = settings.EMBEDDING_MODEL
    logger.info("Loading embedding model", model=model_name)
    _model = SentenceTransformer(model_name)
    logger.info("Embedding model ready", model=model_name)


def _get_model() -> SentenceTransformer:
    """Return the loaded model, raising if it has not been initialised.

    Returns:
        The loaded ``SentenceTransformer`` instance.

    Raises:
        EmbeddingError: If ``initialise()`` has not been called.
    """
    if _model is None:
        raise EmbeddingError(
            "Embedding model is not initialised. "
            "Call embedder.initialise() during application startup."
        )
    return _model


async def embed_text(text: str) -> list[float]:
    """Encode a single string into a dense embedding vector.

    Runs in a thread-pool executor to avoid blocking the event loop.

    Args:
        text: The input string to embed.

    Returns:
        A 384-dimensional list of floats.

    Raises:
        EmbeddingError: If encoding fails for any reason.
    """
    try:
        model = _get_model()
        vector: list[float] = await asyncio.to_thread(
            lambda: model.encode(text, normalize_embeddings=True).tolist()
        )
        return vector
    except EmbeddingError:
        raise
    except Exception as exc:
        raise EmbeddingError(f"Embedding failed: {exc}") from exc


async def embed_batch(texts: list[str]) -> list[list[float]]:
    """Encode a list of strings in a single batched inference call.

    More efficient than calling ``embed_text`` in a loop.

    Args:
        texts: List of input strings.

    Returns:
        List of 384-dimensional float lists, same order as ``texts``.

    Raises:
        EmbeddingError: If encoding fails.
    """
    try:
        model = _get_model()
        vectors: list[list[float]] = await asyncio.to_thread(
            lambda: model.encode(
                texts, normalize_embeddings=True, show_progress_bar=False
            ).tolist()
        )
        return vectors
    except EmbeddingError:
        raise
    except Exception as exc:
        raise EmbeddingError(f"Batch embedding failed: {exc}") from exc
