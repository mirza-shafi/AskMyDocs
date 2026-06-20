"""Groq LLM wrapper with citation-enforcing system prompt.

Wraps the Groq Python SDK to provide a clean async interface. The system
prompt forces the model to cite every factual claim using [SOURCE_ID] tags
and to refuse answering from outside the provided context.
"""

from __future__ import annotations

import structlog
from groq import AsyncGroq

from app.core.config import get_settings
from app.core.exceptions import LLMError

logger = structlog.get_logger(__name__)

SYSTEM_PROMPT = """\
You are a precise, citation-focused document assistant. Answer the user's
question clearly and helpfully using ONLY the CONTEXT CHUNKS provided.

OUTPUT STYLE:
- Lead with a direct 1-2 sentence answer or summary.
- Use short paragraphs; use a Markdown bullet list ("- ") only when listing
  multiple distinct items. Use **bold** for key terms when helpful.
- Be concise. Do not repeat the same point. Do not pad the answer.

CITATIONS:
- Cite with inline tags in the exact format [Sn] (e.g. [S1]), where n is the
  source id given to you in the context.
- Cite each distinct fact ONCE, at the end of the sentence or bullet that
  states it. Do NOT attach a citation to every phrase, and do NOT stack the
  same citation repeatedly (write [S1], not [S1][S1]).

GROUNDING RULES:
- Use only the provided context. Never use outside knowledge or hallucinate.
- If the context does not contain the answer, reply with exactly:
  "I don't know based on the provided documents."
- If the user asks for a summary or overview of the document, synthesize a
  clear, well-organized overview of what the context actually contains.
"""

# Reuse a single AsyncGroq client (and its underlying httpx connection pool)
# across requests instead of constructing one per call, which leaks connections.
_client: AsyncGroq | None = None


def _get_client() -> AsyncGroq:
    """Return a lazily-created, shared AsyncGroq client."""
    global _client  # noqa: PLW0603
    if _client is None:
        _client = AsyncGroq(api_key=get_settings().GROQ_API_KEY)
    return _client


def _build_user_message(question: str, context_blocks: list[str]) -> str:
    """Assemble the user-turn message with numbered context chunks.

    Args:
        question: The user's natural-language question.
        context_blocks: List of chunk texts. Each will be prefixed with its
            SOURCE_ID so the LLM can cite it.

    Returns:
        Formatted user message string.
    """
    numbered_context = "\n\n".join(
        f"[S{i + 1}] {block}" for i, block in enumerate(context_blocks)
    )
    return (
        f"CONTEXT CHUNKS:\n{numbered_context}\n\n"
        f"QUESTION: {question}\n\n"
        "ANSWER (cite every claim with [S<n>]):"
    )


async def generate_answer(
    question: str,
    context_blocks: list[str],
    max_tokens: int = 1024,
) -> str:
    """Generate a grounded, cited answer using the Groq LLM.

    Args:
        question: The user's question.
        context_blocks: Ordered list of context passages (top-N after reranking).
        max_tokens: Maximum tokens in the completion. Raise this for exhaustive
            "list everything" answers that would otherwise be truncated.

    Returns:
        The LLM-generated answer string with inline citations.

    Raises:
        LLMError: If the Groq API call fails or returns an empty response.
    """
    settings = get_settings()
    client = _get_client()

    user_message = _build_user_message(question, context_blocks)

    logger.debug(
        "Calling Groq LLM",
        model=settings.GROQ_MODEL,
        context_chunks=len(context_blocks),
    )

    try:
        response = await client.chat.completions.create(
            model=settings.GROQ_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            temperature=0.1,   # Low temperature for factual, deterministic output
            max_tokens=max_tokens,
        )
    except Exception as exc:
        raise LLMError(f"Groq API call failed: {exc}") from exc

    answer = response.choices[0].message.content
    if not answer or not answer.strip():
        raise LLMError("Groq returned an empty response")

    logger.debug("LLM response received", answer_len=len(answer))
    return answer.strip()
