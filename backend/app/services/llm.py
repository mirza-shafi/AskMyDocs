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
You are a precise, citation-focused document assistant.

Rules you MUST follow without exception:
1. Answer ONLY using information from the CONTEXT CHUNKS provided below.
2. Every factual claim MUST be followed by an inline citation in the exact
   format [SOURCE_ID] where SOURCE_ID is the chunk identifier given to you.
3. If multiple chunks support the same claim, cite all of them: [S1][S2].
4. If the answer cannot be found in the provided context, reply with exactly:
   "I don't know based on the provided documents."
5. NEVER use outside knowledge. NEVER hallucinate.
6. Be concise and structured. Use bullet points for lists.
"""


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
) -> str:
    """Generate a grounded, cited answer using the Groq LLM.

    Args:
        question: The user's question.
        context_blocks: Ordered list of context passages (top-N after reranking).

    Returns:
        The LLM-generated answer string with inline citations.

    Raises:
        LLMError: If the Groq API call fails or returns an empty response.
    """
    settings = get_settings()
    client = AsyncGroq(api_key=settings.GROQ_API_KEY)

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
            max_tokens=1024,
        )
    except Exception as exc:
        raise LLMError(f"Groq API call failed: {exc}") from exc

    answer = response.choices[0].message.content
    if not answer or not answer.strip():
        raise LLMError("Groq returned an empty response")

    logger.debug("LLM response received", answer_len=len(answer))
    return answer.strip()
