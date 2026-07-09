"""
tutor_core.py — Shared LLM plumbing for both tutor surfaces (adaptive + quiz).

Both tutors build their own messages (adaptive: text + RAG/concept context;
quiz: multimodal question/option images) then hand off here. Retry (including
the open→first-token stream phase) now lives in the LLM client, so this module
only shapes messages into a reply or an SSE stream.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from loguru import logger

from api.shared.llm import (
    ainvoke_text_completion,
    astream_text_completion,
    serialize_responses_sse_event,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Awaitable, Callable


async def generate_tutor_text(
    *,
    input_messages: list[dict[str, Any]],
    model: str,
    timeout_sec: float,
    action: str,
) -> str:
    """Non-stream tutor reply. The client retries transient failures."""
    return await ainvoke_text_completion(
        messages=input_messages,
        model=model,
        temperature=0.7,
        timeout=timeout_sec,
        action=action,
    )


async def stream_tutor_sse(
    *,
    input_messages: list[dict[str, Any]],
    model: str,
    timeout_sec: float,
    action: str,
    on_complete: Callable[[str], Awaitable[None]] | None = None,
) -> AsyncIterator[bytes]:
    """SSE stream shared by both tutors.

    The client retries the open→first-token phase; if it still yields nothing it
    raises before any byte is sent, and that propagates to the caller (surfaced
    as an unavailable error). Once the first token is sent the stream is
    committed: a later failure becomes a ``response.failed`` event.
    ``on_complete`` persists the full reply on a clean finish.
    """
    stream = astream_text_completion(
        messages=input_messages,
        model=model,
        temperature=0.7,
        timeout=timeout_sec,
        action=action,
    )
    stream_iter = stream.__aiter__()

    # First token: let a pre-first-token failure propagate (no byte sent yet).
    first_delta = await stream_iter.__anext__()

    async def iterator() -> AsyncIterator[bytes]:
        full_response = first_delta
        yield serialize_responses_sse_event(
            {"type": "response.output_text.delta", "delta": first_delta}
        )

        try:
            async for delta in stream_iter:
                if delta:
                    full_response += delta
                    yield serialize_responses_sse_event(
                        {"type": "response.output_text.delta", "delta": delta}
                    )
        except Exception as exc:
            yield serialize_responses_sse_event(
                {"type": "response.failed", "response": {"error": {"message": str(exc)}}}
            )
            return

        yield serialize_responses_sse_event({"type": "response.completed"})
        if on_complete and full_response.strip():
            try:
                await on_complete(full_response.strip())
            except Exception:
                logger.exception("[Tutor] Failed to run on_complete hook")

    return iterator()
