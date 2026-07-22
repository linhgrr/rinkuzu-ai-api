"""Shared LLM response and SSE plumbing for Ask Rin-chan."""

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
    max_tokens: int | None = None,
) -> str:
    """Non-stream tutor reply. The client retries transient failures."""
    return await ainvoke_text_completion(
        messages=input_messages,
        model=model,
        temperature=0.7,
        timeout=timeout_sec,
        max_tokens=max_tokens,
        action=action,
    )


async def stream_tutor_sse(
    *,
    input_messages: list[dict[str, Any]],
    model: str,
    timeout_sec: float,
    action: str,
    max_tokens: int | None = None,
    on_complete: Callable[[str], Awaitable[None]] | None = None,
) -> AsyncIterator[bytes]:
    """SSE stream shared by both tutors.

    The client retries the open→first-token phase inside the returned iterator
    so ``EventSourceResponse`` can cancel the upstream LLM call if the browser
    disconnects before the first token. Stream failures are surfaced as a
    ``response.failed`` event because the HTTP SSE response is already open.
    ``on_complete`` persists the full reply on a clean finish.
    """

    async def iterator() -> AsyncIterator[bytes]:
        full_response = ""
        stream = astream_text_completion(
            messages=input_messages,
            model=model,
            temperature=0.7,
            timeout=timeout_sec,
            max_tokens=max_tokens,
            action=action,
        )

        try:
            async for delta in stream:
                if delta:
                    full_response += delta
                    yield serialize_responses_sse_event(
                        {"type": "response.output_text.delta", "delta": delta}
                    )
        except GeneratorExit:
            raise
        except Exception as exc:
            yield serialize_responses_sse_event(
                {"type": "response.failed", "response": {"error": {"message": str(exc)}}}
            )
            return

        if not full_response.strip():
            yield serialize_responses_sse_event(
                {
                    "type": "response.failed",
                    "response": {"error": {"message": "LLM returned an empty completion"}},
                }
            )
            return

        yield serialize_responses_sse_event({"type": "response.completed"})
        if on_complete and full_response.strip():
            try:
                await on_complete(full_response.strip())
            except Exception:
                logger.exception("[Tutor] Failed to run on_complete hook")

    return iterator()
