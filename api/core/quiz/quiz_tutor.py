"""
quiz_tutor.py — Quiz ask-AI generation and streaming via the official OpenAI Responses API.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import httpx
from langchain_core.messages import HumanMessage, SystemMessage
from loguru import logger

from api.config import get_settings
from api.core.shared.llm import (
    extract_llm_text,
    get_llm,
    resolve_retry_policy,
    serialize_responses_sse_event,
    sleep_before_retry,
)

from .tutor_chat import (
    TUTOR_SYSTEM_PROMPT,
    build_tutor_prompt,
    validate_chat_input,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

_MIN_EXPLANATION_LENGTH = 20


def _extract_stream_chunk_text(chunk: Any) -> str:
    text_accessor = getattr(chunk, "text", None)
    if text_accessor is not None:
        return str(text_accessor)
    content = getattr(chunk, "content", "")
    if isinstance(content, str):
        return content
    return extract_llm_text(content)


def _resolve_quiz_tutor_model() -> str:
    settings = get_settings()
    return settings.exercise_llm_model or settings.openai_model or "gpt-4o-mini"


def _build_input_message(
    *,
    question: str,
    options: list[str],
    user_question: str | None,
    chat_history: list[dict[str, str]],
    question_image: str | None,
    option_images: list[str | None] | None,
) -> list[Any]:
    prompt = build_tutor_prompt(
        question=question,
        options=options,
        user_question=user_question,
        chat_history=chat_history,
    )

    user_content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
    if question_image:
        user_content.append(
            {
                "type": "image",
                "url": question_image,
            }
        )

    user_content.extend(
        {"type": "image", "url": image_url}
        for image_url in (option_images or [])
        if image_url
    )

    return [
        SystemMessage(content=TUTOR_SYSTEM_PROMPT),
        HumanMessage(content=user_content),
    ]


def _request_quiz_tutor_text(
    *,
    model: str,
    input_messages: list[Any],
    timeout_sec: float,
) -> str:
    llm = get_llm(
        model=model,
        temperature=0.7,
        timeout=timeout_sec,
        use_responses_api=True,
    )
    response = llm.invoke(input_messages)
    return extract_llm_text(response.content)


async def _open_quiz_tutor_stream(
    *,
    model: str,
    input_messages: list[Any],
    timeout_sec: float,
) -> Any:
    max_retries, backoff_sec = resolve_retry_policy()
    stream_timeout = httpx.Timeout(timeout_sec, read=None)

    for attempt in range(1, max_retries + 1):
        llm = get_llm(
            model=model,
            temperature=0.7,
            timeout=stream_timeout,
            streaming=True,
            use_responses_api=True,
        )
        try:
            stream = llm.astream(input_messages)
        except Exception as exc:
            logger.warning(
                "[LLM] ⚠ quiz tutor stream attempt {}/{} failed "
                "(will_retry={}): {}",
                attempt,
                max_retries,
                attempt < max_retries,
                exc,
            )
            if attempt < max_retries:
                await asyncio.sleep(backoff_sec * attempt)
                continue
            raise RuntimeError("Quiz tutor streaming is temporarily unavailable") from exc
        return stream

    raise RuntimeError("Quiz tutor streaming is temporarily unavailable")


def generate_quiz_tutor_response(
    *,
    question: str,
    options: list[str],
    user_question: str | None,
    chat_history: list[dict[str, str]] | None = None,
    question_image: str | None = None,
    option_images: list[str | None] | None = None,
) -> dict:
    if user_question:
        validation_error = validate_chat_input(user_question)
        if validation_error:
            raise ValueError(validation_error)

    settings = get_settings()
    model = _resolve_quiz_tutor_model()
    input_messages = _build_input_message(
        question=question,
        options=options,
        user_question=user_question,
        chat_history=chat_history or [],
        question_image=question_image,
        option_images=option_images,
    )

    max_retries, backoff_sec = resolve_retry_policy()
    started_at = datetime.now(UTC).isoformat()

    for attempt in range(1, max_retries + 1):
        try:
            explanation = _request_quiz_tutor_text(
                model=model,
                input_messages=input_messages,
                timeout_sec=settings.llm_timeout_sec,
            )
        except Exception as exc:
            logger.warning(
                "[LLM] ⚠ quiz tutor attempt {}/{} failed "
                "(will_retry={}): {}",
                attempt,
                max_retries,
                attempt < max_retries,
                exc,
            )
            if attempt < max_retries:
                sleep_before_retry(attempt, backoff_sec)
            continue
        if len(explanation) < _MIN_EXPLANATION_LENGTH:
            logger.warning(
                "[LLM] ⚠ quiz tutor attempt {}/{} failed "
                "(will_retry={}): explanation too short",
                attempt,
                max_retries,
                attempt < max_retries,
            )
            if attempt < max_retries:
                sleep_before_retry(attempt, backoff_sec)
            continue
        logger.info("[LLM] ✓ Quiz tutor chat generated")
        return {
            "success": True,
            "data": {
                "explanation": explanation,
                "structured": None,
                "timestamp": started_at,
                "turn_count": (len(chat_history or []) // 2) + 1,
            },
        }

    raise RuntimeError("Quiz tutor service is temporarily unavailable")


async def create_quiz_tutor_stream(
    *,
    question: str,
    options: list[str],
    user_question: str | None,
    chat_history: list[dict[str, str]] | None = None,
    question_image: str | None = None,
    option_images: list[str | None] | None = None,
) -> AsyncIterator[bytes]:
    if user_question:
        validation_error = validate_chat_input(user_question)
        if validation_error:
            raise ValueError(validation_error)

    settings = get_settings()
    model = _resolve_quiz_tutor_model()
    input_messages = _build_input_message(
        question=question,
        options=options,
        user_question=user_question,
        chat_history=chat_history or [],
        question_image=question_image,
        option_images=option_images,
    )

    stream = await _open_quiz_tutor_stream(
        model=model,
        input_messages=input_messages,
        timeout_sec=settings.llm_timeout_sec,
    )

    async def iterator() -> AsyncIterator[bytes]:
        started_stream = False
        try:
            async for chunk in stream:
                started_stream = True
                delta = _extract_stream_chunk_text(chunk)
                if delta:
                    yield serialize_responses_sse_event(
                        {"type": "response.output_text.delta", "delta": delta}
                    )
        except Exception as exc:
            if not started_stream:
                raise RuntimeError("Quiz tutor streaming is temporarily unavailable") from exc
            yield serialize_responses_sse_event(
                {
                    "type": "response.failed",
                    "response": {"error": {"message": str(exc)}},
                }
            )
            return

        yield serialize_responses_sse_event({"type": "response.completed"})

    return iterator()
