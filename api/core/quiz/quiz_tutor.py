"""
quiz_tutor.py — Quiz ask-AI generation and streaming via the official OpenAI Responses API.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import httpx
from langchain_core.messages import HumanMessage, SystemMessage
from loguru import logger

from api.config import get_settings
from api.core.shared.llm import (
    awith_llm_retry,
    extract_llm_text,
    get_llm,
    serialize_responses_sse_event,
    with_llm_retry,
)

from .tutor_chat import (
    TUTOR_SYSTEM_PROMPT,
    _extract_stream_chunk_text,
    build_tutor_prompt,
    validate_chat_input,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

_MIN_EXPLANATION_LENGTH = 20


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
) -> list[SystemMessage | HumanMessage]:
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
        {"type": "image", "url": image_url} for image_url in (option_images or []) if image_url
    )

    return [
        SystemMessage(content=TUTOR_SYSTEM_PROMPT),
        HumanMessage(content=user_content),  # type: ignore[arg-type]
    ]


def _request_quiz_tutor_text(
    *,
    model: str,
    input_messages: list[SystemMessage | HumanMessage],
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
    input_messages: list[SystemMessage | HumanMessage],
    timeout_sec: float,
) -> Any:
    stream_timeout = httpx.Timeout(timeout_sec, read=None)

    async def _try():
        llm = get_llm(
            model=model,
            temperature=0.7,
            timeout=stream_timeout,
            streaming=True,
            use_responses_api=True,
        )
        return llm.astream(input_messages)

    return await awith_llm_retry(label="quiz tutor stream", fn=_try)


def generate_quiz_tutor_response(
    *,
    question: str,
    options: list[str],
    user_question: str | None,
    chat_history: list[dict[str, str]] | None = None,
    question_image: str | None = None,
    option_images: list[str | None] | None = None,
) -> dict[str, bool | dict[str, str | int | None]]:
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

    def _try():
        explanation = _request_quiz_tutor_text(
            model=model,
            input_messages=input_messages,
            timeout_sec=settings.llm_timeout_sec,
        )
        if len(explanation) < _MIN_EXPLANATION_LENGTH:
            raise ValueError("explanation too short")
        return explanation

    explanation = with_llm_retry(label="quiz tutor", fn=_try)
    logger.info("[LLM] ✓ Quiz tutor chat generated")
    return {
        "success": True,
        "data": {
            "explanation": explanation,
            "structured": None,
            "timestamp": datetime.now(UTC).isoformat(),
            "turn_count": (len(chat_history or []) // 2) + 1,
        },
    }


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
