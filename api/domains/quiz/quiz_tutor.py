"""
quiz_tutor.py — Quiz ask-AI generation and streaming via the shared tutor core.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from loguru import logger

from api.config import get_settings
from api.shared.llm_usage import LlmAction

from .tutor_chat import (
    TUTOR_SYSTEM_PROMPT,
    _resolve_tutor_model,
    build_tutor_prompt,
    validate_chat_input,
)
from .tutor_core import generate_tutor_text, stream_tutor_sse

if TYPE_CHECKING:
    from collections.abc import AsyncIterator


async def _build_input_message(
    *,
    question: str,
    options: list[str],
    user_question: str | None,
    chat_history: list[dict[str, str]],
    question_image: str | None,
    option_images: list[str | None] | None,
) -> list[dict[str, Any]]:
    prompt = await build_tutor_prompt(
        question=question,
        options=options,
        user_question=user_question,
        chat_history=chat_history,
    )

    user_content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
    if question_image:
        user_content.append({"type": "image", "url": question_image})

    user_content.extend(
        {"type": "image", "url": image_url} for image_url in (option_images or []) if image_url
    )

    return [
        {"role": "system", "content": TUTOR_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


async def generate_quiz_tutor_response(
    *,
    question: str,
    options: list[str],
    user_question: str | None,
    chat_history: list[dict[str, str]] | None = None,
    question_image: str | None = None,
    option_images: list[str | None] | None = None,
) -> dict[str, str | int | None]:
    if user_question:
        validation_error = validate_chat_input(user_question)
        if validation_error:
            raise ValueError(validation_error)

    settings = get_settings()
    input_messages = await _build_input_message(
        question=question,
        options=options,
        user_question=user_question,
        chat_history=chat_history or [],
        question_image=question_image,
        option_images=option_images,
    )

    explanation = await generate_tutor_text(
        input_messages=input_messages,
        model=_resolve_tutor_model(),
        timeout_sec=settings.llm_timeout_sec,
        action=LlmAction.QUIZ_TUTOR,
    )
    logger.info("[LLM] ✓ Quiz tutor chat generated")
    return {
        "explanation": explanation,
        "structured": None,
        "timestamp": datetime.now(UTC).isoformat(),
        "turn_count": (len(chat_history or []) // 2) + 1,
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
    input_messages = await _build_input_message(
        question=question,
        options=options,
        user_question=user_question,
        chat_history=chat_history or [],
        question_image=question_image,
        option_images=option_images,
    )

    return await stream_tutor_sse(
        input_messages=input_messages,
        model=_resolve_tutor_model(),
        timeout_sec=settings.llm_timeout_sec,
        action=LlmAction.QUIZ_TUTOR,
    )
