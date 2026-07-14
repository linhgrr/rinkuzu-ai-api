"""
quiz_tutor.py — Quiz ask-AI generation and streaming via the shared tutor core.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING

from loguru import logger

from api.shared.llm_usage import LlmAction

from .tutor_chat import TutorChatRequestContext, get_tutor_chat_service

if TYPE_CHECKING:
    from collections.abc import AsyncIterator


async def generate_quiz_tutor_response(
    *,
    question: str,
    options: list[str],
    user_question: str | None,
    chat_history: list[dict[str, str]] | None = None,
    question_image: str | None = None,
    option_images: list[str | None] | None = None,
) -> dict[str, str | int | None]:
    explanation = await get_tutor_chat_service().generate_response(
        TutorChatRequestContext(
            action=LlmAction.QUIZ_TUTOR,
            question=question,
            options=options,
            user_question=user_question,
            chat_history=chat_history or [],
            question_image=question_image,
            option_images=option_images or [],
        )
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
    return await get_tutor_chat_service().create_stream(
        TutorChatRequestContext(
            action=LlmAction.QUIZ_TUTOR,
            question=question,
            options=options,
            user_question=user_question,
            chat_history=chat_history or [],
            question_image=question_image,
            option_images=option_images or [],
        )
    )
