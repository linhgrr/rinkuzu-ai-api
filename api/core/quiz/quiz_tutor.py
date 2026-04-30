"""
quiz_tutor.py — Quiz ask-AI generation and streaming via shared LLM backend.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING

import httpx
from loguru import logger

from api.config import get_settings
from api.core.shared.llm import (
    build_chat_completions_url,
    extract_llm_text,
    resolve_llm_api_key,
    resolve_retry_policy,
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
_HTTP_ERROR_STATUS_THRESHOLD = 400


def _resolve_quiz_tutor_model() -> str:
    settings = get_settings()
    return settings.exercise_llm_model or settings.llm_model or "gemini-3.0-pro"


def _build_headers(*, accept: str) -> dict[str, str]:
    headers = {
        "Content-Type": "application/json",
        "Accept": accept,
    }
    api_key = resolve_llm_api_key()
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


def _extract_completion_text(payload: dict) -> str:
    choices = payload.get("choices") or []
    if not choices:
        return ""
    message = choices[0].get("message") or {}
    return extract_llm_text(message.get("content"))


def _build_messages(
    *,
    question: str,
    options: list[str],
    user_question: str | None,
    chat_history: list[dict[str, str]],
    question_image: str | None,
    option_images: list[str | None] | None,
) -> list[dict]:
    prompt = build_tutor_prompt(
        question=question,
        options=options,
        user_question=user_question,
        chat_history=chat_history,
    )

    user_content: list[dict] = [{"type": "text", "text": prompt}]
    if question_image:
        user_content.append({
            "type": "image_url",
            "image_url": {"url": question_image},
        })

    user_content.extend(
        {"type": "image_url", "image_url": {"url": img}}
        for img in (option_images or [])
        if img
    )

    return [
        {
            "role": "system",
            "content": TUTOR_SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": user_content,
        },
    ]


def _build_payload(
    *,
    question: str,
    options: list[str],
    user_question: str | None,
    chat_history: list[dict[str, str]],
    question_image: str | None,
    option_images: list[str | None] | None,
    stream: bool,
) -> dict:
    return {
        "model": _resolve_quiz_tutor_model(),
        "temperature": 0.7,
        "stream": stream,
        "messages": _build_messages(
            question=question,
            options=options,
            user_question=user_question,
            chat_history=chat_history,
            question_image=question_image,
            option_images=option_images,
        ),
    }


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
    endpoint = build_chat_completions_url(settings.llm_base_url)
    payload = _build_payload(
        question=question,
        options=options,
        user_question=user_question,
        chat_history=chat_history or [],
        question_image=question_image,
        option_images=option_images,
        stream=False,
    )

    max_retries, backoff_sec = resolve_retry_policy()
    timeout = httpx.Timeout(settings.llm_timeout_sec)
    started_at = datetime.now(UTC).isoformat()

    with httpx.Client(timeout=timeout) as client:
        for attempt in range(1, max_retries + 1):
            try:
                response = client.post(
                    endpoint,
                    headers=_build_headers(accept="application/json"),
                    json=payload,
                )
                response.raise_for_status()
                explanation = _extract_completion_text(response.json())
            except Exception as exc:
                logger.warning(
                    f"[LLM] ⚠ quiz tutor attempt {attempt}/{max_retries} failed "
                    f"(will_retry={attempt < max_retries}): {exc}"
                )
                if attempt < max_retries:
                    sleep_before_retry(attempt, backoff_sec)
                continue
            if len(explanation) < _MIN_EXPLANATION_LENGTH:
                logger.warning(
                    f"[LLM] ⚠ quiz tutor attempt {attempt}/{max_retries} failed "
                    f"(will_retry={attempt < max_retries}): explanation too short"
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
                    "turnCount": (len(chat_history or []) // 2) + 1,
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
    endpoint = build_chat_completions_url(settings.llm_base_url)
    payload = _build_payload(
        question=question,
        options=options,
        user_question=user_question,
        chat_history=chat_history or [],
        question_image=question_image,
        option_images=option_images,
        stream=True,
    )

    timeout = httpx.Timeout(settings.llm_timeout_sec, read=None)
    client = httpx.AsyncClient(timeout=timeout)
    request = client.build_request(
        "POST",
        endpoint,
        headers=_build_headers(accept="text/event-stream"),
        json=payload,
    )
    response = await client.send(request, stream=True)

    if response.status_code >= _HTTP_ERROR_STATUS_THRESHOLD:
        error_body = (await response.aread()).decode("utf-8", errors="replace")
        await response.aclose()
        await client.aclose()
        raise RuntimeError(error_body or f"Quiz tutor streaming failed with status {response.status_code}")

    async def iterator() -> AsyncIterator[bytes]:
        try:
            async for chunk in response.aiter_bytes():
                yield chunk
        finally:
            await response.aclose()
            await client.aclose()

    return iterator()
