"""
quiz_tutor.py — Quiz ask-AI generation and streaming via shared LLM backend.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import AsyncIterator, Dict, List, Optional

import httpx
from loguru import logger

from ..config import get_settings
from .llm import (
    build_chat_completions_url,
    extract_llm_text,
    get_shared_llm,
    resolve_llm_api_key,
    resolve_retry_policy,
    sleep_before_retry,
)
from .tutor_chat import sanitize_chat_input, validate_chat_input


def _resolve_quiz_tutor_model() -> str:
    settings = get_settings()
    return settings.exercise_llm_model or settings.llm_model or "gemini-3.0-pro"


def _build_headers(*, accept: str) -> Dict[str, str]:
    headers = {
        "Content-Type": "application/json",
        "Accept": accept,
    }
    api_key = resolve_llm_api_key()
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


def _extract_completion_text(payload: Dict) -> str:
    choices = payload.get("choices") or []
    if not choices:
        return ""
    message = choices[0].get("message") or {}
    return extract_llm_text(message.get("content"))


def _summarize_chat_history(chat_history: List[Dict[str, str]]) -> str:
    if not chat_history:
        return ""

    llm = get_shared_llm()
    chat_text = "\n\n".join(
        f"{msg.get('role', 'user')}: {msg.get('content', '')}"
        for msg in chat_history[-6:]
        if msg.get("content")
    )
    if not chat_text:
        return ""

    result = llm.invoke([
        (
            "system",
            "Bạn tóm tắt hội thoại học tập ngắn gọn, chỉ giữ lại nội dung cần thiết để tiếp tục giải thích bài.",
        ),
        (
            "human",
            (
                "Tóm tắt hội thoại sau trong 2-3 câu, tập trung vào khái niệm đã bàn và điểm học sinh còn vướng:\n\n"
                f"{chat_text}"
            ),
        ),
    ])
    return extract_llm_text(result.content)


def _build_contextual_info(chat_history: List[Dict[str, str]]) -> str:
    if len(chat_history) > 6:
        summary = _summarize_chat_history(chat_history)
        if summary:
            return f"\nTÓM TẮT HỘI THOẠI TRƯỚC:\n{summary}\n"
        return ""

    if not chat_history:
        return ""

    turns = "\n\n".join(
        f"{msg.get('role', 'user')}: {msg.get('content', '')}"
        for msg in chat_history
        if msg.get("content")
    )
    if not turns:
        return ""
    return f"\nHỘI THOẠI TRƯỚC:\n{turns}\n"


def _build_messages(
    *,
    question: str,
    options: List[str],
    user_question: Optional[str],
    chat_history: List[Dict[str, str]],
    question_image: Optional[str],
    option_images: Optional[List[Optional[str]]],
) -> List[Dict]:
    contextual_info = _build_contextual_info(chat_history)
    sanitized_question = sanitize_chat_input(user_question) if user_question else ""
    prompt = (
        "CÂU HỎI QUIZ:\n"
        f"{question}\n\n"
        "ĐÁP ÁN:\n"
        f"{chr(10).join(f'{chr(65 + idx)}. {option}' for idx, option in enumerate(options))}\n"
        f"{contextual_info}\n"
        f"{f'CÂU HỎI MỚI CỦA HỌC SINH: {sanitized_question}' if sanitized_question else 'HÃY GIẢI THÍCH TỔNG QUÁT CÂU HỎI NÀY CHO HỌC SINH.'}\n\n"
        "YÊU CẦU TRẢ LỜI:\n"
        "- Chỉ giải thích xoay quanh câu hỏi quiz hiện tại và kiến thức liên quan trực tiếp.\n"
        "- Trả lời bằng tiếng Việt tự nhiên, rõ ràng, thân thiện.\n"
        "- Không tiết lộ đáp án theo kiểu chốt nhanh nếu học sinh chưa hỏi trực tiếp; ưu tiên giải thích để hiểu bản chất.\n"
        "- Nếu cần viết công thức toán, bắt buộc dùng LaTeX với $...$ hoặc $$...$$.\n"
        "- Có thể dùng bullet ngắn nếu giúp dễ hiểu hơn.\n"
    )

    user_content: List[Dict] = [{"type": "text", "text": prompt}]
    if question_image:
        user_content.append({
            "type": "image_url",
            "image_url": {"url": question_image},
        })

    for option_image in option_images or []:
        if option_image:
            user_content.append({
                "type": "image_url",
                "image_url": {"url": option_image},
            })

    return [
        {
            "role": "system",
            "content": (
                "Bạn là Rin-chan, gia sư giúp học sinh hiểu câu hỏi trắc nghiệm. "
                "Chỉ thảo luận về bài hiện tại, giữ giọng thân thiện nhưng đi thẳng vào giải thích."
            ),
        },
        {
            "role": "user",
            "content": user_content,
        },
    ]


def _build_payload(
    *,
    question: str,
    options: List[str],
    user_question: Optional[str],
    chat_history: List[Dict[str, str]],
    question_image: Optional[str],
    option_images: Optional[List[Optional[str]]],
    stream: bool,
) -> Dict:
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
    options: List[str],
    user_question: Optional[str],
    chat_history: Optional[List[Dict[str, str]]] = None,
    question_image: Optional[str] = None,
    option_images: Optional[List[Optional[str]]] = None,
) -> Dict:
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
    started_at = datetime.now(timezone.utc).isoformat()

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
                if len(explanation) < 20:
                    raise ValueError("Chat explanation too short")

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
            except Exception as exc:
                logger.warning(
                    f"[LLM] ⚠ quiz tutor attempt {attempt}/{max_retries} failed "
                    f"(will_retry={attempt < max_retries}): {exc}"
                )
                if attempt < max_retries:
                    sleep_before_retry(attempt, backoff_sec)

    raise RuntimeError("Quiz tutor service is temporarily unavailable")


async def create_quiz_tutor_stream(
    *,
    question: str,
    options: List[str],
    user_question: Optional[str],
    chat_history: Optional[List[Dict[str, str]]] = None,
    question_image: Optional[str] = None,
    option_images: Optional[List[Optional[str]]] = None,
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

    if response.status_code >= 400:
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
