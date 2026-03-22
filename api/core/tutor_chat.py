"""
tutor_chat.py — Adaptive tutor-chat prompt and validation logic.
"""

from __future__ import annotations

import asyncio
import re
import time
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


TUTOR_SYSTEM_PROMPT = (
    "Bạn là Rin-chan, gia sư giúp học sinh hiểu câu hỏi trắc nghiệm. "
    "Chỉ thảo luận về bài hiện tại, giữ giọng thân thiện nhưng đi thẳng vào giải thích."
)

TUTOR_RESPONSE_REQUIREMENTS = (
    "YÊU CẦU TRẢ LỜI:\n"
    "- Chỉ giải thích xoay quanh câu hỏi quiz hiện tại và kiến thức liên quan trực tiếp.\n"
    "- Trả lời bằng tiếng Việt tự nhiên, rõ ràng, thân thiện.\n"
    "- Không tiết lộ đáp án theo kiểu chốt nhanh nếu học sinh chưa hỏi trực tiếp; ưu tiên giải thích để hiểu bản chất.\n"
    "- Nếu cần viết công thức toán, bắt buộc dùng LaTeX với $...$ hoặc $$...$$.\n"
    "- Có thể dùng bullet ngắn nếu giúp dễ hiểu hơn.\n"
)


def sanitize_chat_input(input_text: str) -> str:
    return (
        input_text
        .replace("<", "")
        .replace(">", "")
        .strip()[:1000]
    )


def validate_chat_input(user_question: str) -> Optional[str]:
    sanitized = sanitize_chat_input(user_question)
    suspicious_patterns = [
        r"ignore\s+(previous|above|all)\s+(instructions?|prompts?)",
        r"you\s+are\s+now\s+",
        r"forget\s+(everything|all|previous)",
        r"act\s+as\s+(?!.*tutor)",
        r"roleplay|role\s*play",
        r"pretend\s+to\s+be",
        r"system\s*:|admin\s*:|root\s*:",
        r"<script|javascript|eval\(",
    ]
    off_topic_patterns = [
        r"(hack|crack|break)\s+into",
        r"personal\s+information",
        r"phone\s+number|address|email",
    ]

    for pattern in suspicious_patterns + off_topic_patterns:
        if re.search(pattern, sanitized, flags=re.IGNORECASE):
            return (
                "Rin-chan chỉ hỗ trợ giải thích bài hiện tại. "
                "Hãy hỏi về câu hỏi trắc nghiệm hoặc khái niệm liên quan nhé."
            )
    return None


def summarize_chat_history(chat_history: List[Dict[str, str]]) -> str:
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


def build_chat_context(chat_history: Optional[List[Dict[str, str]]]) -> str:
    history = chat_history or []
    if len(history) > 6:
        summary = summarize_chat_history(history)
        if summary:
            return f"\nTÓM TẮT HỘI THOẠI TRƯỚC:\n{summary}\n"
        return ""

    if not history:
        return ""

    turns = "\n\n".join(
        f"{msg.get('role', 'user')}: {msg.get('content', '')}"
        for msg in history
        if msg.get("content")
    )
    if turns:
        return f"\nHỘI THOẠI TRƯỚC:\n{turns}\n"
    return ""


def build_tutor_prompt(
    *,
    question: str,
    options: List[str],
    user_question: Optional[str],
    chat_history: Optional[List[Dict[str, str]]] = None,
    concept_name: Optional[str] = None,
    bloom_level: Optional[int] = None,
    general_instruction: str = "HÃY GIẢI THÍCH TỔNG QUÁT CÂU HỎI NÀY CHO HỌC SINH.",
) -> str:
    contextual_info = build_chat_context(chat_history)
    sanitized_question = sanitize_chat_input(user_question) if user_question else ""
    learner_prompt = (
        f"CÂU HỎI MỚI CỦA HỌC SINH: {sanitized_question}"
        if sanitized_question
        else general_instruction
    )

    concept_block = ""
    if concept_name is not None or bloom_level is not None:
        concept_block = (
            f"KHÁI NIỆM: {concept_name or 'Không rõ'}\n"
            f"MỨC BLOOM: {bloom_level if bloom_level is not None else 'Không rõ'}\n"
        )

    return (
        "CÂU HỎI QUIZ:\n"
        f"{question}\n\n"
        "ĐÁP ÁN:\n"
        f"{chr(10).join(f'{chr(65 + idx)}. {option}' for idx, option in enumerate(options))}\n\n"
        f"{concept_block}"
        f"{contextual_info}\n"
        f"{learner_prompt}\n\n"
        f"{TUTOR_RESPONSE_REQUIREMENTS}"
    )


def _resolve_tutor_model() -> str:
    settings = get_settings()
    return settings.exercise_llm_model or settings.llm_model or "gemini-3.0-pro"


def _build_stream_headers(*, accept: str) -> Dict[str, str]:
    headers = {
        "Content-Type": "application/json",
        "Accept": accept,
    }
    api_key = resolve_llm_api_key()
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


async def create_tutor_chat_stream(
    *,
    question: str,
    options: List[str],
    user_question: str,
    chat_history: Optional[List[Dict[str, str]]] = None,
    concept_name: Optional[str] = None,
    bloom_level: Optional[int] = None,
) -> AsyncIterator[bytes]:
    validation_error = validate_chat_input(user_question)
    if validation_error:
        raise ValueError(validation_error)

    settings = get_settings()
    endpoint = build_chat_completions_url(settings.llm_base_url)
    prompt = await asyncio.to_thread(
        build_tutor_prompt,
        question=question,
        options=options,
        user_question=user_question,
        chat_history=chat_history,
        concept_name=concept_name,
        bloom_level=bloom_level,
    )
    payload = {
        "model": _resolve_tutor_model(),
        "temperature": 0.7,
        "stream": True,
        "messages": [
            {
                "role": "system",
                "content": TUTOR_SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
    }

    timeout = httpx.Timeout(settings.llm_timeout_sec, read=None)
    client = httpx.AsyncClient(timeout=timeout)
    request = client.build_request(
        "POST",
        endpoint,
        headers=_build_stream_headers(accept="text/event-stream"),
        json=payload,
    )
    response = await client.send(request, stream=True)

    if response.status_code >= 400:
        error_body = (await response.aread()).decode("utf-8", errors="replace")
        await response.aclose()
        await client.aclose()
        raise RuntimeError(
            error_body or f"Tutor chat streaming failed with status {response.status_code}"
        )

    async def iterator() -> AsyncIterator[bytes]:
        try:
            async for chunk in response.aiter_bytes():
                yield chunk
        finally:
            await response.aclose()
            await client.aclose()

    return iterator()


def generate_tutor_chat_response(
    question: str,
    options: List[str],
    user_question: str,
    chat_history: Optional[List[Dict[str, str]]] = None,
    concept_name: Optional[str] = None,
    bloom_level: Optional[int] = None,
) -> str:
    validation_error = validate_chat_input(user_question)
    if validation_error:
        raise ValueError(validation_error)

    llm = get_shared_llm()
    prompt = build_tutor_prompt(
        question=question,
        options=options,
        user_question=user_question,
        chat_history=chat_history,
        concept_name=concept_name,
        bloom_level=bloom_level,
    )

    t0 = time.time()
    max_retries, backoff_sec = resolve_retry_policy()
    for attempt in range(1, max_retries + 1):
        try:
            result = llm.invoke([
                (
                    "system",
                    TUTOR_SYSTEM_PROMPT,
                ),
                ("human", prompt),
            ])
            explanation = extract_llm_text(result.content)
            if len(explanation) < 20:
                raise ValueError("Chat explanation too short")
            logger.info(f"[LLM] ✓ Tutor chat generated in {time.time() - t0:.2f}s")
            return explanation
        except Exception as exc:
            logger.warning(
                f"[LLM] ⚠ tutor chat attempt {attempt}/{max_retries} failed "
                f"(will_retry={attempt < max_retries}): {exc}"
            )
            if attempt < max_retries:
                sleep_before_retry(attempt, backoff_sec)

    raise RuntimeError("Tutor chat service is temporarily unavailable")
