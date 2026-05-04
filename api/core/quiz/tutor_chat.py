"""
tutor_chat.py — Adaptive tutor-chat prompt and validation logic.
"""

from __future__ import annotations

import asyncio
import codecs
import json
import re
import time
from typing import TYPE_CHECKING

import httpx
from loguru import logger

from api.config import get_settings
from api.core.shared.llm import (
    build_chat_completions_url,
    extract_llm_text,
    get_shared_llm,
    resolve_llm_api_key,
    resolve_retry_policy,
    sleep_before_retry,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Awaitable, Callable

_CHAT_HISTORY_SUMMARIZE_THRESHOLD = 6
_HTTP_CLIENT_ERROR_STATUS = 400
_MIN_EXPLANATION_LENGTH = 20

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
    return input_text.replace("<", "").replace(">", "").strip()[:1000]


def validate_chat_input(user_question: str) -> str | None:
    sanitized = sanitize_chat_input(user_question)
    suspicious_patterns = [
        r"ignore\b[\s\S]{0,120}\b(instructions?|prompts?)",
        r"you\s+are\s+now\s+",
        r"forget\b[\s\S]{0,120}\b(previous|everything|all)",
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


def normalize_chat_history(chat_history: list[dict[str, str]] | None) -> list[dict[str, str]]:
    normalized: list[dict[str, str]] = []

    for message in chat_history or []:
        role = message.get("role")
        if role not in {"user", "assistant"}:
            continue

        content = sanitize_chat_input(str(message.get("content", "")))
        if not content:
            continue

        if validate_chat_input(content):
            logger.warning("[TutorChat] Dropped suspicious historical chat message")
            continue

        normalized.append(
            {
                "role": role,
                "content": content,
            }
        )

    return normalized[-12:]


def summarize_chat_history(chat_history: list[dict[str, str]]) -> str:
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

    try:
        result = llm.invoke(
            [
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
            ]
        )
        return extract_llm_text(result.content)
    except Exception as exc:
        logger.warning(f"[TutorChat] Failed to summarize chat history: {exc}")
        return ""


def build_chat_context(chat_history: list[dict[str, str]] | None) -> str:
    history = normalize_chat_history(chat_history)
    if len(history) > _CHAT_HISTORY_SUMMARIZE_THRESHOLD:
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
    options: list[str],
    user_question: str | None,
    chat_history: list[dict[str, str]] | None = None,
    concept_name: str | None = None,
    bloom_level: int | None = None,
    rag_context: str = "",
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

    rag_block = ""
    if rag_context:
        rag_block = (
            f"NGỮ CẢNH TỪ TÀI LIỆU (dùng để trả lời chính xác):\n{rag_context}\n\n"
            "Nếu ngữ cảnh trên đủ để trả lời, hãy dùng nó. "
            "Nếu không đủ, bổ sung bằng kiến thức của bạn và ghi rõ đó là suy luận.\n\n"
        )

    return (
        "CÂU HỎI QUIZ:\n"
        f"{question}\n\n"
        "ĐÁP ÁN:\n"
        f"{chr(10).join(f'{chr(65 + idx)}. {option}' for idx, option in enumerate(options))}\n\n"
        f"{rag_block}"
        f"{concept_block}"
        f"{contextual_info}\n"
        f"{learner_prompt}\n\n"
        f"{TUTOR_RESPONSE_REQUIREMENTS}"
    )


def _resolve_tutor_model() -> str:
    settings = get_settings()
    return settings.exercise_llm_model or settings.llm_model or "gemini-3.0-pro"


def _build_stream_headers(*, accept: str) -> dict[str, str]:
    headers = {
        "Content-Type": "application/json",
        "Accept": accept,
    }
    api_key = resolve_llm_api_key()
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


def _extract_openai_delta_content(payload: dict) -> str:
    choices = payload.get("choices") or []
    if choices:
        delta = choices[0].get("delta") or {}
        content = delta.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, str):
                    parts.append(item)
                    continue
                if isinstance(item, dict) and isinstance(item.get("text"), str):
                    parts.append(item["text"])
            return "".join(parts)

    content = payload.get("content")
    return content if isinstance(content, str) else ""


def _split_sse_events(buffer: str) -> tuple[list[str], str]:
    events: list[str] = []
    last_index = 0

    for match in re.finditer(r"\r?\n\r?\n", buffer):
        events.append(buffer[last_index : match.start()])
        last_index = match.end()

    return events, buffer[last_index:]


def _parse_sse_event(event: str) -> tuple[str, bool]:
    data_lines = [
        line[5:].strip() for line in re.split(r"\r?\n", event) if line.startswith("data:")
    ]
    if not data_lines:
        return "", False

    data = "\n".join(data_lines)
    if data == "[DONE]":
        return "", True

    try:
        payload = json.loads(data)
    except json.JSONDecodeError as exc:
        raise RuntimeError("Tutor chat stream returned invalid SSE payload") from exc

    error = payload.get("error")
    if error:
        raise RuntimeError(error if isinstance(error, str) else "Tutor chat streaming failed")

    finish_reason = None
    choices = payload.get("choices") or []
    if choices:
        finish_reason = choices[0].get("finish_reason")

    return _extract_openai_delta_content(payload), payload.get(
        "done"
    ) is True or finish_reason is not None


async def _raise_if_error_status(response: httpx.Response) -> None:
    """Raise RuntimeError if the response has an HTTP error status."""
    if response.status_code >= _HTTP_CLIENT_ERROR_STATUS:
        error_body = (await response.aread()).decode("utf-8", errors="replace")
        await response.aclose()
        raise RuntimeError(
            error_body or f"Tutor chat streaming failed with status {response.status_code}"
        )


async def _connect_stream_with_retries(
    endpoint: str,
    payload: dict,
    headers: dict,
    http_timeout: httpx.Timeout,
    max_retries: int,
    backoff_sec: float,
) -> tuple[httpx.AsyncClient, httpx.Response]:
    """Attempt to open an SSE stream, retrying on transient failures."""
    for attempt in range(1, max_retries + 1):
        candidate_client = httpx.AsyncClient(timeout=http_timeout)
        try:
            request = candidate_client.build_request(
                "POST", endpoint, headers=headers, json=payload
            )
            candidate_response = await candidate_client.send(request, stream=True)
            await _raise_if_error_status(candidate_response)
        except Exception as exc:
            logger.warning(
                f"[LLM] ⚠ tutor chat stream attempt {attempt}/{max_retries} failed "
                f"(will_retry={attempt < max_retries}): {exc}"
            )
            await candidate_client.aclose()
            if attempt < max_retries:
                await asyncio.sleep(backoff_sec * attempt)
                continue
            if isinstance(exc, RuntimeError):
                raise
            raise RuntimeError("Tutor chat service is temporarily unavailable") from exc
        else:
            return candidate_client, candidate_response
    raise RuntimeError("Tutor chat service is temporarily unavailable")


async def _process_sse_buffer(
    buffer: str, full_response: str, *, saw_terminal: bool
) -> tuple[str, str, bool]:
    """Process all complete SSE events from buffer; return (remaining_buffer, full_response, saw_terminal)."""
    events, remaining = _split_sse_events(buffer)
    for event in events:
        delta, is_terminal = _parse_sse_event(event)
        if delta:
            full_response += delta
        if is_terminal:
            saw_terminal = True
    return remaining, full_response, saw_terminal


async def create_tutor_chat_stream(
    *,
    question: str,
    options: list[str],
    user_question: str,
    chat_history: list[dict[str, str]] | None = None,
    concept_name: str | None = None,
    bloom_level: int | None = None,
    rag_context: str = "",
    on_complete: Callable[[str], Awaitable[None]] | None = None,
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
        rag_context=rag_context,
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
    max_retries, backoff_sec = resolve_retry_policy()
    client, response = await _connect_stream_with_retries(
        endpoint,
        payload,
        _build_stream_headers(accept="text/event-stream"),
        http_timeout=timeout,
        max_retries=max_retries,
        backoff_sec=backoff_sec,
    )

    async def iterator() -> AsyncIterator[bytes]:
        decoder = codecs.getincrementaldecoder("utf-8")()
        buf = ""
        full_response = ""
        saw_terminal = False

        try:
            async for chunk in response.aiter_bytes():
                yield chunk
                buf += decoder.decode(chunk)
                buf, full_response, saw_terminal = await _process_sse_buffer(
                    buf, full_response, saw_terminal=saw_terminal
                )

            buf += decoder.decode(b"", final=True)
            buf, full_response, saw_terminal = await _process_sse_buffer(
                buf, full_response, saw_terminal=saw_terminal
            )

            if buf.strip():
                delta, is_terminal = _parse_sse_event(buf)
                if delta:
                    full_response += delta
                if is_terminal:
                    saw_terminal = True

            if not saw_terminal:
                raise RuntimeError("Tutor chat stream ended before completion signal")

            if on_complete and full_response.strip():
                try:
                    await on_complete(full_response.strip())
                except Exception as exc:
                    logger.warning(f"[TutorChat] Failed to persist chat history: {exc}")
        finally:
            await response.aclose()
            await client.aclose()

    return iterator()


def generate_tutor_chat_response(
    question: str,
    options: list[str],
    user_question: str,
    chat_history: list[dict[str, str]] | None = None,
    concept_name: str | None = None,
    bloom_level: int | None = None,
    rag_context: str = "",
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
        rag_context=rag_context,
    )

    t0 = time.time()
    max_retries, backoff_sec = resolve_retry_policy()
    for attempt in range(1, max_retries + 1):
        try:
            result = llm.invoke([("system", TUTOR_SYSTEM_PROMPT), ("human", prompt)])
            explanation = extract_llm_text(result.content)
        except Exception as exc:
            logger.warning(
                f"[LLM] ⚠ tutor chat attempt {attempt}/{max_retries} failed "
                f"(will_retry={attempt < max_retries}): {exc}"
            )
            if attempt < max_retries:
                sleep_before_retry(attempt, backoff_sec)
            continue
        else:
            if len(explanation) < _MIN_EXPLANATION_LENGTH:
                raise ValueError("Chat explanation too short")
            logger.info(f"[LLM] ✓ Tutor chat generated in {time.time() - t0:.2f}s")
            return explanation

    raise RuntimeError("Tutor chat service is temporarily unavailable")
