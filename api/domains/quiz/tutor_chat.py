"""
tutor_chat.py — Adaptive tutor-chat prompt and validation logic.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

from loguru import logger

from api.config import get_settings
from api.shared.llm import (
    LLMConfigurationError,
    _resolve_shared_llm_model,
    invoke_text_completion,
)
from api.shared.llm_usage import LlmAction

from .tutor_core import generate_tutor_text, stream_tutor_sse

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Awaitable, Callable

_CHAT_HISTORY_SUMMARIZE_THRESHOLD = 6
_SUMMARY_SYSTEM_PROMPT = "Bạn tóm tắt hội thoại học tập ngắn gọn, chỉ giữ lại nội dung cần thiết để tiếp tục giải thích bài."

TUTOR_SYSTEM_PROMPT = (
    "Bạn là Rin-chan, gia sư giúp học sinh hiểu câu hỏi trắc nghiệm. "
    "Chỉ thảo luận về bài hiện tại, giữ giọng thân thiện nhưng đi thẳng vào giải thích."
)

TUTOR_RESPONSE_REQUIREMENTS = (
    "YÊU CẦU TRẢ LỜI:\n"
    "- Chỉ giải thích xoay quanh câu hỏi quiz hiện tại và kiến thức liên quan trực tiếp.\n"
    "- Trả lời bằng tiếng Việt tự nhiên, rõ ràng, thân thiện.\n"
    "- Chỉ chào và tự giới thiệu tên ở lượt đầu tiên. Nếu phần ngữ cảnh có HỘI THOẠI TRƯỚC "
    "hoặc TÓM TẮT HỘI THOẠI TRƯỚC, nghĩa là cuộc trò chuyện đang tiếp diễn: vào thẳng phần "
    "giải thích, KHÔNG chào lại, KHÔNG tự giới thiệu lại.\n"
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


def _request_text_response(
    *,
    instructions: str,
    user_text: str,
    model: str,
    temperature: float,
    timeout_sec: float,
) -> str:
    return invoke_text_completion(
        messages=[
            {"role": "system", "content": instructions},
            {"role": "user", "content": user_text},
        ],
        model=model,
        temperature=temperature,
        timeout=timeout_sec,
        action=LlmAction.ADAPTIVE_TUTOR_CHAT,
    )


def summarize_chat_history(chat_history: list[dict[str, str]]) -> str:
    if not chat_history:
        return ""

    chat_text = "\n\n".join(
        f"{msg.get('role', 'user')}: {msg.get('content', '')}"
        for msg in chat_history[-6:]
        if msg.get("content")
    )
    if not chat_text:
        return ""

    try:
        return _request_text_response(
            instructions=_SUMMARY_SYSTEM_PROMPT,
            user_text=(
                "Tóm tắt hội thoại sau trong 2-3 câu, tập trung vào khái niệm đã bàn và điểm học sinh còn vướng:\n\n"
                f"{chat_text}"
            ),
            model=_resolve_tutor_model(),
            temperature=0.2,
            timeout_sec=get_settings().llm_timeout_sec,
        )
    except Exception:
        logger.exception("[TutorChat] Failed to summarize chat history")
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
    try:
        return _resolve_shared_llm_model(None)
    except LLMConfigurationError as exc:
        raise LLMConfigurationError(
            "LLM model is not set. Configure EXERCISE_LLM_MODEL or LLM_MODEL."
        ) from exc


def _tutor_chat_messages(prompt: str) -> list[dict[str, Any]]:
    return [
        {"role": "system", "content": TUTOR_SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]


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

    prompt = build_tutor_prompt(
        question=question,
        options=options,
        user_question=user_question,
        chat_history=chat_history,
        concept_name=concept_name,
        bloom_level=bloom_level,
        rag_context=rag_context,
    )
    return await stream_tutor_sse(
        input_messages=_tutor_chat_messages(prompt),
        model=_resolve_tutor_model(),
        timeout_sec=get_settings().llm_timeout_sec,
        action=LlmAction.ADAPTIVE_TUTOR_CHAT,
        on_complete=on_complete,
    )


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

    prompt = build_tutor_prompt(
        question=question,
        options=options,
        user_question=user_question,
        chat_history=chat_history,
        concept_name=concept_name,
        bloom_level=bloom_level,
        rag_context=rag_context,
    )
    return generate_tutor_text(
        input_messages=_tutor_chat_messages(prompt),
        model=_resolve_tutor_model(),
        timeout_sec=get_settings().llm_timeout_sec,
        action=LlmAction.ADAPTIVE_TUTOR_CHAT,
    )
