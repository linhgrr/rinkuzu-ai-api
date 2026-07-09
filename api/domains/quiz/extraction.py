"""Shared quiz extraction helpers used by direct and draft APIs."""

from __future__ import annotations

import asyncio
import time
from typing import Literal, cast

from loguru import logger
from pydantic import BaseModel, ConfigDict, Field, model_validator

from api.config import Settings, get_settings
from api.shared.document_text import (
    DocumentTextConfigurationError,
    ExtractedDocumentText,
    build_ocr_api_config,
    extract_document_text_from_bytes,
)
from api.shared.llm import invoke_structured_completion, resolve_llm_api_key
from api.shared.retry import llm_retry_call

EXTRACTION_PROMPT = """
You are given educational content that may include questions, explanations, and references to images or diagrams.

Your task is to extract or generate quiz questions (both single-choice and multiple-choice) from this content.

If the content is clearly a list of questions (e.g., a quiz, test, or practice worksheet), you MUST extract them and format them directly.

Important Instructions:

1. **DO NOT OMIT ANY TEXT.** Even if the text refers to or is adjacent to an image, you must extract the full surrounding question text exactly as it appears.
2. **IGNORE IMAGES ENTIRELY.** Do not describe, summarize, or attempt to interpret any image content. Only process the visible text — even if it partially depends on an image.
3. Every question must have EXACTLY 4 to 5 options. If there are fewer, you must logically create plausible distractors to reach at least 4.
4. Provide the correct answer using a zero-based index (`correctIndex` or an array of `correctIndexes`).
5. **PRESERVE THE ORIGINAL LANGUAGE.** If questions are in Vietnamese, keep ALL text in Vietnamese. Do NOT translate.
6. **OPEN-ENDED QUESTIONS:** If a question is essay-style or open-ended, convert it into a multiple-choice format if possible. If not convertible, skip it entirely.
7. **MATH FORMATTING:** All mathematical expressions MUST use LaTeX notation. Inline: $...$ (e.g., $x^2 + 1$). Display: $$...$$ (e.g., $$\\Delta = b^2 - 4ac$$). Do NOT write formulas as plain text.

Return ONLY valid JSON.

User constraint overrides (if any, follow these above defaults):
<<<USER_PROMPT>>>
"""


class ExtractedQuizQuestion(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str = Field(min_length=1)
    type: Literal["single", "multiple"]
    options: list[str] = Field(min_length=4, max_length=5)
    correct_index: int | None = Field(default=None, alias="correctIndex")
    correct_indexes: list[int] = Field(default_factory=list, alias="correctIndexes")

    @model_validator(mode="after")
    def validate_answer_shape(self) -> ExtractedQuizQuestion:
        option_count = len(self.options)
        if self.type == "single":
            if self.correct_index is None:
                raise ValueError("single-choice questions require correctIndex")
            if self.correct_indexes:
                raise ValueError("single-choice questions must not include correctIndexes")
            if not 0 <= self.correct_index < option_count:
                raise ValueError("correctIndex is out of range")
            return self

        if self.correct_index is not None:
            raise ValueError("multiple-choice questions must not include correctIndex")
        if not self.correct_indexes:
            raise ValueError("multiple-choice questions require correctIndexes")
        if len(set(self.correct_indexes)) != len(self.correct_indexes):
            raise ValueError("correctIndexes must be unique")
        if any(index < 0 or index >= option_count for index in self.correct_indexes):
            raise ValueError("correctIndexes contains an out-of-range value")
        return self

    def to_public_dict(self) -> dict[str, str | int | list[str] | list[int]]:
        payload: dict[str, str | int | list[str] | list[int]] = {
            "question": self.question,
            "type": self.type,
            "options": self.options,
        }
        if self.type == "single":
            payload["correctIndex"] = cast("int", self.correct_index)
        else:
            payload["correctIndexes"] = self.correct_indexes
        return payload


class ExtractedQuizQuestionBatch(BaseModel):
    questions: list[ExtractedQuizQuestion] = Field(default_factory=list)


def build_extraction_prompt(user_prompt: str | None) -> str:
    """Render the extraction prompt with optional user constraints."""
    return EXTRACTION_PROMPT.replace(
        "<<<USER_PROMPT>>>",
        user_prompt or "No additional constraints.",
    )


async def invoke_document_text_extract_llm(
    *,
    document_text: ExtractedDocumentText,
    filename: str,
    prompt: str,
    model: str,
) -> list[dict[str, str | int | list[str] | list[int]]]:
    timeout_sec = max(1.0, float(get_settings().llm_timeout_sec))
    llm_start = time.perf_counter()
    questions = await asyncio.to_thread(
        _extract_questions_from_document_text_sync,
        document_text,
        filename,
        prompt,
        model,
        timeout_sec,
    )
    llm_duration_ms = int((time.perf_counter() - llm_start) * 1000)
    logger.info(
        "[quiz_extract] llm_request_done duration_ms={} extracted_questions={}",
        llm_duration_ms,
        len(questions),
    )
    return questions


def _extract_questions_from_pdf_bytes_sync(
    pdf_bytes: bytes,
    filename: str,
    prompt: str,
    model: str,
    timeout_sec: float,
) -> list[dict[str, str | int | list[str] | list[int]]]:
    document_text = extract_document_text_from_bytes(pdf_bytes, filename=filename)
    return _extract_questions_from_document_text_sync(
        document_text,
        filename,
        prompt,
        model,
        timeout_sec,
    )


def _clamp_document_text(text: str, max_chars: int) -> str:
    """Trim full document text to the single-call budget, warning on overflow."""
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    logger.warning(
        "[quiz_extract] document_text_truncated original_chars={} max_chars={}",
        len(text),
        max_chars,
    )
    return text[:max_chars]


def _extract_questions_from_document_text_sync(
    document_text: ExtractedDocumentText,
    filename: str,
    prompt: str,
    model: str,
    timeout_sec: float,
) -> list[dict[str, str | int | list[str] | list[int]]]:
    max_chars = int(get_settings().quiz_extract_max_chars)
    full_text = _clamp_document_text(document_text.text, max_chars)
    page_count = document_text.metadata.get("page_count", len(document_text.pages))

    def _invoke() -> ExtractedQuizQuestionBatch:
        return invoke_structured_completion(
            schema=ExtractedQuizQuestionBatch,
            model=model,
            temperature=0.0,
            timeout=timeout_sec,
            messages=[
                {"role": "system", "content": prompt},
                {
                    "role": "user",
                    "content": (
                        f"## FILE\nTên file: {filename}\n"
                        f"Số trang: {page_count}\n\n"
                        "<document_text>\n"
                        f"{full_text}\n"
                        "</document_text>\n\n"
                        "Hãy trích xuất câu hỏi theo JSON/schema đã chỉ định."
                    ),
                },
            ],
        )

    payload = llm_retry_call(label="quiz extraction", fn=_invoke)
    return [question.to_public_dict() for question in payload.questions]


def validate_quiz_extract_dependencies(settings: Settings, s3_client: object) -> None:
    """Raise ValueError when external dependencies are not configured."""
    if not s3_client or not settings.object_storage_bucket:
        raise ValueError("S3 is not configured.")
    if not settings.llm_model:
        raise ValueError("LLM configuration is missing.")
    if not resolve_llm_api_key():
        raise ValueError("LLM API key is missing.")
    try:
        build_ocr_api_config(settings)
    except DocumentTextConfigurationError as exc:
        raise ValueError("OCR configuration is missing.") from exc
