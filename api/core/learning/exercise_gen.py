"""
exercise_gen.py — LLM-powered exercise generation and answer evaluation.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any, TypeVar, cast

from loguru import logger
from pydantic import BaseModel

from api.core.shared.llm import (
    _resolve_shared_llm_model,
    get_structured_llm,
    resolve_retry_policy,
    sleep_before_retry,
)

from .exercise_types import ExerciseType, ShortAnswerEvaluationOutput, select_exercise_type
from .prompts import (
    TheoryOutput,
    build_exercise_messages,
    build_grading_messages,
    build_theory_messages,
    get_prompt_spec,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

StructuredModelT = TypeVar("StructuredModelT", bound=BaseModel)


def _build_generation_spec(
    concept_name: str,
    concept_definition: str,
    bloom_level: int,
    exercise_type: ExerciseType,
    recent_same_concept_exercises: Sequence[dict[str, Any]] | None = None,
    subject_context: str = "",
):
    spec = get_prompt_spec(exercise_type)
    messages = build_exercise_messages(
        concept_name=concept_name,
        concept_definition=concept_definition,
        bloom_level=bloom_level,
        exercise_type=exercise_type,
        recent_exercises=recent_same_concept_exercises,
        subject_context=subject_context,
    )
    return spec.schema, messages, spec.serializer


def _invoke_structured_llm(
    *,
    schema: type[StructuredModelT],
    messages: list[Any],
    temperature: float = 0.3,
) -> StructuredModelT:
    model = _resolve_shared_llm_model(None)
    if not model:
        raise RuntimeError("OPENAI_MODEL (or EXERCISE_LLM_MODEL) is required.")
    structured_llm = get_structured_llm(
        schema,
        model=model,
        temperature=temperature,
        method="json_schema",
        strict=True,
    )
    result = structured_llm.invoke(messages)
    if not isinstance(result, schema):
        raise TypeError(f"LLM returned invalid structured output type: {type(result)}")
    return result


def generate_exercise(
    concept_name: str,
    concept_definition: str,
    bloom_level: int,
    mastery: float | None = None,
    recent_same_concept_exercises: Sequence[dict[str, Any]] | None = None,
    subject_context: str = "",
) -> dict[str, Any] | None:
    """Generate an exercise via LLM with a type selected from Bloom level and mastery."""
    exercise_type = select_exercise_type(bloom_level, mastery)
    logger.info(
        "[LLM-Gen] Concept: {} | Bloom: {} | Type: {} | Mastery: {}",
        concept_name,
        bloom_level,
        exercise_type,
        mastery,
    )

    schema, messages, serializer = _build_generation_spec(
        concept_name=concept_name,
        concept_definition=concept_definition,
        bloom_level=bloom_level,
        exercise_type=exercise_type,
        recent_same_concept_exercises=recent_same_concept_exercises,
        subject_context=subject_context,
    )

    t0 = time.time()
    max_retries, backoff_sec = resolve_retry_policy()
    for attempt in range(1, max_retries + 1):
        try:
            logger.debug("[LLM] ⏳ generate_exercise attempt {}/{}", attempt, max_retries)
            result = _invoke_structured_llm(
                schema=schema,
                messages=messages,
            )
        except Exception as exc:
            logger.exception(
                "[LLM] ⚠ generate_exercise attempt {}/{} failed (will_retry={})",
                attempt,
                max_retries,
                attempt < max_retries,
            )
            if attempt < max_retries:
                sleep_before_retry(attempt, backoff_sec)
            continue
        elapsed = time.time() - t0
        logger.info("[LLM] ✓ Exercise generated in {:.2f}s", elapsed)
        return cast("dict[str, Any] | None", serializer(result))

    elapsed = time.time() - t0
    logger.error("[LLM] ✗ generate_exercise failed after {:.2f}s", elapsed)
    raise RuntimeError("Exercise generation service is temporarily unavailable")


def evaluate_short_answer(
    *,
    concept_name: str,
    question: str,
    rubric: Sequence[str],
    sample_answer: str,
    student_answer: str,
) -> dict[str, Any]:
    """Evaluate short-answer exercises against a rubric using structured LLM output."""
    logger.info("[LLM-Grade] Short answer grading for concept: {}", concept_name)
    messages = build_grading_messages(
        question=question,
        rubric=rubric,
        sample_answer=sample_answer,
        student_answer=student_answer,
    )

    t0 = time.time()
    max_retries, backoff_sec = resolve_retry_policy()
    for attempt in range(1, max_retries + 1):
        try:
            logger.debug("[LLM] ⏳ evaluate_short_answer attempt {}/{}", attempt, max_retries)
            result = _invoke_structured_llm(
                schema=ShortAnswerEvaluationOutput,
                messages=messages,
            )
        except Exception as exc:
            logger.exception(
                "[LLM] ⚠ evaluate_short_answer attempt {}/{} failed (will_retry={})",
                attempt,
                max_retries,
                attempt < max_retries,
            )
            if attempt < max_retries:
                sleep_before_retry(attempt, backoff_sec)
            continue
        elapsed = time.time() - t0
        logger.info("[LLM] ✓ Short answer graded in {:.2f}s", elapsed)
        return result.model_dump()

    elapsed = time.time() - t0
    logger.error("[LLM] ✗ evaluate_short_answer failed after {:.2f}s", elapsed)
    raise RuntimeError("Short-answer grading service is temporarily unavailable")


def generate_theory(
    concept_name: str,
    concept_definition: str,
    bloom_level: int = 2,
) -> dict[str, Any] | None:
    """Generate a concise theory summary and examples via LLM using robust json schema."""
    logger.info("[LLM-Theory] Concept: {} | Bloom: {}", concept_name, bloom_level)
    messages = build_theory_messages(
        concept_name=concept_name,
        concept_definition=concept_definition,
        bloom_level=bloom_level,
    )

    t0 = time.time()
    max_retries, backoff_sec = resolve_retry_policy()
    fallback = {
        "content": f"Lý thuyết cơ bản về {concept_name}: {concept_definition}",
        "examples": ["Ví dụ 1: ...", "Ví dụ 2: ..."],
    }

    for attempt in range(1, max_retries + 1):
        try:
            logger.debug("[LLM] ⏳ generate_theory attempt {}/{}", attempt, max_retries)
            result = _invoke_structured_llm(
                schema=TheoryOutput,
                messages=messages,
            )
        except Exception as exc:
            logger.exception(
                "[LLM] ⚠ generate_theory attempt {}/{} failed (will_retry={})",
                attempt,
                max_retries,
                attempt < max_retries,
            )
            if attempt < max_retries:
                sleep_before_retry(attempt, backoff_sec)
            continue
        elapsed = time.time() - t0
        logger.info("[LLM] ✓ Theory generated in {:.2f}s", elapsed)
        return result.model_dump()

    elapsed = time.time() - t0
    logger.error("[LLM] ✗ generate_theory failed after {:.2f}s", elapsed)
    return fallback
