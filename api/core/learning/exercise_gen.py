"""
exercise_gen.py — LLM-powered exercise generation and answer evaluation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypeVar, cast

from loguru import logger
from pydantic import BaseModel

from api.core.shared.llm import (
    _resolve_shared_llm_model,
    invoke_structured_completion,
)
from api.core.shared.retry import llm_retry_call

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

    from langchain_core.messages import BaseMessage

StructuredModelT = TypeVar("StructuredModelT", bound=BaseModel)


def _build_generation_spec(
    concept_name: str,
    concept_definition: str,
    bloom_level: int,
    exercise_type: ExerciseType,
    recent_same_concept_exercises: Sequence[dict[str, object]] | None = None,
    subject_context: str = "",
) -> Any:
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
    messages: Sequence[BaseMessage],
    temperature: float = 0.3,
) -> StructuredModelT:
    model = _resolve_shared_llm_model(None)
    return invoke_structured_completion(
        schema=schema,
        messages=messages,
        model=model,
        temperature=temperature,
    )


def generate_exercise(
    concept_name: str,
    concept_definition: str,
    bloom_level: int,
    mastery: float | None = None,
    recent_same_concept_exercises: Sequence[dict[str, object]] | None = None,
    subject_context: str = "",
) -> dict[str, str | bool | list[str] | dict[str, str] | list[dict[str, str]]] | None:
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

    result = llm_retry_call(
        label="generate_exercise",
        fn=lambda: _invoke_structured_llm(schema=schema, messages=messages),
    )
    return cast(
        "dict[str, str | bool | list[str] | dict[str, str] | list[dict[str, str]]] | None",
        serializer(result),
    )


def evaluate_short_answer(
    *,
    concept_name: str,
    question: str,
    rubric: Sequence[str],
    sample_answer: str,
    student_answer: str,
) -> dict[str, bool | str | int]:
    """Evaluate short-answer exercises against a rubric using structured LLM output."""
    logger.info("[LLM-Grade] Short answer grading for concept: {}", concept_name)
    messages = build_grading_messages(
        question=question,
        rubric=rubric,
        sample_answer=sample_answer,
        student_answer=student_answer,
    )

    result = llm_retry_call(
        label="evaluate_short_answer",
        fn=lambda: _invoke_structured_llm(schema=ShortAnswerEvaluationOutput, messages=messages),
    )
    return result.model_dump()


def generate_theory(
    concept_name: str,
    concept_definition: str,
    bloom_level: int = 2,
) -> dict[str, str | list[str]] | None:
    """Generate a concise theory summary and examples via LLM using robust json schema."""
    logger.info("[LLM-Theory] Concept: {} | Bloom: {}", concept_name, bloom_level)
    messages = build_theory_messages(
        concept_name=concept_name,
        concept_definition=concept_definition,
        bloom_level=bloom_level,
    )

    fallback: dict[str, str | list[str]] = {
        "content": f"Lý thuyết cơ bản về {concept_name}: {concept_definition}",
        "examples": ["Ví dụ 1: ...", "Ví dụ 2: ..."],
    }

    return llm_retry_call(
        label="generate_theory",
        fn=lambda: _invoke_structured_llm(
            schema=TheoryOutput,
            messages=messages,
        ).model_dump(),
        on_exhausted=lambda: fallback,
    )
