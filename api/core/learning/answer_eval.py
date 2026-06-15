from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable

    from .session import ExerciseRecord


def normalize_text(value: str) -> str:
    return " ".join(value.strip().casefold().split())


def serialize_answer_for_history(exercise: ExerciseRecord, answer: dict[str, Any]) -> str | None:
    from .exercise_types.registry import get_handler

    payload = exercise.payload
    if payload is None:
        raise ValueError("ExerciseRecord.payload is required to serialize answers")
    return get_handler(payload.exercise_type).serialize_answer(exercise, answer)


def evaluate_answer(
    exercise: ExerciseRecord,
    answer: dict[str, Any],
    *,
    short_answer_grader: Callable[..., dict[str, bool | str | int]] | None = None,
) -> tuple[bool, str]:
    from .exercise_types.registry import get_handler

    payload = exercise.payload
    if payload is None:
        raise ValueError("ExerciseRecord.payload is required to evaluate answers")
    handler = get_handler(payload.exercise_type, short_answer_grader=short_answer_grader)
    return handler.evaluate(exercise, answer)
