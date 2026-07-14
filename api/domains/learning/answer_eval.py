from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .exercise_types.selection import normalize_text as normalize_text  # noqa: PLC0414  # re-export

if TYPE_CHECKING:
    from .session import ExerciseRecord


def serialize_answer_for_history(exercise: ExerciseRecord, answer: dict[str, Any]) -> str | None:
    from .exercise_types.registry import get_handler

    return get_handler(exercise.payload.exercise_type).serialize_answer(exercise, answer)


def evaluate_answer(exercise: ExerciseRecord, answer: dict[str, Any]) -> tuple[bool, str]:
    """Evaluate CPU-only exercise types. Short-answer is LLM-graded in the service layer."""
    from .exercise_types.registry import get_handler

    return get_handler(exercise.payload.exercise_type).evaluate(exercise, answer)
