from __future__ import annotations

from typing import Any, Callable

from .exercise_types import ExerciseType


def normalize_text(value: str) -> str:
    return " ".join(value.strip().casefold().split())


def serialize_answer_for_history(exercise: Any, answer: dict[str, Any]) -> Any:
    exercise_type = getattr(exercise, "exercise_type", ExerciseType.MCQ)
    if exercise_type in {ExerciseType.MCQ, ExerciseType.MULTI_CORRECT}:
        choices = answer.get("choices") or []
        if choices:
            return ", ".join(sorted(choices))
        return answer.get("choice")
    if exercise_type == ExerciseType.TRUE_FALSE:
        value = answer.get("boolean")
        return None if value is None else ("True" if value else "False")
    if exercise_type == ExerciseType.FILL_BLANK:
        blanks = [item.strip() for item in (answer.get("blanks") or []) if item and item.strip()]
        return ", ".join(blanks)
    if exercise_type == ExerciseType.ORDERING:
        ordering = [item.strip() for item in (answer.get("ordering") or []) if item and item.strip()]
        return " → ".join(ordering)
    if exercise_type == ExerciseType.MATCHING:
        matching = answer.get("matching") or {}
        return ", ".join(f"{left} -> {right}" for left, right in matching.items())
    return (answer.get("text") or "").strip()


def evaluate_answer(
    exercise: Any,
    answer: dict[str, Any],
    *,
    short_answer_grader: Callable[..., dict[str, Any]] | None = None,
) -> tuple[bool, str]:
    exercise_type = getattr(exercise, "exercise_type", ExerciseType.MCQ)

    if exercise_type == ExerciseType.MCQ:
        selected = (answer.get("choice") or "").strip().upper()
        return selected == exercise.correct_option.strip().upper(), selected

    if exercise_type == ExerciseType.TRUE_FALSE:
        selected = answer.get("boolean")
        expected = bool(exercise.correct_answer)
        return selected is not None and bool(selected) == expected, "True" if selected else "False"

    if exercise_type == ExerciseType.FILL_BLANK:
        user_values = [normalize_text(item) for item in (answer.get("blanks") or []) if item and item.strip()]
        accepted = [normalize_text(item) for item in (exercise.correct_answer or []) if isinstance(item, str)]
        is_correct = bool(user_values and accepted and user_values[0] in accepted)
        return is_correct, ", ".join(answer.get("blanks") or [])

    if exercise_type == ExerciseType.MULTI_CORRECT:
        selected = sorted({item.strip().upper() for item in (answer.get("choices") or []) if item and item.strip()})
        expected = sorted({item.strip().upper() for item in (exercise.correct_answer or []) if isinstance(item, str)})
        return selected == expected, ", ".join(selected)

    if exercise_type == ExerciseType.ORDERING:
        selected = [normalize_text(item) for item in (answer.get("ordering") or []) if item and item.strip()]
        expected = [normalize_text(item) for item in (exercise.correct_answer or []) if isinstance(item, str)]
        return bool(selected) and selected == expected, " → ".join(answer.get("ordering") or [])

    if exercise_type == ExerciseType.MATCHING:
        selected = {
            normalize_text(left): normalize_text(right)
            for left, right in (answer.get("matching") or {}).items()
            if left and right
        }
        expected = {
            normalize_text(left): normalize_text(right)
            for left, right in (exercise.correct_answer or {}).items()
            if isinstance(left, str) and isinstance(right, str)
        }
        return bool(selected) and selected == expected, ", ".join(
            f"{left} -> {right}" for left, right in (answer.get("matching") or {}).items()
        )

    if short_answer_grader is None:
        raise RuntimeError("short_answer_grader is required for short_answer exercises")

    student_answer = (answer.get("text") or "").strip()
    grading = short_answer_grader(
        concept_name=exercise.concept_name,
        question=exercise.question,
        rubric=exercise.rubric,
        sample_answer=str(exercise.correct_answer or exercise.correct_option),
        student_answer=student_answer,
    )
    exercise.explanation_correct = grading["explanation"]
    exercise.explanation_incorrect = grading["explanation"]
    return bool(grading["is_correct"]), student_answer
