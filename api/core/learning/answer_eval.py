from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .exercise_types import ExerciseType

if TYPE_CHECKING:
    from collections.abc import Callable

    from .session import ExerciseRecord


def normalize_text(value: str) -> str:
    return " ".join(value.strip().casefold().split())


def serialize_answer_for_history(exercise: ExerciseRecord, answer: dict[str, Any]) -> str | None:
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
        ordering = [
            item.strip() for item in (answer.get("ordering") or []) if item and item.strip()
        ]
        return " → ".join(ordering)
    if exercise_type == ExerciseType.MATCHING:
        matching = answer.get("matching") or {}
        return ", ".join(f"{left} -> {right}" for left, right in matching.items())
    return (answer.get("text") or "").strip()


def evaluate_answer(
    exercise: ExerciseRecord,
    answer: dict[str, Any],
    *,
    short_answer_grader: Callable[..., dict[str, bool | str | int]] | None = None,
) -> tuple[bool, str]:
    exercise_type = getattr(exercise, "exercise_type", ExerciseType.MCQ)

    if exercise_type == ExerciseType.MCQ:
        selected = (answer.get("choice") or "").strip().upper()
        return selected == exercise.correct_option.strip().upper(), selected

    if exercise_type == ExerciseType.TRUE_FALSE:
        selected_boolean = answer.get("boolean")
        expected_boolean = bool(exercise.correct_answer)
        return (
            selected_boolean is not None and bool(selected_boolean) == expected_boolean,
            "True" if selected_boolean else "False",
        )

    if exercise_type == ExerciseType.FILL_BLANK:
        user_blanks = [
            normalize_text(item) for item in (answer.get("blanks") or []) if item and item.strip()
        ]
        accepted = [
            normalize_text(item)
            for item in (exercise.correct_answer or [])
            if isinstance(item, str)
        ]
        is_correct = bool(user_blanks and accepted and user_blanks[0] in accepted)
        return is_correct, ", ".join(answer.get("blanks") or [])

    if exercise_type == ExerciseType.MULTI_CORRECT:
        selected_choices = sorted(
            {
                item.strip().upper()
                for item in (answer.get("choices") or [])
                if item and item.strip()
            }
        )
        expected_choices = sorted(
            {
                item.strip().upper()
                for item in (exercise.correct_answer or [])
                if isinstance(item, str)
            }
        )
        return selected_choices == expected_choices, ", ".join(selected_choices)

    if exercise_type == ExerciseType.ORDERING:
        selected_ordering = [
            normalize_text(item) for item in (answer.get("ordering") or []) if item and item.strip()
        ]
        expected_ordering = [
            normalize_text(item)
            for item in (exercise.correct_answer or [])
            if isinstance(item, str)
        ]
        return (
            bool(selected_ordering) and selected_ordering == expected_ordering,
            " → ".join(answer.get("ordering") or []),
        )

    if exercise_type == ExerciseType.MATCHING:
        selected_matching = {
            normalize_text(left): normalize_text(right)
            for left, right in (answer.get("matching") or {}).items()
            if left and right
        }
        expected_matching = {
            normalize_text(left): normalize_text(right)
            for left, right in (exercise.correct_answer or {}).items()
            if isinstance(left, str) and isinstance(right, str)
        }
        return bool(selected_matching) and selected_matching == expected_matching, ", ".join(
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
    exercise.explanation_correct = str(grading["explanation"])
    exercise.explanation_incorrect = str(grading["explanation"])
    return bool(grading["is_correct"]), student_answer
