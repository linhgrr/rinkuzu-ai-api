"""
registry.py — Maps each ExerciseType to its handler class.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

    from .base import ExerciseTypeHandler
    from .models import ExerciseType

_HANDLER_CLASSES: dict[ExerciseType, type[ExerciseTypeHandler]] = {}


def register(cls: type[ExerciseTypeHandler]) -> type[ExerciseTypeHandler]:
    _HANDLER_CLASSES[cls.exercise_type] = cls
    return cls


def get_handler(
    exercise_type: ExerciseType,
    *,
    short_answer_grader: Callable[..., dict] | None = None,
) -> ExerciseTypeHandler:
    return _HANDLER_CLASSES[exercise_type](short_answer_grader=short_answer_grader)
