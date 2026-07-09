"""
registry.py — Maps each ExerciseType to its handler class.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .base import ExerciseTypeHandler
    from .models import ExerciseType

_HANDLER_CLASSES: dict[ExerciseType, type[ExerciseTypeHandler]] = {}


def register(cls: type[ExerciseTypeHandler]) -> type[ExerciseTypeHandler]:
    _HANDLER_CLASSES[cls.exercise_type] = cls
    return cls


def get_handler(exercise_type: ExerciseType) -> ExerciseTypeHandler:
    return _HANDLER_CLASSES[exercise_type]()
