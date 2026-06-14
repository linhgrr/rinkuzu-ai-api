"""
base.py — The ExerciseTypeHandler contract every exercise type implements.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, ClassVar

if TYPE_CHECKING:
    from collections.abc import Callable

    from pydantic import BaseModel

    from api.core.learning.session import ExerciseRecord

    from .models import ExerciseBaseOutput, ExerciseType


class ExerciseTypeHandler(ABC):
    exercise_type: ClassVar[ExerciseType]
    output_model: ClassVar[type[ExerciseBaseOutput]]
    payload_model: ClassVar[type[BaseModel]]

    def __init__(self, *, short_answer_grader: Callable[..., dict] | None = None) -> None:
        self._grader = short_answer_grader

    # 1. generation config (replaces PROMPT_REGISTRY entry)
    @abstractmethod
    def prompt_instruction(self) -> str: ...
    @abstractmethod
    def negative_constraints(self) -> str: ...
    @abstractmethod
    def explanation_guidance(self) -> str: ...

    # 2. LM output model -> payload (canonical; no shuffle)
    @abstractmethod
    def payload_from_output(self, result: ExerciseBaseOutput) -> BaseModel: ...

    # 3. ExerciseRecord -> API response dict (same shape as today; shuffle from exercise_id)
    @abstractmethod
    def to_response_dict(self, exercise: ExerciseRecord) -> dict[str, Any]: ...

    # 4. grading (short_answer uses self._grader; others ignore it)
    @abstractmethod
    def evaluate(self, exercise: ExerciseRecord, answer: dict[str, Any]) -> tuple[bool, str]: ...

    # 5. tutor context for the chatbot
    @abstractmethod
    def tutor_question(self, exercise: ExerciseRecord) -> str: ...
    @abstractmethod
    def tutor_options(self, exercise: ExerciseRecord) -> list[str]: ...

    # 6. user answer -> history string
    @abstractmethod
    def serialize_answer(self, exercise: ExerciseRecord, answer: dict[str, Any]) -> str | None: ...
