"""
payloads.py — Typed per-exercise content, stored canonical-only and persisted nested.
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, Field

from .models import ExerciseType


class MCQPayload(BaseModel):
    exercise_type: Literal[ExerciseType.MCQ] = ExerciseType.MCQ
    options: dict[str, str]
    correct_option: str


class TrueFalsePayload(BaseModel):
    exercise_type: Literal[ExerciseType.TRUE_FALSE] = ExerciseType.TRUE_FALSE
    statement: str
    correct_answer: bool


class FillBlankPayload(BaseModel):
    exercise_type: Literal[ExerciseType.FILL_BLANK] = ExerciseType.FILL_BLANK
    sentence: str
    hint: str
    blank_answers: list[str]


class MultiCorrectPayload(BaseModel):
    exercise_type: Literal[ExerciseType.MULTI_CORRECT] = ExerciseType.MULTI_CORRECT
    options: dict[str, str]
    correct_options: list[str]


class OrderingPayload(BaseModel):
    exercise_type: Literal[ExerciseType.ORDERING] = ExerciseType.ORDERING
    correct_order: list[str]


class MatchingPairPayload(BaseModel):
    left: str
    right: str


class MatchingPayload(BaseModel):
    exercise_type: Literal[ExerciseType.MATCHING] = ExerciseType.MATCHING
    pairs: list[MatchingPairPayload]


class ShortAnswerPayload(BaseModel):
    exercise_type: Literal[ExerciseType.SHORT_ANSWER] = ExerciseType.SHORT_ANSWER
    rubric: list[str]
    sample_answer: str


ExercisePayload = Annotated[
    MCQPayload
    | TrueFalsePayload
    | FillBlankPayload
    | MultiCorrectPayload
    | OrderingPayload
    | MatchingPayload
    | ShortAnswerPayload,
    Field(discriminator="exercise_type"),
]
