from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from api.core.learning.exercise_types.registry import get_handler

if TYPE_CHECKING:
    from api.core.learning.exercise_types import ExerciseBaseOutput, ExerciseType


@dataclass(frozen=True)
class ExercisePromptSpec:
    schema: type[ExerciseBaseOutput]
    instruction: str
    negative_constraints: str
    explanation_guidance: str


def get_prompt_spec(exercise_type: ExerciseType) -> ExercisePromptSpec:
    handler = get_handler(exercise_type)
    return ExercisePromptSpec(
        schema=handler.output_model,
        instruction=handler.prompt_instruction(),
        negative_constraints=handler.negative_constraints(),
        explanation_guidance=handler.explanation_guidance(),
    )
