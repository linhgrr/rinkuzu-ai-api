"""
schemas/exercise.py — Exercise-related Pydantic models.
"""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from api.core.learning.exercise_types import ExerciseType


class ExerciseOption(BaseModel):
    key: str
    value: str


class NextConceptResponse(BaseModel):
    concept_name: str
    concept_idx: int
    bloom_level: int
    bloom_label: str
    step: int
    max_steps: int


class RecommendationPrerequisite(BaseModel):
    name: str
    mastery: float


class RecommendationReason(BaseModel):
    concept_name: str
    bloom_level: int
    bloom_label: str
    satisfied_prereqs: list[RecommendationPrerequisite] = Field(default_factory=list)
    current_mastery: float
    next_milestone: float


class TheoryResponse(BaseModel):
    content: str
    examples: list[str]


class ExerciseResponse(BaseModel):
    exercise_id: str
    concept_name: str
    concept_idx: int
    bloom_level: int
    bloom_label: str
    exercise_type: ExerciseType = ExerciseType.MCQ
    question: str
    sentence: str | None = None
    options: dict[str, str] = Field(default_factory=dict)
    statement: str | None = None
    hint: str | None = None
    items: list[str] = Field(default_factory=list)
    pairs: list[dict[str, str]] = Field(default_factory=list)
    right_items: list[str] = Field(default_factory=list)

    step: int
    max_steps: int
    theory: dict[str, Any] | None = None
    recommendation_reason: RecommendationReason | None = None


class SubmitAnswerPayload(BaseModel):
    choice: str | None = None
    choices: list[str] = Field(default_factory=list)
    boolean: bool | None = None
    blanks: list[str] = Field(default_factory=list)
    ordering: list[str] = Field(default_factory=list)
    matching: dict[str, str] = Field(default_factory=dict)
    text: str | None = None


class SubmitAnswerRequest(BaseModel):
    answer: SubmitAnswerPayload


class SubmitAnswerResponse(BaseModel):
    is_correct: bool
    correct_option: str
    explanation: str
    concept_name: str
    bloom_level: int
    mastery_after: float
    avg_mastery: float
    step: int
    session_completed: bool
    stats: dict[str, Any]


class TutorChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(..., min_length=1, max_length=4000)


class TutorChatRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    user_question: str = Field(..., alias="userQuestion", min_length=1, max_length=1000)
    chat_history: list[TutorChatMessage] = Field(default_factory=list, alias="chatHistory")
    stream: bool = False


class TutorChatResponse(BaseModel):
    explanation: str
