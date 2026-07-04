"""
schemas/exercise.py — Exercise-related Pydantic models.
"""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

from api.core.learning.exercise_types import ExerciseType
from api.core.learning.exercise_types.payloads import MatchingPairPayload

from .enums import BloomLabel


class ExerciseOption(BaseModel):
    key: str
    value: str


class NextConceptResponse(BaseModel):
    concept_name: str
    concept_idx: int
    bloom_level: int
    bloom_label: BloomLabel
    step: int
    max_steps: int


class RecommendationPrerequisite(BaseModel):
    name: str
    mastery: float


class RecommendationReason(BaseModel):
    concept_name: str
    bloom_level: int
    bloom_label: BloomLabel
    satisfied_prereqs: list[RecommendationPrerequisite] = Field(default_factory=list)
    current_mastery: float
    next_milestone: float


class TheoryResponse(BaseModel):
    content: str
    examples: list[str]


class ExerciseResponseBase(BaseModel):
    exercise_id: str
    concept_name: str
    concept_idx: int
    bloom_level: int
    bloom_label: BloomLabel
    question: str
    step: int
    max_steps: int
    theory: TheoryResponse | None = None
    recommendation_reason: RecommendationReason | None = None


class MCQExerciseResponse(ExerciseResponseBase):
    exercise_type: Literal[ExerciseType.MCQ] = ExerciseType.MCQ
    options: dict[str, str]


class TrueFalseExerciseResponse(ExerciseResponseBase):
    exercise_type: Literal[ExerciseType.TRUE_FALSE] = ExerciseType.TRUE_FALSE
    statement: str


class FillBlankExerciseResponse(ExerciseResponseBase):
    exercise_type: Literal[ExerciseType.FILL_BLANK] = ExerciseType.FILL_BLANK
    sentence: str
    hint: str


class MultiCorrectExerciseResponse(ExerciseResponseBase):
    exercise_type: Literal[ExerciseType.MULTI_CORRECT] = ExerciseType.MULTI_CORRECT
    options: dict[str, str]


class OrderingExerciseResponse(ExerciseResponseBase):
    exercise_type: Literal[ExerciseType.ORDERING] = ExerciseType.ORDERING
    items: list[str]


class MatchingExerciseResponse(ExerciseResponseBase):
    exercise_type: Literal[ExerciseType.MATCHING] = ExerciseType.MATCHING
    pairs: list[MatchingPairPayload]
    right_items: list[str]


class ShortAnswerExerciseResponse(ExerciseResponseBase):
    exercise_type: Literal[ExerciseType.SHORT_ANSWER] = ExerciseType.SHORT_ANSWER
    rubric: list[str]


ExerciseResponse = Annotated[
    MCQExerciseResponse
    | TrueFalseExerciseResponse
    | FillBlankExerciseResponse
    | MultiCorrectExerciseResponse
    | OrderingExerciseResponse
    | MatchingExerciseResponse
    | ShortAnswerExerciseResponse,
    Field(discriminator="exercise_type"),
]


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


class SubmitAnswerStats(BaseModel):
    total_correct: int
    total_answered: int
    accuracy: float


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
    stats: SubmitAnswerStats


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
