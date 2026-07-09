"""Pydantic request/response models for the learning domain.

Covers exercises, session lifecycle, knowledge graph, and subject history.
"""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

from api.core.content_pipeline.domain.jobs import PipelineStatus
from api.schemas.enums import (
    BloomLabel,
    ConceptStatus,
    LearningSessionStatus,
    SubjectHistoryStatus,
    SubjectProgressStatus,
)
from api.schemas.pipeline import PipelineJobListItemResponse

from .exercise_types import ExerciseType
from .exercise_types.payloads import ExercisePayload, MatchingPairPayload


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


# ── Session lifecycle ───────────────────────────────────────


class SessionConceptSummary(BaseModel):
    id: str
    name: str
    index: int


class SessionExerciseSummary(BaseModel):
    exercise_id: str
    concept_name: str
    bloom_level: int
    is_correct: bool | None = None


class SessionCreateRequest(BaseModel):
    max_steps: int = Field(
        default=9999, ge=5, le=10000, description="Maximum number of exercise steps in the session."
    )
    use_default_data: bool = Field(
        default=True, description="If true, seeds the session with built-in example concepts."
    )


class SessionCreateResponse(BaseModel):
    session_id: str = Field(description="Unique session identifier.")
    n_concepts: int = Field(description="Number of concepts available in this session.")
    concepts: list[SessionConceptSummary] = Field(description="Ordered list of concept summaries.")
    status: LearningSessionStatus = Field(
        default=LearningSessionStatus.ACTIVE, description="Current learning session state."
    )


class SessionStatusResponse(BaseModel):
    session_id: str = Field(description="Session identifier.")
    status: LearningSessionStatus = Field(description="Current learning session status.")
    step: int = Field(description="Number of exercises answered so far.")
    max_steps: int = Field(description="Session step limit.")
    concepts_visited: int = Field(description="Number of distinct concepts encountered.")
    total_concepts: int = Field(description="Total concepts in the session curriculum.")
    unlocked_concepts: int = Field(
        description="Concepts whose prerequisites are currently satisfied."
    )
    locked_concepts: int = Field(description="Concepts still locked by prerequisite mastery.")
    mastered_concepts: int = Field(
        description="Unlocked concepts at or above the mastery threshold."
    )
    avg_mastery: float = Field(description="Average SAINT mastery across unlocked concepts.")
    progress_percent: int = Field(
        description="Percentage of unlocked concepts mastered; locked concepts are excluded."
    )
    coverage: float = Field(
        description="Fraction of concepts that have been visited at least once."
    )
    total_correct: int = Field(description="Cumulative count of correct answers.")
    total_answered: int = Field(description="Cumulative count of answered exercises.")
    accuracy: float = Field(description="Ratio of correct answers to total answered.")
    exercises: list[SessionExerciseSummary] = Field(description="Recent exercise history entries.")


# ── Knowledge graph & mastery ───────────────────────────────


class KnowledgeNodeResponse(BaseModel):
    id: str
    index: int
    name: str
    mastery: float
    status: ConceptStatus
    visited: bool


class KnowledgeEdgeResponse(BaseModel):
    source: str
    target: str


class KnowledgeGraphResponse(BaseModel):
    nodes: list[KnowledgeNodeResponse]
    edges: list[KnowledgeEdgeResponse]


class MasteryRow(BaseModel):
    concept_id: str
    concept_name: str
    bloom_levels: list[float]


class MasteryMatrixResponse(BaseModel):
    matrix: list[MasteryRow]
    bloom_labels: list[str]


class ConceptPrereq(BaseModel):
    id: str
    name: str
    mastery: float


class ConceptDetailResponse(BaseModel):
    id: str
    name: str
    definition: str
    mastery: float
    status: ConceptStatus
    bloom_mastery: list[float]
    prerequisites: list[ConceptPrereq]
    dependents: list[ConceptPrereq]
    visited: bool
    visit_count: int


# ── Subject history & progress ──────────────────────────────


class SubjectHistoryResponse(BaseModel):
    job_id: str
    filename: str
    subject_id: str
    status: Literal[PipelineStatus.COMPLETED]
    concepts_extracted: int
    concepts_after_merge: int
    relations_verified: int
    completed_at: float
    mastered_concept: int
    all_concept: int
    unlocked_concept: int
    locked_concept: int
    progress_percent: int


class SubjectHistoryListResponse(BaseModel):
    subjects: list[SubjectHistoryResponse]
    count: int


class PipelineJobHistoryListResponse(BaseModel):
    jobs: list[PipelineJobListItemResponse]
    count: int


class SubjectProgressSummaryResponse(BaseModel):
    job_id: str
    filename: str
    subject_id: str
    status: SubjectProgressStatus
    total_correct: int
    total_answered: int
    accuracy: float
    avg_mastery: float
    unlocked_concepts: int
    locked_concepts: int
    mastered_concepts: int
    progress_percent: int
    step: int
    max_steps: int
    created_at: float
    updated_at: float
    last_session_id: str | None


class SubjectProgressListResponse(BaseModel):
    subjects: list[SubjectProgressSummaryResponse]
    count: int


class SubjectHistoryDetailResponse(BaseModel):
    job_id: str
    filename: str
    subject_id: str
    status: SubjectHistoryStatus
    total_correct: int
    total_answered: int
    accuracy: float
    step: int
    max_steps: int
    avg_mastery: float
    unlocked_concepts: int
    locked_concepts: int
    mastered_concepts: int
    progress_percent: int
    concept_names: dict[str, str]
    concept_mastery: list[float]
    bloom_mastery: list[list[float]]
    exercise_history: list["ExerciseHistoryResponse"]
    created_at: float
    updated_at: float
    last_session_id: str | None


class DeleteSubjectResponse(BaseModel):
    job_id: str
    deleted_job: int
    deleted_sessions: int
    status: Literal["deleted"]


class ExerciseHistoryResponse(BaseModel):
    exercise_id: str
    concept_idx: int
    concept_name: str
    bloom_level: int = Field(ge=1, le=6)
    question: str
    explanation: str
    payload: ExercisePayload
    explanation_correct: str
    explanation_incorrect: str
    theory: TheoryResponse | None
    user_answer: str | None
    is_correct: bool | None
    timestamp: float
