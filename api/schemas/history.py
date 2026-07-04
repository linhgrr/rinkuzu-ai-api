"""History-related Pydantic models."""

from typing import Literal

from pydantic import BaseModel, Field

from api.core.content_pipeline.domain.jobs import PipelineStatus
from api.core.learning.exercise_types.payloads import ExercisePayload

from .enums import SubjectHistoryStatus, SubjectProgressStatus
from .exercise import TheoryResponse
from .pipeline import PipelineJobListItemResponse


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
