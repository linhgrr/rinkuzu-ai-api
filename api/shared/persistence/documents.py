from __future__ import annotations

from datetime import datetime  # noqa: TC003
from enum import StrEnum
from typing import Any, ClassVar, Literal

from beanie import Document, Insert, Replace, SaveChanges, before_event
from pydantic import BaseModel, ConfigDict, Field
from pymongo import ASCENDING, DESCENDING, IndexModel

from api.domains.content_pipeline.domain.jobs import PipelineStatus

from .common import utc_now


class QuizDraftStatus(StrEnum):
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    SUBMITTED = "submitted"
    EXPIRED = "expired"


class QuizQuestion(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    question: str = Field(min_length=1)
    type: Literal["single", "multiple"]
    options: list[str] = Field(min_length=4, max_length=5)
    correct_index: int | None = Field(default=None, alias="correctIndex")
    correct_indexes: list[int] = Field(default_factory=list, alias="correctIndexes")


class QuizDraftPdf(BaseModel):
    s3_key: str | None = None
    file_name: str | None = None
    file_size: int | None = None
    page_count: int | None = None


class QuizDraftProgressPayload(BaseModel):
    processed: int = 0
    total: int = 0
    percent: int = 0


class ConceptMasteryEntry(BaseModel):
    concept_idx: int
    mastery: float


class BloomMasteryEntry(BaseModel):
    concept_idx: int
    levels: list[float] = Field(min_length=6, max_length=6)


class ExerciseEntry(BaseModel):
    exercise_id: str
    concept_idx: int
    concept_name: str
    bloom_level: int
    question: str
    explanation: str
    payload: dict[str, Any] = Field(default_factory=dict)
    explanation_correct: str = ""
    explanation_incorrect: str = ""
    theory: dict[str, Any] | None = None
    user_answer: str | None = None
    is_correct: bool | None = None
    timestamp: datetime


class PipelineJobDocument(Document):
    job_id: str
    filename: str
    subject_id: str
    user_id: str | None = None
    status: PipelineStatus = PipelineStatus.PENDING
    current_step: str = ""
    progress: float = 0.0
    total_chunks: int = 0
    total_pages: int = 0
    page_batch_size: int = 10
    batch_count: int = 0
    failed_batch_count: int = 0
    partial_success: bool = False
    concepts_extracted: int = 0
    concepts_after_merge: int = 0
    relations_verified: int = 0
    graph_stats: dict[str, Any] = Field(default_factory=dict)
    quality_report: dict[str, Any] | None = None
    debug_trace: list[dict[str, Any]] = Field(default_factory=list)
    result: dict[str, Any] | None = None
    partial_graph: dict[str, Any] | None = None
    error_message: str | None = None
    error_code: str | None = None
    user_message: str | None = None
    retryable: bool = False
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)
    heartbeat_at: datetime = Field(default_factory=utc_now)
    completed_at: datetime | None = None
    source_s3_key: str | None = None
    prs_threshold: float | None = None
    min_confidence: float = 0.6
    apply_reduction: bool = True
    retry_count: int = 0
    cancel_requested: bool = False
    eta_seconds: float | None = None

    class Settings:
        name = "al_pipeline_jobs"
        indexes: ClassVar[list[IndexModel]] = [
            IndexModel([("job_id", ASCENDING)], unique=True),
            IndexModel(
                [("user_id", ASCENDING), ("status", ASCENDING), ("completed_at", DESCENDING)]
            ),
        ]


class PipelineJobListProjection(BaseModel):
    job_id: str
    filename: str
    subject_id: str
    status: PipelineStatus
    page_batch_size: int = 10
    batch_count: int = 0
    failed_batch_count: int = 0
    partial_success: bool = False
    concepts_extracted: int = 0
    concepts_after_merge: int = 0
    relations_verified: int = 0
    completed_at: datetime | None = None


class PipelineJobLookupProjection(BaseModel):
    job_id: str
    filename: str = ""
    subject_id: str = ""


class PipelineJobCancelProjection(BaseModel):
    cancel_requested: bool = False


class PipelineJobActiveProjection(BaseModel):
    job_id: str
    filename: str
    subject_id: str
    status: PipelineStatus
    current_step: str = ""
    progress: float = 0.0
    page_batch_size: int = 10
    batch_count: int = 0
    failed_batch_count: int = 0
    partial_success: bool = False
    concepts_extracted: int = 0
    concepts_after_merge: int = 0
    relations_verified: int = 0
    quality_report: dict[str, Any] | None = None
    error_code: str | None = None
    user_message: str | None = None
    retryable: bool = False
    retry_count: int = 0
    eta_seconds: float | None = None
    source_s3_key: str | None = None
    prs_threshold: float | None = None
    min_confidence: float = 0.6
    apply_reduction: bool = True
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)
    heartbeat_at: datetime = Field(default_factory=utc_now)
    completed_at: datetime | None = None


class SubjectProgressDocument(Document):
    job_id: str
    user_id: str
    last_session_id: str | None = None
    status: str = "active"
    total_correct: int = 0
    total_answered: int = 0
    accuracy: float = 0.0
    step: int = 0
    max_steps: int = 9999
    avg_mastery: float = 0.0
    unlocked_concepts: int = 0
    locked_concepts: int = 0
    mastered_concepts: int = 0
    progress_percent: int = 0
    concept_names: dict[str, str] = Field(default_factory=dict)
    concept_indices: dict[str, int] = Field(default_factory=dict)
    concept_mastery: dict[str, ConceptMasteryEntry] = Field(default_factory=dict)
    bloom_mastery: dict[str, BloomMasteryEntry] = Field(default_factory=dict)
    exercise_history: list[ExerciseEntry] = Field(default_factory=list)
    current_exercise: ExerciseEntry | None = None
    pending_concept_idx: int | None = None
    pending_bloom_level: int | None = None
    pending_action: int | None = None
    recommendation_reason: dict[str, Any] | None = None
    submission_receipts: dict[str, dict[str, Any]] = Field(default_factory=dict)
    version: int = 0
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)

    @before_event([Insert, Replace, SaveChanges])
    def touch_updated_at(self) -> None:
        self.updated_at = utc_now()

    class Settings:
        name = "al_subject_progress"
        indexes: ClassVar[list[IndexModel]] = [
            IndexModel([("user_id", ASCENDING), ("job_id", ASCENDING)], unique=True),
            IndexModel([("user_id", ASCENDING), ("last_session_id", ASCENDING)]),
            IndexModel([("user_id", ASCENDING), ("updated_at", DESCENDING)]),
        ]


class SubjectProgressSummaryProjection(BaseModel):
    job_id: str
    last_session_id: str | None = None
    status: str = "active"
    total_correct: int = 0
    total_answered: int = 0
    accuracy: float = 0.0
    avg_mastery: float = 0.0
    unlocked_concepts: int = 0
    locked_concepts: int = 0
    mastered_concepts: int = 0
    progress_percent: int = 0
    step: int = 0
    max_steps: int = 9999
    created_at: datetime
    updated_at: datetime


class QuizDraftDocument(Document):
    draft_id: str
    user_id: str
    title: str
    description: str = ""
    category_id: str | None = None
    prompt: str | None = None
    pdf: QuizDraftPdf = Field(default_factory=QuizDraftPdf)
    status: QuizDraftStatus = QuizDraftStatus.QUEUED
    progress: QuizDraftProgressPayload = Field(default_factory=QuizDraftProgressPayload)
    questions: list[QuizQuestion] = Field(default_factory=list)
    error: str | None = None
    submitted_quiz_id: str | None = None
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)
    expires_at: datetime = Field(default_factory=utc_now)

    @before_event([Insert, Replace, SaveChanges])
    def touch_updated_at(self) -> None:
        self.updated_at = utc_now()

    class Settings:
        name = "quiz_drafts"
        indexes: ClassVar[list[IndexModel]] = [
            IndexModel([("draft_id", ASCENDING)], unique=True),
            IndexModel([("user_id", ASCENDING), ("status", ASCENDING), ("created_at", DESCENDING)]),
            IndexModel("expires_at", expireAfterSeconds=0),
        ]


class QuizDraftListProjection(BaseModel):
    draft_id: str
    user_id: str
    title: str
    description: str = ""
    category_id: str | None = None
    prompt: str | None = None
    pdf: QuizDraftPdf = Field(default_factory=QuizDraftPdf)
    status: QuizDraftStatus
    progress: QuizDraftProgressPayload = Field(default_factory=QuizDraftProgressPayload)
    error: str | None = None
    submitted_quiz_id: str | None = None
    created_at: datetime
    updated_at: datetime
    expires_at: datetime


class DocumentOCRPage(BaseModel):
    page_number: int
    text: str


class DocumentOCRRecordDocument(Document):
    file_hash: str
    file_name: str
    file_size_bytes: int | None = None
    text: str
    page_count: int = 0
    provider: str | None = None
    model: str | None = None
    pages: list[DocumentOCRPage] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)

    @before_event([Insert, Replace, SaveChanges])
    def touch_updated_at(self) -> None:
        self.updated_at = utc_now()

    class Settings:
        name = "al_document_ocr_records"
        indexes: ClassVar[list[IndexModel]] = [
            IndexModel([("file_hash", ASCENDING)], unique=True),
            IndexModel([("updated_at", DESCENDING)]),
        ]


class DocumentChunkDocument(Document):
    job_id: str
    subject_id: str
    chunk_index: int
    text: str
    start_page: int = 0
    end_page: int = 0
    created_at: datetime = Field(default_factory=utc_now)

    class Settings:
        name = "al_document_chunks"
        indexes: ClassVar[list[IndexModel]] = [
            IndexModel([("job_id", ASCENDING), ("chunk_index", ASCENDING)], unique=True),
            IndexModel([("subject_id", ASCENDING), ("job_id", ASCENDING)]),
        ]


class LlmUsageDocument(Document):
    user_id: str | None = None
    action: str | None = None
    model: str
    provider: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    cost_usd: float = 0.0
    created_at: datetime = Field(default_factory=utc_now)

    class Settings:
        name = "llm_usage"
        indexes: ClassVar[list[IndexModel]] = [
            IndexModel([("user_id", ASCENDING), ("created_at", DESCENDING)]),
            IndexModel([("created_at", DESCENDING)]),
            IndexModel([("model", ASCENDING), ("created_at", DESCENDING)]),
        ]
