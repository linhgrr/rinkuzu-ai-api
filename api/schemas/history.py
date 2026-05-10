"""
schemas/history.py — History-related Pydantic models.
"""

from pydantic import BaseModel


class SubjectHistoryResponse(BaseModel):
    job_id: str
    filename: str
    subject_id: str
    status: str
    concepts_extracted: int = 0
    concepts_after_merge: int = 0
    relations_verified: int = 0
    completed_at: float
    mastered_concept: int = 0
    all_concept: int = 0
    progress_percent: int = 0


class SubjectHistoryListResponse(BaseModel):
    subjects: list[SubjectHistoryResponse]
    count: int


class PipelineJobHistoryListResponse(BaseModel):
    jobs: list[SubjectHistoryResponse]
    count: int


class SubjectProgressSummaryResponse(BaseModel):
    job_id: str
    filename: str = ""
    subject_id: str = ""
    status: str
    total_correct: int = 0
    total_answered: int = 0
    accuracy: float = 0.0
    avg_mastery: float = 0.0
    step: int = 0
    max_steps: int = 0
    created_at: float = 0
    updated_at: float = 0
    last_session_id: str | None = None


class SubjectProgressListResponse(BaseModel):
    subjects: list[SubjectProgressSummaryResponse]
    count: int


class SubjectHistoryDetailResponse(BaseModel):
    job_id: str
    filename: str = ""
    subject_id: str = ""
    status: str
    total_correct: int = 0
    total_answered: int = 0
    accuracy: float = 0.0
    step: int = 0
    max_steps: int = 0
    avg_mastery: float = 0.0
    concept_names: dict[str, str] = {}
    concept_mastery: list[float] = []
    bloom_mastery: list[list[float]] = []
    exercise_history: list[dict[str, object]] = []
    created_at: float = 0
    updated_at: float = 0
    last_session_id: str | None = None


class DeleteSubjectResponse(BaseModel):
    job_id: str
    deleted_job: int
    deleted_sessions: int
    status: str
