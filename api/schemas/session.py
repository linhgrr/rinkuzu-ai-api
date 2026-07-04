"""
schemas/session.py — Session-related Pydantic models.
"""

from pydantic import BaseModel, Field

from .enums import LearningSessionStatus


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
    avg_mastery: float = Field(description="Average BKT mastery across all concepts.")
    coverage: float = Field(
        description="Fraction of concepts that have been visited at least once."
    )
    total_correct: int = Field(description="Cumulative count of correct answers.")
    total_answered: int = Field(description="Cumulative count of answered exercises.")
    accuracy: float = Field(description="Ratio of correct answers to total answered.")
    exercises: list[SessionExerciseSummary] = Field(description="Recent exercise history entries.")
