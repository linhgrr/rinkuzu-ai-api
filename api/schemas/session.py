"""
schemas/session.py — Session-related Pydantic models.
"""

from typing import Any

from pydantic import BaseModel, Field


class SessionCreateRequest(BaseModel):
    max_steps: int = Field(default=9999, ge=5, le=10000)
    use_default_data: bool = True


class SessionCreateResponse(BaseModel):
    session_id: str
    n_concepts: int
    concepts: list[dict[str, Any]]
    status: str = "active"


class SessionStatusResponse(BaseModel):
    session_id: str
    status: str
    step: int
    max_steps: int
    concepts_visited: int
    total_concepts: int
    avg_mastery: float
    coverage: float
    total_correct: int
    total_answered: int
    accuracy: float
    exercises: list[dict[str, Any]]
