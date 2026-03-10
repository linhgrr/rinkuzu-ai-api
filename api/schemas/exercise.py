"""
schemas/exercise.py — Exercise-related Pydantic models.
"""

from typing import Optional, Dict, Any, List

from pydantic import BaseModel, Field


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


class TheoryResponse(BaseModel):
    content: str
    examples: List[str]


class ExerciseResponse(BaseModel):
    exercise_id: str
    concept_name: str
    concept_idx: int
    bloom_level: int
    bloom_label: str
    question: str
    options: Dict[str, str]
    step: int
    max_steps: int
    theory: Optional[Dict[str, Any]] = None


class SubmitAnswerRequest(BaseModel):
    answer: str = Field(..., min_length=1, max_length=10)


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
    stats: Dict[str, Any]
