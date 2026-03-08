"""
schemas.py — Pydantic request/response models
"""

from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field


# --- Session ---

class SessionCreateRequest(BaseModel):
    max_steps: int = Field(default=50, ge=5, le=200)
    use_default_data: bool = True


class SessionCreateResponse(BaseModel):
    session_id: str
    n_concepts: int
    concepts: List[Dict[str, Any]]
    status: str = "active"


# --- Exercise ---

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


# --- Session Status ---

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
    exercises: List[Dict[str, Any]]


# --- Knowledge Graph ---

class KnowledgeNodeResponse(BaseModel):
    id: str
    index: int
    name: str
    mastery: float
    status: str
    visited: bool


class KnowledgeEdgeResponse(BaseModel):
    source: str
    target: str


class KnowledgeGraphResponse(BaseModel):
    nodes: List[KnowledgeNodeResponse]
    edges: List[KnowledgeEdgeResponse]


# --- Mastery Matrix ---

class MasteryRow(BaseModel):
    concept_id: str
    concept_name: str
    bloom_levels: List[float]


class MasteryMatrixResponse(BaseModel):
    matrix: List[MasteryRow]
    bloom_labels: List[str]


# --- Concept Detail ---

class ConceptPrereq(BaseModel):
    id: str
    name: str
    mastery: float


class ConceptDetailResponse(BaseModel):
    id: str
    name: str
    definition: str
    mastery: float
    status: str
    bloom_mastery: List[float]
    prerequisites: List[ConceptPrereq]
    dependents: List[ConceptPrereq]
    visited: bool
    visit_count: int
