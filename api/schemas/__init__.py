"""
schemas — Re-export all Pydantic models for backward compatibility.
"""

from .session import (
    SessionCreateRequest,
    SessionCreateResponse,
    SessionStatusResponse,
)
from .exercise import (
    ExerciseOption,
    NextConceptResponse,
    TheoryResponse,
    ExerciseResponse,
    SubmitAnswerRequest,
    SubmitAnswerResponse,
)
from .knowledge import (
    KnowledgeNodeResponse,
    KnowledgeEdgeResponse,
    KnowledgeGraphResponse,
    MasteryRow,
    MasteryMatrixResponse,
    ConceptPrereq,
    ConceptDetailResponse,
)

__all__ = [
    "SessionCreateRequest",
    "SessionCreateResponse",
    "SessionStatusResponse",
    "ExerciseOption",
    "NextConceptResponse",
    "TheoryResponse",
    "ExerciseResponse",
    "SubmitAnswerRequest",
    "SubmitAnswerResponse",
    "KnowledgeNodeResponse",
    "KnowledgeEdgeResponse",
    "KnowledgeGraphResponse",
    "MasteryRow",
    "MasteryMatrixResponse",
    "ConceptPrereq",
    "ConceptDetailResponse",
]
