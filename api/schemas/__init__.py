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
    TutorChatMessage,
    TutorChatRequest,
    TutorChatResponse,
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
from .history import (
    SubjectHistoryResponse,
    SubjectHistoryListResponse,
    SubjectProgressSummaryResponse,
    SubjectProgressListResponse,
    SubjectHistoryDetailResponse,
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
    "TutorChatMessage",
    "TutorChatRequest",
    "TutorChatResponse",
    "KnowledgeNodeResponse",
    "KnowledgeEdgeResponse",
    "KnowledgeGraphResponse",
    "MasteryRow",
    "MasteryMatrixResponse",
    "ConceptPrereq",
    "ConceptDetailResponse",
    "SubjectHistoryResponse",
    "SubjectHistoryListResponse",
    "SubjectProgressSummaryResponse",
    "SubjectProgressListResponse",
    "SubjectHistoryDetailResponse",
]
