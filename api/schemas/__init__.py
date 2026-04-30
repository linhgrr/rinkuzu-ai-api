"""
schemas — Re-export all Pydantic models for backward compatibility.
"""

from .exercise import (
    ExerciseOption,
    ExerciseResponse,
    NextConceptResponse,
    SubmitAnswerPayload,
    SubmitAnswerRequest,
    SubmitAnswerResponse,
    TheoryResponse,
    TutorChatMessage,
    TutorChatRequest,
    TutorChatResponse,
)
from .history import (
    SubjectHistoryDetailResponse,
    SubjectHistoryListResponse,
    SubjectHistoryResponse,
    SubjectProgressListResponse,
    SubjectProgressSummaryResponse,
)
from .knowledge import (
    ConceptDetailResponse,
    ConceptPrereq,
    KnowledgeEdgeResponse,
    KnowledgeGraphResponse,
    KnowledgeNodeResponse,
    MasteryMatrixResponse,
    MasteryRow,
)
from .quiz_tutor import (
    QuizTutorChatMessage,
    QuizTutorRequest,
    QuizTutorResponse,
    QuizTutorResponseData,
)
from .session import (
    SessionCreateRequest,
    SessionCreateResponse,
    SessionStatusResponse,
)

__all__ = [
    "ConceptDetailResponse",
    "ConceptPrereq",
    "ExerciseOption",
    "ExerciseResponse",
    "KnowledgeEdgeResponse",
    "KnowledgeGraphResponse",
    "KnowledgeNodeResponse",
    "MasteryMatrixResponse",
    "MasteryRow",
    "NextConceptResponse",
    "QuizTutorChatMessage",
    "QuizTutorRequest",
    "QuizTutorResponse",
    "QuizTutorResponseData",
    "SessionCreateRequest",
    "SessionCreateResponse",
    "SessionStatusResponse",
    "SubjectHistoryDetailResponse",
    "SubjectHistoryListResponse",
    "SubjectHistoryResponse",
    "SubjectProgressListResponse",
    "SubjectProgressSummaryResponse",
    "SubmitAnswerPayload",
    "SubmitAnswerRequest",
    "SubmitAnswerResponse",
    "TheoryResponse",
    "TutorChatMessage",
    "TutorChatRequest",
    "TutorChatResponse",
]
