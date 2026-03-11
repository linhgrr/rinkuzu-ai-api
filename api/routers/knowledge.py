"""
Knowledge router — Knowledge graph and mastery endpoints.
"""

from fastapi import APIRouter, Depends

from ..schemas import (
    KnowledgeGraphResponse, MasteryMatrixResponse, ConceptDetailResponse,
)
from ..dependencies import get_session_manager, get_current_user
from ..exceptions import SessionNotFoundError

router = APIRouter(prefix="/api/session", tags=["knowledge"])


@router.get("/{session_id}/graph", response_model=KnowledgeGraphResponse)
async def get_knowledge_graph(
    session_id: str,
    manager=Depends(get_session_manager),
    user_id: str = Depends(get_current_user),
):
    session = manager.get_session(session_id)
    if not session or getattr(session, "user_id", None) != user_id:
        raise SessionNotFoundError(session_id)
    data = manager.get_knowledge_graph(session_id)
    if not data:
        raise SessionNotFoundError(session_id)
    return KnowledgeGraphResponse(**data)


@router.get("/{session_id}/mastery-matrix", response_model=MasteryMatrixResponse)
async def get_mastery_matrix(
    session_id: str,
    manager=Depends(get_session_manager),
    user_id: str = Depends(get_current_user),
):
    session = manager.get_session(session_id)
    if not session or getattr(session, "user_id", None) != user_id:
        raise SessionNotFoundError(session_id)
    data = manager.get_mastery_matrix(session_id)
    if not data:
        raise SessionNotFoundError(session_id)
    return MasteryMatrixResponse(**data)


@router.get("/{session_id}/concept/{concept_id}", response_model=ConceptDetailResponse)
async def get_concept_detail(
    session_id: str,
    concept_id: str,
    manager=Depends(get_session_manager),
    user_id: str = Depends(get_current_user),
):
    session = manager.get_session(session_id)
    if not session or getattr(session, "user_id", None) != user_id:
        raise SessionNotFoundError(session_id)
    data = manager.get_concept_detail(session_id, concept_id)
    if not data:
        raise SessionNotFoundError(session_id)
    return ConceptDetailResponse(**data)
