"""
Knowledge router — Knowledge graph and mastery endpoints.
"""

from fastapi import APIRouter, Depends

from api.dependencies import get_current_user, get_session_manager
from api.exceptions import SessionNotFoundError
from api.schemas.common import StandardResponse
from api.schemas import ConceptDetailResponse, KnowledgeGraphResponse, MasteryMatrixResponse

router = APIRouter(prefix="/api/session", tags=["knowledge"])


async def _resolve_user_session(manager, session_id: str, user_id: str):
    session = await manager.get_or_recover_session(session_id, user_id)
    if not session:
        raise SessionNotFoundError(session_id)
    return session


@router.get("/{session_id}/graph", response_model=StandardResponse[KnowledgeGraphResponse])
async def get_knowledge_graph(
    session_id: str,
    manager=Depends(get_session_manager),
    user_id: str = Depends(get_current_user),
):
    await _resolve_user_session(manager, session_id, user_id)
    data = manager.get_knowledge_graph(session_id)
    if not data:
        raise SessionNotFoundError(session_id)
    return {"success": True, "data": KnowledgeGraphResponse(**data).model_dump()}


@router.get("/{session_id}/mastery-matrix", response_model=StandardResponse[MasteryMatrixResponse])
async def get_mastery_matrix(
    session_id: str,
    manager=Depends(get_session_manager),
    user_id: str = Depends(get_current_user),
):
    await _resolve_user_session(manager, session_id, user_id)
    data = manager.get_mastery_matrix(session_id)
    if not data:
        raise SessionNotFoundError(session_id)
    return {"success": True, "data": MasteryMatrixResponse(**data).model_dump()}


@router.get("/{session_id}/concept/{concept_id}", response_model=StandardResponse[ConceptDetailResponse])
async def get_concept_detail(
    session_id: str,
    concept_id: str,
    manager=Depends(get_session_manager),
    user_id: str = Depends(get_current_user),
):
    await _resolve_user_session(manager, session_id, user_id)
    data = manager.get_concept_detail(session_id, concept_id)
    if not data:
        raise SessionNotFoundError(session_id)
    return {"success": True, "data": ConceptDetailResponse(**data).model_dump()}
