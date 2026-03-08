"""
Knowledge router — Knowledge graph and mastery endpoints
"""

from fastapi import APIRouter, HTTPException

from ..schemas import (
    KnowledgeGraphResponse, MasteryMatrixResponse, ConceptDetailResponse,
)

router = APIRouter(prefix="/api/session", tags=["knowledge"])

session_manager = None


@router.get("/{session_id}/graph", response_model=KnowledgeGraphResponse)
async def get_knowledge_graph(session_id: str):
    if session_manager is None:
        raise HTTPException(500, "Server not initialized")

    data = session_manager.get_knowledge_graph(session_id)
    if not data:
        raise HTTPException(404, "Session not found")

    return KnowledgeGraphResponse(**data)


@router.get("/{session_id}/mastery-matrix", response_model=MasteryMatrixResponse)
async def get_mastery_matrix(session_id: str):
    if session_manager is None:
        raise HTTPException(500, "Server not initialized")

    data = session_manager.get_mastery_matrix(session_id)
    if not data:
        raise HTTPException(404, "Session not found")

    return MasteryMatrixResponse(**data)


@router.get("/{session_id}/concept/{concept_id}", response_model=ConceptDetailResponse)
async def get_concept_detail(session_id: str, concept_id: str):
    if session_manager is None:
        raise HTTPException(500, "Server not initialized")

    data = session_manager.get_concept_detail(session_id, concept_id)
    if not data:
        raise HTTPException(404, "Session or concept not found")

    return ConceptDetailResponse(**data)
