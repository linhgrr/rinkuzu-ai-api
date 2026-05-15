"""
Knowledge router — Knowledge graph and mastery endpoints.
"""

from collections.abc import Callable
from typing import Any

from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel

from api.config import get_settings
from api.dependencies import get_current_user, get_session_manager, resolve_user_session
from api.exceptions import SessionNotFoundError
from api.rate_limit import is_admin_request, limiter
from api.schemas import ConceptDetailResponse, KnowledgeGraphResponse, MasteryMatrixResponse
from api.schemas.common import StandardResponse, ok
from api.schemas.validators import PathID

router = APIRouter(prefix="/api/session", tags=["knowledge"])


async def _get_session_resource(
    manager: Any,
    session_id: str,
    user_id: str,
    fetcher: Callable[[], dict[str, Any] | None],
    response_cls: type[BaseModel],
) -> Any:
    """Resolve session → fetch data → build response, raising 404 on miss."""
    await resolve_user_session(manager, session_id, user_id)
    data = fetcher()
    if not data:
        raise SessionNotFoundError(session_id)
    return ok(response_cls(**data).model_dump())


@router.get("/{session_id}/graph", response_model=StandardResponse[KnowledgeGraphResponse])
@limiter.limit(get_settings().rate_limit_session, exempt_when=is_admin_request)
async def get_knowledge_graph(
    request: Request,
    session_id: PathID,
    manager: Any = Depends(get_session_manager),
    user_id: str = Depends(get_current_user),
) -> Any:
    """Return the prerequisite knowledge graph for a session."""
    del request
    return await _get_session_resource(
        manager,
        session_id,
        user_id,
        fetcher=lambda: manager.get_knowledge_graph(session_id),
        response_cls=KnowledgeGraphResponse,
    )


@router.get("/{session_id}/mastery-matrix", response_model=StandardResponse[MasteryMatrixResponse])
@limiter.limit(get_settings().rate_limit_session, exempt_when=is_admin_request)
async def get_mastery_matrix(
    request: Request,
    session_id: PathID,
    manager: Any = Depends(get_session_manager),
    user_id: str = Depends(get_current_user),
) -> Any:
    """Return the concept x Bloom-level mastery matrix for a session."""
    del request
    return await _get_session_resource(
        manager,
        session_id,
        user_id,
        fetcher=lambda: manager.get_mastery_matrix(session_id),
        response_cls=MasteryMatrixResponse,
    )


@router.get(
    "/{session_id}/concept/{concept_id}", response_model=StandardResponse[ConceptDetailResponse]
)
@limiter.limit(get_settings().rate_limit_session, exempt_when=is_admin_request)
async def get_concept_detail(
    request: Request,
    session_id: PathID,
    concept_id: PathID,
    manager: Any = Depends(get_session_manager),
    user_id: str = Depends(get_current_user),
) -> Any:
    """Return detailed information about a specific concept."""
    del request
    return await _get_session_resource(
        manager,
        session_id,
        user_id,
        fetcher=lambda: manager.get_concept_detail(session_id, concept_id),
        response_cls=ConceptDetailResponse,
    )
