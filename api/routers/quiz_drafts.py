"""FastAPI-owned quiz draft processing endpoints."""

from typing import Annotated

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query

from api.core.quiz.draft_service import (
    QuizDraftDependencyError,
    QuizDraftNotFoundError,
    QuizDraftService,
    QuizDraftValidationError,
    public_draft,
)
from api.dependencies import get_current_user
from api.schemas.common import StandardResponse
from api.schemas.quiz_draft import (
    QuizDraftCreateRequest,
    QuizDraftPatchRequest,
    QuizDraftSubmitRequest,
)

router = APIRouter(prefix="/api/quiz/drafts", tags=["quiz-drafts"])


def _service_error_to_http(exc: Exception) -> HTTPException:
    if isinstance(exc, QuizDraftNotFoundError):
        return HTTPException(status_code=404, detail=str(exc))
    if isinstance(exc, QuizDraftValidationError):
        return HTTPException(status_code=400, detail=str(exc))
    if isinstance(exc, QuizDraftDependencyError):
        return HTTPException(status_code=503, detail=str(exc))
    return HTTPException(status_code=500, detail="Quiz draft operation failed.")


@router.post("", response_model=StandardResponse[dict])
async def create_quiz_draft(
    req: QuizDraftCreateRequest,
    background_tasks: BackgroundTasks,
    user_id: Annotated[str, Depends(get_current_user)],
):
    service = QuizDraftService()
    try:
        draft = await service.create_draft(req, user_id)
    except Exception as exc:
        raise _service_error_to_http(exc) from exc

    background_tasks.add_task(service.process_draft, draft["draft_id"], user_id)
    return {"success": True, "data": {"draft": public_draft(draft)}}


@router.get("", response_model=StandardResponse[dict])
async def list_quiz_drafts(
    user_id: Annotated[str, Depends(get_current_user)],
    limit: Annotated[int, Query(ge=1, le=50)] = 20,
):
    service = QuizDraftService()
    drafts = await service.list_drafts(user_id, limit)
    return {"success": True, "data": {"drafts": [public_draft(draft) for draft in drafts]}}


@router.get("/{draft_id}", response_model=StandardResponse[dict])
async def get_quiz_draft(draft_id: str, user_id: Annotated[str, Depends(get_current_user)]):
    service = QuizDraftService()
    try:
        draft = await service.get_draft(draft_id, user_id)
    except Exception as exc:
        raise _service_error_to_http(exc) from exc
    return {"success": True, "data": {"draft": public_draft(draft)}}


@router.patch("/{draft_id}", response_model=StandardResponse[dict])
async def patch_quiz_draft(
    draft_id: str,
    req: QuizDraftPatchRequest,
    user_id: Annotated[str, Depends(get_current_user)],
):
    service = QuizDraftService()
    try:
        draft = await service.patch_draft(draft_id, user_id, req)
    except Exception as exc:
        raise _service_error_to_http(exc) from exc
    return {"success": True, "data": {"draft": public_draft(draft)}}


@router.delete("/{draft_id}", response_model=StandardResponse[dict])
async def delete_quiz_draft(draft_id: str, user_id: Annotated[str, Depends(get_current_user)]):
    service = QuizDraftService()
    try:
        draft = await service.delete_draft(draft_id, user_id)
    except Exception as exc:
        raise _service_error_to_http(exc) from exc
    return {"success": True, "data": {"draft": public_draft(draft)}}


@router.post("/{draft_id}/submit", response_model=StandardResponse[dict])
async def submit_quiz_draft(
    draft_id: str,
    req: QuizDraftSubmitRequest,
    user_id: Annotated[str, Depends(get_current_user)],
):
    service = QuizDraftService()
    try:
        draft = await service.mark_submitted(draft_id, user_id, req.quiz_id)
    except Exception as exc:
        raise _service_error_to_http(exc) from exc
    return {"success": True, "data": {"draft": public_draft(draft)}}
