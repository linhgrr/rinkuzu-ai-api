"""FastAPI-owned quiz draft processing endpoints."""

from typing import Annotated, Any

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query, Request

from api.config import get_settings
from api.core.quiz.draft_service import (
    QuizDraftDependencyError,
    QuizDraftNotFoundError,
    QuizDraftService,
    QuizDraftValidationError,
    public_draft,
)
from api.dependencies import get_current_user
from api.rate_limit import is_admin_request, limiter
from api.schemas.common import StandardResponse, ok
from api.schemas.quiz_draft import (
    QuizDraftCreateRequest,
    QuizDraftListResponse,
    QuizDraftPatchRequest,
    QuizDraftSingleResponse,
    QuizDraftSubmitRequest,
)
from api.schemas.validators import PathID

router = APIRouter(prefix="/api/quiz/drafts", tags=["quiz-drafts"])


def _service_error_to_http(exc: Exception) -> HTTPException:
    if isinstance(exc, QuizDraftNotFoundError):
        return HTTPException(status_code=404, detail=str(exc))
    if isinstance(exc, QuizDraftValidationError):
        return HTTPException(status_code=400, detail=str(exc))
    if isinstance(exc, QuizDraftDependencyError):
        return HTTPException(status_code=503, detail=str(exc))
    return HTTPException(status_code=500, detail="Quiz draft operation failed.")


@router.post("", response_model=StandardResponse[QuizDraftSingleResponse])
@limiter.limit(get_settings().rate_limit_quiz_drafts, exempt_when=is_admin_request)
async def create_quiz_draft(
    request: Request,
    req: QuizDraftCreateRequest,
    background_tasks: BackgroundTasks,
    user_id: Annotated[str, Depends(get_current_user)],
) -> Any:
    """Create a new quiz draft and enqueue AI processing."""
    del request
    service = QuizDraftService()
    try:
        draft = await service.create_draft(req, user_id)
    except Exception as exc:
        raise _service_error_to_http(exc) from exc

    background_tasks.add_task(service.process_draft, draft["draft_id"], user_id)
    return ok({"draft": public_draft(draft)})


@router.get("", response_model=StandardResponse[QuizDraftListResponse])
@limiter.limit(get_settings().rate_limit_quiz_drafts, exempt_when=is_admin_request)
async def list_quiz_drafts(
    request: Request,
    user_id: Annotated[str, Depends(get_current_user)],
    limit: Annotated[int, Query(ge=1, le=50)] = 20,
) -> Any:
    """List recent quiz drafts for the authenticated user."""
    del request
    service = QuizDraftService()
    drafts = await service.list_drafts(user_id, limit)
    return ok({"drafts": [public_draft(draft) for draft in drafts]})


@router.get("/{draft_id}", response_model=StandardResponse[QuizDraftSingleResponse])
@limiter.limit(get_settings().rate_limit_quiz_drafts, exempt_when=is_admin_request)
async def get_quiz_draft(
    request: Request,
    draft_id: PathID,
    user_id: Annotated[str, Depends(get_current_user)],
) -> Any:
    """Retrieve a single quiz draft by ID."""
    del request
    service = QuizDraftService()
    try:
        draft = await service.get_draft(draft_id, user_id)
    except Exception as exc:
        raise _service_error_to_http(exc) from exc
    return ok({"draft": public_draft(draft)})


@router.patch("/{draft_id}", response_model=StandardResponse[QuizDraftSingleResponse])
@limiter.limit(get_settings().rate_limit_quiz_drafts, exempt_when=is_admin_request)
async def patch_quiz_draft(
    request: Request,
    draft_id: PathID,
    req: QuizDraftPatchRequest,
    user_id: Annotated[str, Depends(get_current_user)],
) -> Any:
    """Update fields of an existing quiz draft."""
    del request
    service = QuizDraftService()
    try:
        draft = await service.patch_draft(draft_id, user_id, req)
    except Exception as exc:
        raise _service_error_to_http(exc) from exc
    return ok({"draft": public_draft(draft)})


@router.delete("/{draft_id}", response_model=StandardResponse[QuizDraftSingleResponse])
@limiter.limit(get_settings().rate_limit_quiz_drafts, exempt_when=is_admin_request)
async def delete_quiz_draft(
    request: Request,
    draft_id: PathID,
    user_id: Annotated[str, Depends(get_current_user)],
) -> Any:
    """Delete a quiz draft owned by the authenticated user."""
    del request
    service = QuizDraftService()
    try:
        draft = await service.delete_draft(draft_id, user_id)
    except Exception as exc:
        raise _service_error_to_http(exc) from exc
    return ok({"draft": public_draft(draft)})


@router.post("/{draft_id}/submit", response_model=StandardResponse[QuizDraftSingleResponse])
@limiter.limit(get_settings().rate_limit_quiz_drafts, exempt_when=is_admin_request)
async def submit_quiz_draft(
    request: Request,
    draft_id: PathID,
    req: QuizDraftSubmitRequest,
    user_id: Annotated[str, Depends(get_current_user)],
) -> Any:
    """Finalize a quiz draft and persist it as a published quiz."""
    del request
    service = QuizDraftService()
    try:
        draft = await service.mark_submitted(draft_id, user_id, req.quiz_id)
    except Exception as exc:
        raise _service_error_to_http(exc) from exc
    return ok({"draft": public_draft(draft)})
