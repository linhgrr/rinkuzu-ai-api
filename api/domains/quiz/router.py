"""Quiz draft processing endpoints."""

from typing import Annotated, Any, NoReturn

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from loguru import logger

from api.config import get_settings
from api.dependencies import get_current_user
from api.exceptions import AppError
from api.rate_limit import is_admin_request, limiter
from api.schemas.common import StandardResponse, ok
from api.schemas.validators import PathID
from api.shared.persistence.common import is_storage_infra_error

from .draft_service import (
    QuizDraftDependencyError,
    QuizDraftNotFoundError,
    QuizDraftService,
    QuizDraftValidationError,
    public_draft,
)
from .draft_tasks import quiz_draft_task_manager
from .schemas import (
    QuizDraftCreateRequest,
    QuizDraftListResponse,
    QuizDraftPatchRequest,
    QuizDraftSingleResponse,
    QuizDraftSubmitRequest,
)

drafts_router = APIRouter(prefix="/api/v1/quiz/drafts", tags=["quiz-drafts"])


def _service_error_to_http(exc: Exception) -> NoReturn:
    """Map known draft errors; re-raise unexpected so the global handler returns 500.

    Storage infrastructure → AppError 503 retryable. Known NotFound/Validation stay
    404/400. Dependency (config) stays 503. Programmer/Pydantic errors are not
    converted to DependencyError or synthetic HTTPException 500.
    """
    if isinstance(exc, QuizDraftNotFoundError):
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    if isinstance(exc, QuizDraftValidationError):
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if isinstance(exc, QuizDraftDependencyError):
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    if isinstance(exc, AppError):
        raise exc
    if is_storage_infra_error(exc):
        logger.exception("[quiz_draft] storage infra error")
        raise AppError(
            code="service_unavailable",
            message="Quiz draft storage unavailable",
            detail="Unable to complete quiz draft operation; retry may succeed",
            status_code=503,
            meta={"retryable": True},
        ) from exc
    raise exc


@drafts_router.post("", response_model=StandardResponse[QuizDraftSingleResponse])
@limiter.limit(get_settings().rate_limit_quiz_drafts, exempt_when=is_admin_request)
async def create_quiz_draft(
    request: Request,
    req: QuizDraftCreateRequest,
    user_id: Annotated[str, Depends(get_current_user)],
) -> Any:
    """Create a new quiz draft and enqueue AI processing."""
    del request
    service = QuizDraftService()
    try:
        draft = await service.create_draft(req, user_id)
    except Exception as exc:
        _service_error_to_http(exc)

    quiz_draft_task_manager.schedule(draft["draft_id"], user_id)
    return ok({"draft": public_draft(draft)})


@drafts_router.get("", response_model=StandardResponse[QuizDraftListResponse])
@limiter.limit(get_settings().rate_limit_quiz_drafts, exempt_when=is_admin_request)
async def list_quiz_drafts(
    request: Request,
    user_id: Annotated[str, Depends(get_current_user)],
    limit: Annotated[int, Query(ge=1, le=50)] = 20,
) -> Any:
    """List recent quiz drafts for the authenticated user."""
    del request
    service = QuizDraftService()
    try:
        drafts = await service.list_drafts(user_id, limit)
    except Exception as exc:
        _service_error_to_http(exc)
    return ok({"drafts": [public_draft(draft) for draft in drafts]})


@drafts_router.get("/{draft_id}", response_model=StandardResponse[QuizDraftSingleResponse])
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
        _service_error_to_http(exc)
    return ok({"draft": public_draft(draft)})


@drafts_router.patch("/{draft_id}", response_model=StandardResponse[QuizDraftSingleResponse])
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
        _service_error_to_http(exc)
    return ok({"draft": public_draft(draft)})


@drafts_router.delete("/{draft_id}", response_model=StandardResponse[QuizDraftSingleResponse])
@limiter.limit(get_settings().rate_limit_quiz_drafts, exempt_when=is_admin_request)
async def delete_quiz_draft(
    request: Request,
    draft_id: PathID,
    user_id: Annotated[str, Depends(get_current_user)],
) -> Any:
    """Delete a quiz draft owned by the authenticated user.

    Observed delete is 200; genuine absence is 404; infrastructure errors are
    503. Not universally idempotent after metadata removal.
    """
    del request
    service = QuizDraftService()
    try:
        draft = await service.delete_draft(draft_id, user_id)
    except Exception as exc:
        _service_error_to_http(exc)
    return ok({"draft": public_draft(draft)})


@drafts_router.post("/{draft_id}/submit", response_model=StandardResponse[QuizDraftSingleResponse])
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
        _service_error_to_http(exc)
    return ok({"draft": public_draft(draft)})
