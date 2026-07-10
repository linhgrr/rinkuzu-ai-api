"""Quiz domain HTTP endpoints: draft processing + ask-AI tutor."""

from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from sse_starlette import EventSourceResponse

from api.config import get_settings
from api.dependencies import get_current_user
from api.exceptions import AppError
from api.rate_limit import is_admin_request, limiter
from api.schemas.common import StandardResponse, ok
from api.schemas.validators import PathID
from api.shared.llm import SSE_STREAM_HEADERS

from .draft_service import (
    QuizDraftDependencyError,
    QuizDraftNotFoundError,
    QuizDraftService,
    QuizDraftValidationError,
    public_draft,
)
from .draft_tasks import quiz_draft_task_manager
from .quiz_tutor import create_quiz_tutor_stream, generate_quiz_tutor_response
from .schemas import (
    QuizDraftCreateRequest,
    QuizDraftListResponse,
    QuizDraftPatchRequest,
    QuizDraftSingleResponse,
    QuizDraftSubmitRequest,
    QuizTutorRequest,
    QuizTutorResponseData,
)

drafts_router = APIRouter(prefix="/api/v1/quiz/drafts", tags=["quiz-drafts"])
tutor_router = APIRouter(prefix="/api/v1/quiz", tags=["quiz"])


def _service_error_to_http(exc: Exception) -> HTTPException:
    if isinstance(exc, QuizDraftNotFoundError):
        return HTTPException(status_code=404, detail=str(exc))
    if isinstance(exc, QuizDraftValidationError):
        return HTTPException(status_code=400, detail=str(exc))
    if isinstance(exc, QuizDraftDependencyError):
        return HTTPException(status_code=503, detail=str(exc))
    return HTTPException(status_code=500, detail="Quiz draft operation failed.")


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
        raise _service_error_to_http(exc) from exc

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
    drafts = await service.list_drafts(user_id, limit)
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
        raise _service_error_to_http(exc) from exc
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
        raise _service_error_to_http(exc) from exc
    return ok({"draft": public_draft(draft)})


@drafts_router.delete("/{draft_id}", response_model=StandardResponse[QuizDraftSingleResponse])
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
    await quiz_draft_task_manager.cancel(draft_id)
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
        raise _service_error_to_http(exc) from exc
    return ok({"draft": public_draft(draft)})


@tutor_router.post("/ask-ai", response_model=StandardResponse[QuizTutorResponseData])
@limiter.limit(get_settings().rate_limit_ask_ai, exempt_when=is_admin_request)
async def ask_ai_about_quiz(
    request: Request,
    req: QuizTutorRequest,
    user_id: Annotated[str, Depends(get_current_user)],
) -> Any:
    """Ask an AI tutor for help understanding a quiz question (stream or single response)."""
    del request
    del user_id

    try:
        if req.stream:
            stream = await create_quiz_tutor_stream(
                question=req.question,
                options=req.options,
                user_question=req.user_question,
                chat_history=[item.model_dump() for item in req.chat_history],
                question_image=req.question_image,
                option_images=req.option_images,
            )
            return EventSourceResponse(
                stream,
                headers=SSE_STREAM_HEADERS,
                ping=15,
                send_timeout=30,
            )

        payload = await generate_quiz_tutor_response(
            question=req.question,
            options=req.options,
            user_question=req.user_question,
            chat_history=[item.model_dump() for item in req.chat_history],
            question_image=req.question_image,
            option_images=req.option_images,
        )
        data = QuizTutorResponseData.model_validate(payload)
        return ok(data.model_dump())
    except ValueError as exc:
        raise AppError(
            code="validation_error",
            message="Invalid tutor request",
            detail=str(exc),
            status_code=400,
        ) from exc
    except RuntimeError as exc:
        raise AppError(
            code="service_unavailable",
            message="Tutor service unavailable",
            detail=str(exc),
            status_code=502,
        ) from exc
