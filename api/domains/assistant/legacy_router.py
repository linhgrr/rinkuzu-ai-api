"""Temporary compatibility routes for clients deployed before Ask Rin-chan v1.

These endpoints deliberately contain no tutor implementation. They translate the
legacy request contracts into the shared AskRinChanService so backend-first
deployments do not break an older frontend while the new routes roll out.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Annotated, Any, Literal, NoReturn

from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel, ConfigDict, Field
from sse_starlette import EventSourceResponse

from api.config import get_settings
from api.dependencies import (
    get_chunk_chroma_store,
    get_current_user,
    get_session_manager,
    resolve_user_session,
)
from api.domains.learning.router import (
    _build_rag_context,
    _resolve_exercise_options,
    _resolve_exercise_question,
)
from api.exceptions import AppError, ExerciseGenerationError
from api.rate_limit import is_admin_request, limiter
from api.schemas.common import StandardResponse, ok
from api.schemas.validators import PathID  # noqa: TC001 - FastAPI resolves at runtime.
from api.shared.llm import SSE_STREAM_HEADERS
from api.shared.llm_usage import LlmAction

from .service import (
    AskRinImageUnsupportedError,
    AskRinRequestContext,
    get_ask_rin_chan_service,
)

legacy_quiz_router = APIRouter(prefix="/api/v1/quiz", tags=["legacy-ask-rin-chan"])
legacy_session_router = APIRouter(prefix="/api/v1/session", tags=["legacy-ask-rin-chan"])


class LegacyChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=4_000)


class LegacyQuizTutorRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    question: str = Field(min_length=1, max_length=12_000)
    options: list[str] = Field(min_length=2, max_length=8)
    user_question: str | None = Field(default=None, alias="userQuestion", max_length=1_000)
    question_image: str | None = Field(default=None, alias="questionImage")
    option_images: list[str | None] = Field(default_factory=list, alias="optionImages")
    chat_history: list[LegacyChatMessage] = Field(default_factory=list, alias="chatHistory")
    stream: bool = False


class LegacyTutorChatRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    user_question: str = Field(alias="userQuestion", min_length=1, max_length=1_000)
    chat_history: list[LegacyChatMessage] = Field(default_factory=list, alias="chatHistory")
    stream: bool = False


class LegacyQuizTutorResponse(BaseModel):
    explanation: str
    structured: dict[str, Any] | None = None
    timestamp: str
    turn_count: int


class LegacyTutorChatResponse(BaseModel):
    explanation: str


def _raise_legacy_tutor_error(exc: Exception) -> NoReturn:
    if isinstance(exc, AskRinImageUnsupportedError):
        raise AppError(
            code="ask_rin_image_unsupported",
            message="Image question is unavailable",
            detail=str(exc),
            status_code=422,
        ) from exc
    if isinstance(exc, ValueError):
        raise AppError(
            code="validation_error",
            message="Invalid tutor request",
            detail=str(exc),
            status_code=400,
        ) from exc
    if isinstance(exc, RuntimeError):
        raise AppError(
            code="service_unavailable",
            message="Tutor service unavailable",
            detail=str(exc),
            status_code=502,
        ) from exc
    raise exc


@legacy_quiz_router.post(
    "/ask-ai",
    response_model=StandardResponse[LegacyQuizTutorResponse],
    deprecated=True,
)
@limiter.limit(get_settings().rate_limit_session, exempt_when=is_admin_request)
async def legacy_quiz_ask_ai(
    request: Request,
    req: LegacyQuizTutorRequest,
    user_id: Annotated[str, Depends(get_current_user)],
) -> Any:
    """Translate the removed quiz tutor route into the shared Ask Rin-chan service."""
    del request
    del user_id
    context = AskRinRequestContext(
        action=LlmAction.ASK_RIN_CHAN,
        question=req.question,
        options=req.options,
        user_question=req.user_question,
        chat_history=[item.model_dump() for item in req.chat_history],
        question_image=req.question_image,
        option_images=req.option_images,
    )
    service = get_ask_rin_chan_service()
    try:
        if req.stream:
            stream = await service.create_stream(context)
            return EventSourceResponse(
                stream,
                headers=SSE_STREAM_HEADERS,
                ping=15,
                send_timeout=30,
            )
        explanation = await service.generate_response(context)
    except Exception as exc:
        _raise_legacy_tutor_error(exc)

    return ok(
        {
            "explanation": explanation,
            "structured": None,
            "timestamp": datetime.now(UTC).isoformat(),
            "turn_count": len(req.chat_history) // 2 + 1,
        }
    )


@legacy_session_router.post(
    "/{session_id}/chat",
    response_model=StandardResponse[LegacyTutorChatResponse],
    deprecated=True,
)
@limiter.limit(get_settings().rate_limit_session, exempt_when=is_admin_request)
async def legacy_adaptive_chat(
    request: Request,
    session_id: PathID,
    req: LegacyTutorChatRequest,
    manager: Any = Depends(get_session_manager),
    user_id: str = Depends(get_current_user),
    chunk_store: Any = Depends(get_chunk_chroma_store),
) -> Any:
    """Translate the removed adaptive chat route into the shared Ask Rin-chan service."""
    del request
    session = await resolve_user_session(manager, session_id, user_id)
    exercise = session.current_exercise or (
        session.exercise_history[-1] if session.exercise_history else None
    )
    if exercise is None:
        raise ExerciseGenerationError("No exercise context available for chat")

    context = AskRinRequestContext(
        action=LlmAction.ASK_RIN_CHAN,
        question=_resolve_exercise_question(exercise),
        options=_resolve_exercise_options(exercise),
        user_question=req.user_question,
        chat_history=[item.model_dump() for item in req.chat_history],
        concept_name=exercise.concept_name,
        bloom_level=exercise.bloom_level,
        rag_context=await _build_rag_context(session, chunk_store, req.user_question, k=3),
    )
    service = get_ask_rin_chan_service()
    try:
        if req.stream:
            stream = await service.create_stream(context)
            return EventSourceResponse(
                stream,
                headers=SSE_STREAM_HEADERS,
                ping=15,
                send_timeout=30,
            )
        explanation = await service.generate_response(context)
    except Exception as exc:
        _raise_legacy_tutor_error(exc)

    return ok({"explanation": explanation})
