"""
Session router — Session lifecycle endpoints.
"""

import asyncio
from typing import Any

from fastapi import APIRouter, BackgroundTasks, Depends, Request
from fastapi.responses import StreamingResponse
from loguru import logger

from api.config import get_settings
from api.core.quiz.tutor_chat import (
    create_tutor_chat_stream,
    generate_tutor_chat_response,
    sanitize_chat_input,
)
from api.core.shared.llm import SSE_STREAM_HEADERS
from api.dependencies import (
    get_chunk_chroma_store,
    get_current_user,
    get_session_manager,
    get_session_service,
    resolve_user_session,
)
from api.exceptions import (
    AppError,
    ExerciseGenerationError,
    SessionCompletedError,
    SessionNotFoundError,
)
from api.rate_limit import is_admin_request, limiter
from api.schemas import (
    ExerciseResponse,
    NextConceptResponse,
    SessionCreateRequest,
    SessionCreateResponse,
    SessionStatusResponse,
    SubmitAnswerRequest,
    SubmitAnswerResponse,
    TheoryResponse,
    TutorChatRequest,
    TutorChatResponse,
)
from api.schemas.common import StandardResponse, ok
from api.schemas.validators import PathID

router = APIRouter(prefix="/api/session", tags=["session"])

BLOOM_LABELS = {
    1: "Remember",
    2: "Understand",
    3: "Apply",
    4: "Analyze",
    5: "Evaluate",
    6: "Create",
}


async def _get_tutor_chat_history(session: Any, exercise_id: str) -> list[dict[str, str]]:
    async with session._lock:
        if session.tutor_chat_exercise_id != exercise_id:
            session.tutor_chat_exercise_id = exercise_id
            session.tutor_chat_history = []
        return [dict(item) for item in session.tutor_chat_history]


async def _append_tutor_chat_turn(
    session: Any,
    *,
    exercise_id: str,
    user_question: str,
    assistant_response: str,
) -> None:
    sanitized_user_question = sanitize_chat_input(user_question)
    sanitized_assistant_response = (
        str(assistant_response).replace("<", "").replace(">", "").strip()[:4000]
    )

    if not sanitized_user_question or not sanitized_assistant_response:
        return

    async with session._lock:
        if session.tutor_chat_exercise_id != exercise_id:
            session.tutor_chat_exercise_id = exercise_id
            session.tutor_chat_history = []

        session.tutor_chat_history.extend(
            [
                {"role": "user", "content": sanitized_user_question},
                {"role": "assistant", "content": sanitized_assistant_response},
            ]
        )
        session.tutor_chat_history = session.tutor_chat_history[-12:]


async def _build_rag_context(
    session: Any,
    chunk_store: Any,
    user_question: str,
    k: int = 3,
) -> str:
    """Retrieve top-k relevant document chunks for the user's question.

    Returns an empty string if retrieval fails or no chunks are found.
    """
    if chunk_store is None or not session.job_id:
        return ""

    try:
        docs = await chunk_store.aretrieve(
            query=user_question,
            job_id=session.job_id,
            k=k,
        )
        if not docs:
            return ""

        blocks = []
        for i, doc in enumerate(docs, 1):
            page = doc.metadata.get("start_page", "?")
            blocks.append(f"[Đoạn {i}] (trang {page})\n{doc.page_content}")
        return "\n\n".join(blocks)
    except BaseException as exc:
        logger.warning("[RAG] Retrieval failed, continuing without context: {}", exc)
        return ""


@router.post("/start", response_model=StandardResponse[SessionCreateResponse])
@limiter.limit(get_settings().rate_limit_session, exempt_when=is_admin_request)
async def start_session(
    request: Request,
    req: SessionCreateRequest,
    background_tasks: BackgroundTasks,
    manager: Any = Depends(get_session_manager),
    exercise_svc: Any = Depends(get_session_service),
    user_id: str = Depends(get_current_user),
) -> Any:
    """Create a new adaptive learning session."""
    del request
    session = await manager.create_session(max_steps=req.max_steps, user_id=user_id)

    id_to_concept = {v: k for k, v in session.concept_map.items()}
    concepts = [
        {
            "id": id_to_concept.get(i, str(i)),
            "name": session.concept_names.get(id_to_concept.get(i, str(i)), str(i)),
            "index": i,
        }
        for i in range(len(session.concept_map))
    ]

    # Fire eager prefetch via built-in BackgroundTasks
    try:
        background_tasks.add_task(exercise_svc.eager_generate_first_exercise, session)
    except TypeError as exc:
        logger.warning("[SessionRouter] Failed to schedule eager prefetch: {}", exc)

    return ok(
        SessionCreateResponse(
            session_id=session.session_id,
            n_concepts=len(session.concept_map),
            concepts=concepts,
            status="active",
        ).model_dump()
    )


@router.post("/{session_id}/next-concept", response_model=StandardResponse[NextConceptResponse])
@limiter.limit(get_settings().rate_limit_session, exempt_when=is_admin_request)
async def next_concept(
    request: Request,
    session_id: PathID,
    manager: Any = Depends(get_session_manager),
    exercise_svc: Any = Depends(get_session_service),
    user_id: str = Depends(get_current_user),
) -> Any:
    """Recommend the next concept to study based on mastery and prerequisites."""
    del request
    session = await resolve_user_session(manager, session_id, user_id)
    if session.status != "active":
        raise SessionCompletedError(session_id)

    concept_info = await exercise_svc.get_next_concept(session)
    if not concept_info:
        raise ExerciseGenerationError("Failed to determine next concept")

    return ok(NextConceptResponse(**concept_info).model_dump())


@router.get("/{session_id}/theory", response_model=StandardResponse[TheoryResponse])
@limiter.limit(get_settings().rate_limit_session, exempt_when=is_admin_request)
async def theory(
    request: Request,
    session_id: PathID,
    manager: Any = Depends(get_session_manager),
    exercise_svc: Any = Depends(get_session_service),
    user_id: str = Depends(get_current_user),
) -> Any:
    """Retrieve theory content for the current pending concept."""
    del request
    session = await resolve_user_session(manager, session_id, user_id)

    theory_data = await exercise_svc.get_theory(session)
    if not theory_data:
        raise ExerciseGenerationError("No pending concept to generate theory")

    return ok(TheoryResponse(**theory_data).model_dump())


@router.post("/{session_id}/exercise", response_model=StandardResponse[ExerciseResponse])
@limiter.limit(get_settings().rate_limit_session, exempt_when=is_admin_request)
async def generate_exercise(
    request: Request,
    session_id: PathID,
    background_tasks: BackgroundTasks,
    manager: Any = Depends(get_session_manager),
    exercise_svc: Any = Depends(get_session_service),
    user_id: str = Depends(get_current_user),
) -> Any:
    """Generate an exercise for the current concept at the appropriate Bloom level."""
    del request
    session = await resolve_user_session(manager, session_id, user_id)
    if session.status != "active":
        raise SessionCompletedError(session_id)

    exercise = await exercise_svc.generate_exercise(session, background_tasks)
    if not exercise:
        raise ExerciseGenerationError

    env_stats = session.env.get_session_stats()

    from api.core.learning.exercise_types.registry import get_handler

    content = get_handler(exercise.payload.exercise_type).to_response_dict(exercise)
    return ok(
        ExerciseResponse(
            exercise_id=exercise.exercise_id,
            concept_name=exercise.concept_name,
            concept_idx=exercise.concept_idx,
            bloom_level=exercise.bloom_level,
            bloom_label=BLOOM_LABELS.get(exercise.bloom_level, "Unknown"),
            exercise_type=exercise.payload.exercise_type,
            question=exercise.question,
            sentence=content.get("sentence"),
            options=content.get("options", {}),
            statement=content.get("statement"),
            hint=content.get("hint"),
            items=content.get("items", []),
            pairs=content.get("pairs", []),
            right_items=content.get("right_items", []),
            step=env_stats["step"],
            max_steps=env_stats["max_steps"],
            theory=exercise.theory,
            recommendation_reason=getattr(session, "_current_recommendation_reason", None),
        ).model_dump()
    )


@router.post("/{session_id}/submit", response_model=StandardResponse[SubmitAnswerResponse])
@limiter.limit(get_settings().rate_limit_session, exempt_when=is_admin_request)
async def submit_answer(
    request: Request,
    session_id: PathID,
    req: SubmitAnswerRequest,
    background_tasks: BackgroundTasks,
    manager: Any = Depends(get_session_manager),
    exercise_svc: Any = Depends(get_session_service),
    user_id: str = Depends(get_current_user),
) -> Any:
    """Evaluate the user's answer and update BKT mastery estimates."""
    del request
    session = await resolve_user_session(manager, session_id, user_id)

    result = await exercise_svc.submit_answer(session, req.answer.model_dump(), background_tasks)
    if not result:
        raise ExerciseGenerationError("No pending exercise or session not found")

    return ok(SubmitAnswerResponse(**result).model_dump())


def _resolve_exercise_question(exercise: Any) -> str:
    from api.core.learning.exercise_types.registry import get_handler

    return get_handler(exercise.payload.exercise_type).tutor_question(exercise)


def _resolve_exercise_options(exercise: Any) -> list[str]:
    from api.core.learning.exercise_types.registry import get_handler

    return get_handler(exercise.payload.exercise_type).tutor_options(exercise)


@router.post("/{session_id}/chat", response_model=StandardResponse[TutorChatResponse])
@limiter.limit(get_settings().rate_limit_tutor_chat, exempt_when=is_admin_request)
async def chat_about_exercise(
    request: Request,
    session_id: PathID,
    req: TutorChatRequest,
    manager: Any = Depends(get_session_manager),
    user_id: str = Depends(get_current_user),
    chunk_store: Any = Depends(get_chunk_chroma_store),
) -> Any:
    """Chat with an AI tutor about the current exercise, with optional RAG context."""
    del request
    session = await resolve_user_session(manager, session_id, user_id)
    exercise = session.current_exercise or (
        session.exercise_history[-1] if session.exercise_history else None
    )
    if not exercise:
        raise ExerciseGenerationError("No exercise context available for chat")

    question = _resolve_exercise_question(exercise)
    options = _resolve_exercise_options(exercise)

    chat_history = await _get_tutor_chat_history(session, exercise.exercise_id)

    # RAG: retrieve relevant document chunks for this question
    rag_context = await _build_rag_context(
        session,
        chunk_store,
        req.user_question,
        k=3,
    )

    try:
        if req.stream:

            async def persist_chat_history(full_response: str) -> None:
                await _append_tutor_chat_turn(
                    session,
                    exercise_id=exercise.exercise_id,
                    user_question=req.user_question,
                    assistant_response=full_response,
                )

            stream = await create_tutor_chat_stream(
                question=question,
                options=options,
                user_question=req.user_question,
                chat_history=chat_history,
                concept_name=exercise.concept_name,
                bloom_level=exercise.bloom_level,
                rag_context=rag_context,
                on_complete=persist_chat_history,
            )
            return StreamingResponse(
                stream,
                media_type="text/event-stream",
                headers=SSE_STREAM_HEADERS,
            )

        explanation = await asyncio.to_thread(
            generate_tutor_chat_response,
            question=question,
            options=options,
            user_question=req.user_question,
            chat_history=chat_history,
            concept_name=exercise.concept_name,
            bloom_level=exercise.bloom_level,
            rag_context=rag_context,
        )
        await _append_tutor_chat_turn(
            session,
            exercise_id=exercise.exercise_id,
            user_question=req.user_question,
            assistant_response=explanation,
        )
    except ValueError as exc:
        logger.warning("[SessionRouter] Tutor chat ValueError: {}", exc)
        raise AppError(
            code="validation_error",
            message="Invalid tutor chat request",
            detail=str(exc),
            status_code=400,
        ) from exc
    except RuntimeError as exc:
        logger.warning("[SessionRouter] Tutor chat RuntimeError: {}", exc)
        raise AppError(
            code="service_unavailable",
            message="Tutor service temporarily unavailable",
            detail=str(exc),
            status_code=502,
        ) from exc
    else:
        return ok({"explanation": explanation})
    # Unexpected errors fall through to the global unexpected_exception_handler.


@router.get("/{session_id}/status", response_model=StandardResponse[SessionStatusResponse])
async def session_status(
    session_id: PathID,
    manager: Any = Depends(get_session_manager),
    user_id: str = Depends(get_current_user),
) -> Any:
    """Return the current status and progress of a learning session."""
    await resolve_user_session(manager, session_id, user_id)
    status = manager.get_session_status(session_id)
    if not status:
        raise SessionNotFoundError(session_id)

    return ok(SessionStatusResponse(**status).model_dump())
