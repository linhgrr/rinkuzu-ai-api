"""
Session router — Session lifecycle endpoints.
"""

import asyncio

from fastapi import APIRouter, BackgroundTasks, Depends, Request
from fastapi.responses import JSONResponse, StreamingResponse
from loguru import logger

from api.config import get_settings
from api.core.quiz.tutor_chat import (
    create_tutor_chat_stream,
    generate_tutor_chat_response,
    sanitize_chat_input,
)
from api.dependencies import (
    get_chunk_chroma_store,
    get_current_user,
    get_session_manager,
    get_session_service,
)
from api.exceptions import ExerciseGenerationError, SessionCompletedError, SessionNotFoundError
from api.schemas.common import StandardResponse
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

router = APIRouter(prefix="/api/session", tags=["session"])

BLOOM_LABELS = {
    1: "Remember",
    2: "Understand",
    3: "Apply",
    4: "Analyze",
    5: "Evaluate",
    6: "Create",
}


async def _resolve_user_session(manager, session_id: str, user_id: str):
    session = await manager.get_or_recover_session(session_id, user_id)
    if not session:
        raise SessionNotFoundError(session_id)
    return session


async def _get_tutor_chat_history(session, exercise_id: str) -> list[dict[str, str]]:
    async with session._lock:
        if session.tutor_chat_exercise_id != exercise_id:
            session.tutor_chat_exercise_id = exercise_id
            session.tutor_chat_history = []
        return [dict(item) for item in session.tutor_chat_history]


async def _append_tutor_chat_turn(
    session,
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
    session,
    chunk_store,
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
async def start_session(
    req: SessionCreateRequest,
    background_tasks: BackgroundTasks,
    manager=Depends(get_session_manager),
    exercise_svc=Depends(get_session_service),
    user_id: str = Depends(get_current_user),
):
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

    return {
        "success": True,
        "data": SessionCreateResponse(
            session_id=session.session_id,
            n_concepts=len(session.concept_map),
            concepts=concepts,
            status="active",
        ).model_dump()
    }


@router.post("/{session_id}/next-concept", response_model=StandardResponse[NextConceptResponse])
async def next_concept(
    session_id: str,
    manager=Depends(get_session_manager),
    exercise_svc=Depends(get_session_service),
    user_id: str = Depends(get_current_user),
):
    session = await _resolve_user_session(manager, session_id, user_id)
    if session.status != "active":
        raise SessionCompletedError(session_id)

    concept_info = await exercise_svc.get_next_concept(session)
    if not concept_info:
        raise ExerciseGenerationError("Failed to determine next concept")

    return {"success": True, "data": NextConceptResponse(**concept_info).model_dump()}


@router.get("/{session_id}/theory", response_model=StandardResponse[TheoryResponse])
async def theory(
    session_id: str,
    manager=Depends(get_session_manager),
    exercise_svc=Depends(get_session_service),
    user_id: str = Depends(get_current_user),
):
    session = await _resolve_user_session(manager, session_id, user_id)

    theory_data = await exercise_svc.get_theory(session)
    if not theory_data:
        raise ExerciseGenerationError("No pending concept to generate theory")

    return {"success": True, "data": TheoryResponse(**theory_data).model_dump()}


@router.post("/{session_id}/exercise", response_model=StandardResponse[ExerciseResponse])
async def generate_exercise(
    session_id: str,
    background_tasks: BackgroundTasks,
    manager=Depends(get_session_manager),
    exercise_svc=Depends(get_session_service),
    user_id: str = Depends(get_current_user),
):
    session = await _resolve_user_session(manager, session_id, user_id)
    if session.status != "active":
        raise SessionCompletedError(session_id)

    exercise = await exercise_svc.generate_exercise(session, background_tasks)
    if not exercise:
        raise ExerciseGenerationError

    env_stats = session.env.get_session_stats()

    return {
        "success": True,
        "data": ExerciseResponse(
            exercise_id=exercise.exercise_id,
            concept_name=exercise.concept_name,
            concept_idx=exercise.concept_idx,
            bloom_level=exercise.bloom_level,
            bloom_label=BLOOM_LABELS.get(exercise.bloom_level, "Unknown"),
            exercise_type=exercise.exercise_type,
            question=exercise.question,
            sentence=exercise.sentence,
            options=exercise.options,
            statement=exercise.statement,
            hint=exercise.hint,
            items=exercise.items,
            pairs=exercise.pairs,
            right_items=exercise.right_items,
            step=env_stats["step"],
            max_steps=env_stats["max_steps"],
            theory=exercise.theory,
            recommendation_reason=getattr(session, "_current_recommendation_reason", None),
        ).model_dump()
    }


@router.post("/{session_id}/submit", response_model=StandardResponse[SubmitAnswerResponse])
async def submit_answer(
    session_id: str,
    req: SubmitAnswerRequest,
    background_tasks: BackgroundTasks,
    manager=Depends(get_session_manager),
    exercise_svc=Depends(get_session_service),
    user_id: str = Depends(get_current_user),
):
    session = await _resolve_user_session(manager, session_id, user_id)

    result = await exercise_svc.submit_answer(session, req.answer.model_dump(), background_tasks)
    if not result:
        raise ExerciseGenerationError("No pending exercise or session not found")

    return {"success": True, "data": SubmitAnswerResponse(**result).model_dump()}


def _resolve_exercise_options(exercise) -> list[str]:
    """Return the display options list for a given exercise."""
    option_keys = sorted(exercise.options.keys())
    options = [exercise.options[key] for key in option_keys if exercise.options.get(key)]
    if options:
        return options
    fallbacks: dict[str, list[str] | None] = {
        "true_false": ["True", "False"],
        "ordering": exercise.items,
        "matching": exercise.right_items,
    }
    if exercise.exercise_type in fallbacks:
        return fallbacks[exercise.exercise_type] or []
    if exercise.exercise_type == "fill_blank" and exercise.hint:
        return [f"Gợi ý: {exercise.hint}"]
    if exercise.exercise_type == "short_answer":
        return exercise.rubric or ["Trả lời ngắn gọn, bám sát câu hỏi."]
    return ["Xem lại yêu cầu của bài tập hiện tại."]


@router.post("/{session_id}/chat", response_model=StandardResponse[TutorChatResponse])
@limiter.limit(get_settings().rate_limit_tutor_chat, exempt_when=is_admin_request)
async def chat_about_exercise(
    request: Request,
    session_id: str,
    req: TutorChatRequest,
    manager=Depends(get_session_manager),
    user_id: str = Depends(get_current_user),
    chunk_store=Depends(get_chunk_chroma_store),
):
    del request
    session = await _resolve_user_session(manager, session_id, user_id)
    exercise = session.current_exercise or (
        session.exercise_history[-1] if session.exercise_history else None
    )
    if not exercise:
        raise ExerciseGenerationError("No exercise context available for chat")

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
                question=exercise.question,
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
                headers={
                    "Cache-Control": "no-cache, no-transform",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                },
            )

        explanation = await asyncio.to_thread(
            generate_tutor_chat_response,
            question=exercise.question,
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
        return {"success": True, "data": {"explanation": explanation}}
    except ValueError as exc:
        logger.warning("[SessionRouter] Tutor chat ValueError: {}", exc)
        raise AppError("Invalid tutor chat request", status_code=400)
    except RuntimeError as exc:
        logger.warning("[SessionRouter] Tutor chat RuntimeError: {}", exc)
        raise AppError("Tutor service temporarily unavailable", status_code=502)
    # Unexpected errors fall through to the global unexpected_exception_handler.


@router.get("/{session_id}/status", response_model=StandardResponse[SessionStatusResponse])
async def session_status(
    session_id: str,
    manager=Depends(get_session_manager),
    user_id: str = Depends(get_current_user),
):
    await _resolve_user_session(manager, session_id, user_id)
    status = manager.get_session_status(session_id)
    if not status:
        raise SessionNotFoundError(session_id)

    return {"success": True, "data": SessionStatusResponse(**status).model_dump()}
