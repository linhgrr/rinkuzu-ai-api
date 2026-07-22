"""
Session router — Session lifecycle endpoints.
"""

from typing import Any

from fastapi import APIRouter, BackgroundTasks, Depends, Request
from loguru import logger

from api.config import get_settings
from api.dependencies import (
    get_current_user,
    get_session_manager,
    get_session_service,
    resolve_user_session,
)
from api.domains.assistant.context_tokens import ExerciseContext, issue_context_token
from api.domains.learning.bloom import BLOOM_LABELS
from api.exceptions import (
    AppError,
    ExerciseGenerationError,
    SessionCompletedError,
    SessionNotFoundError,
)
from api.rate_limit import is_admin_request, limiter
from api.schemas.common import StandardResponse, ok
from api.schemas.validators import PathID
from api.shared.persistence import load_pipeline_job_for_user

from .schemas import (
    ExerciseResponse,
    LearningStepResponse,
    NextConceptResponse,
    SessionCreateRequest,
    SessionCreateResponse,
    SessionStatusResponse,
    SubmitAnswerRequest,
    SubmitAnswerResponse,
    TheoryResponse,
)

router = APIRouter(prefix="/api/v1/session", tags=["session"])


def _build_exercise_response_payload(
    exercise: Any, env_stats: dict[str, Any], content: dict[str, Any]
) -> dict[str, Any]:
    """Build the public exercise response without exposing canonical answers."""
    exercise_type = exercise.payload.exercise_type
    payload: dict[str, Any] = {
        "exercise_id": exercise.exercise_id,
        "concept_name": exercise.concept_name,
        "concept_idx": exercise.concept_idx,
        "bloom_level": exercise.bloom_level,
        "bloom_label": BLOOM_LABELS.get(exercise.bloom_level, "Unknown"),
        "exercise_type": exercise_type,
        "question": exercise.question,
        "step": env_stats["step"],
        "max_steps": env_stats["max_steps"],
        "theory": exercise.theory,
        "recommendation_reason": None,
    }

    match exercise_type:
        case "mcq" | "multi_correct":
            payload["options"] = content["options"]
        case "true_false":
            payload["statement"] = content["statement"]
        case "fill_blank":
            payload["sentence"] = content["sentence"]
            payload["hint"] = content["hint"]
        case "ordering":
            payload["items"] = content["items"]
        case "matching":
            payload["pairs"] = content["pairs"]
            payload["right_items"] = content["right_items"]
        case "short_answer":
            payload["rubric"] = content["rubric"]

    return payload


def _build_public_exercise_payload(session: Any, exercise: Any) -> dict[str, Any]:
    env_stats = session.env.get_session_stats()

    from api.domains.learning.exercise_types.registry import get_handler

    content = get_handler(exercise.payload.exercise_type).to_response_dict(exercise)
    payload = _build_exercise_response_payload(exercise, env_stats, content)
    if not session.user_id:
        raise AppError(
            code="service_unavailable",
            message="Exercise context unavailable",
            detail="The learning session has no authenticated owner",
            status_code=503,
        )
    context_id = f"adaptive:{session.session_id}:{exercise.exercise_id}"
    payload["exercise_context_id"] = context_id
    payload["exercise_context_token"] = issue_context_token(
        ExerciseContext(
            context_id=context_id,
            user_id=session.user_id,
            question=_resolve_exercise_question(exercise),
            options=_resolve_exercise_options(exercise),
            concept_name=exercise.concept_name,
            bloom_level=exercise.bloom_level,
            session_id=session.session_id,
            exercise_id=exercise.exercise_id,
        )
    )
    payload["recommendation_reason"] = session._current_recommendation_reason
    return payload


async def _build_rag_context(
    session: Any,
    chunk_store: Any,
    user_question: str,
    k: int = 3,
) -> str:
    """Retrieve top-k relevant document chunks for the user's question.

    Returns an empty string if retrieval fails or no chunks are found.
    """
    if chunk_store is None or not session.job_id or not session.user_id:
        return ""

    try:
        job = await load_pipeline_job_for_user(session.job_id, session.user_id)
        if not job or job.get("status") != "completed":
            return ""
        generation = int(job.get("retry_count") or 0)
        docs = await chunk_store.aretrieve(
            query=user_question,
            job_id=session.job_id,
            generation=generation,
            k=k,
        )
        if not docs:
            return ""

        blocks = []
        for i, doc in enumerate(docs, 1):
            page = doc.metadata.get("start_page", "?")
            blocks.append(f"[Đoạn {i}] (trang {page})\n{doc.page_content}")
        return "\n\n".join(blocks)
    except Exception as exc:
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

    id_to_concept = session.id_to_concept
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

    if not await manager.persist_subject_progress(session):
        manager.remove_session(session.session_id)
        raise ExerciseGenerationError("Failed to persist current exercise")

    payload = _build_public_exercise_payload(session, exercise)
    return ok(payload)


@router.post("/{session_id}/step", response_model=StandardResponse[LearningStepResponse])
@limiter.limit(get_settings().rate_limit_session, exempt_when=is_admin_request)
async def learning_step(
    request: Request,
    session_id: PathID,
    background_tasks: BackgroundTasks,
    manager: Any = Depends(get_session_manager),
    exercise_svc: Any = Depends(get_session_service),
    user_id: str = Depends(get_current_user),
) -> Any:
    """Return the current learning step, creating and persisting one when needed."""
    del request
    session = await resolve_user_session(manager, session_id, user_id)
    if session.status != "active":
        raise SessionCompletedError(session_id)

    step = await exercise_svc.get_or_create_learning_step(session, background_tasks)
    if not step:
        raise ExerciseGenerationError("Failed to prepare learning step")

    payload = {
        "concept": step["concept"],
        "theory": step["theory"],
        "exercise": _build_public_exercise_payload(session, step["exercise"]),
        "cache_status": step["cache_status"],
    }
    return ok(LearningStepResponse(**payload).model_dump())


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

    result = await exercise_svc.submit_answer(
        session,
        req.answer.model_dump(),
        background_tasks,
        exercise_id=req.exercise_id,
        idempotency_key=req.idempotency_key,
    )
    if not result:
        raise ExerciseGenerationError("No pending exercise or session not found")

    return ok(SubmitAnswerResponse(**result).model_dump())


def _resolve_exercise_question(exercise: Any) -> str:
    from api.domains.learning.exercise_types.registry import get_handler

    return get_handler(exercise.payload.exercise_type).tutor_question(exercise)


def _resolve_exercise_options(exercise: Any) -> list[str]:
    from api.domains.learning.exercise_types.registry import get_handler

    return get_handler(exercise.payload.exercise_type).tutor_options(exercise)


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
