"""
Session router — Session lifecycle endpoints.
"""

from fastapi import APIRouter, Depends
from loguru import logger

from ..schemas import (
    SessionCreateRequest, SessionCreateResponse,
    NextConceptResponse, TheoryResponse, ExerciseResponse,
    SubmitAnswerRequest, SubmitAnswerResponse,
    SessionStatusResponse,
)
from ..dependencies import get_session_manager, get_session_service, get_current_user
from ..exceptions import SessionNotFoundError, SessionCompletedError, ExerciseGenerationError

router = APIRouter(prefix="/api/session", tags=["session"])

BLOOM_LABELS = {
    1: "Remember", 2: "Understand", 3: "Apply",
    4: "Analyze", 5: "Evaluate", 6: "Create",
}


@router.post("/start", response_model=SessionCreateResponse)
async def start_session(
    req: SessionCreateRequest,
    manager=Depends(get_session_manager),
    exercise_svc=Depends(get_session_service),
    user_id: str = Depends(get_current_user),
):
    session = manager.create_session(max_steps=req.max_steps, user_id=user_id)

    id_to_concept = {v: k for k, v in session.concept_map.items()}
    concepts = [
        {
            "id": id_to_concept.get(i, str(i)),
            "name": session.concept_names.get(id_to_concept.get(i, str(i)), str(i)),
            "index": i,
        }
        for i in range(len(session.concept_map))
    ]

    # Fire eager prefetch via exercise service
    try:
        import asyncio
        asyncio.create_task(exercise_svc.eager_generate_first_exercise(session))
    except Exception as exc:
        logger.warning(f"[SessionRouter] Failed to schedule eager prefetch: {exc}")

    return SessionCreateResponse(
        session_id=session.session_id,
        n_concepts=len(session.concept_map),
        concepts=concepts,
        status="active",
    )


@router.post("/{session_id}/next-concept", response_model=NextConceptResponse)
async def next_concept(
    session_id: str,
    manager=Depends(get_session_manager),
    exercise_svc=Depends(get_session_service),
    user_id: str = Depends(get_current_user),
):
    session = manager.get_session(session_id)
    if not session or getattr(session, "user_id", None) != user_id:
        raise SessionNotFoundError(session_id)
    if session.status != "active":
        raise SessionCompletedError(session_id)

    concept_info = await exercise_svc.get_next_concept(session)
    if not concept_info:
        raise ExerciseGenerationError("Failed to determine next concept")

    return NextConceptResponse(**concept_info)


@router.get("/{session_id}/theory", response_model=TheoryResponse)
async def theory(
    session_id: str,
    manager=Depends(get_session_manager),
    exercise_svc=Depends(get_session_service),
    user_id: str = Depends(get_current_user),
):
    session = manager.get_session(session_id)
    if not session or getattr(session, "user_id", None) != user_id:
        raise SessionNotFoundError(session_id)

    theory_data = await exercise_svc.get_theory(session)
    if not theory_data:
        raise ExerciseGenerationError("No pending concept to generate theory")

    return TheoryResponse(**theory_data)


@router.post("/{session_id}/exercise", response_model=ExerciseResponse)
async def generate_exercise(
    session_id: str,
    manager=Depends(get_session_manager),
    exercise_svc=Depends(get_session_service),
    user_id: str = Depends(get_current_user),
):
    session = manager.get_session(session_id)
    if not session or getattr(session, "user_id", None) != user_id:
        raise SessionNotFoundError(session_id)
    if session.status != "active":
        raise SessionCompletedError(session_id)

    exercise = await exercise_svc.generate_exercise(session)
    if not exercise:
        raise ExerciseGenerationError()

    env_stats = session.env.get_session_stats()

    return ExerciseResponse(
        exercise_id=exercise.exercise_id,
        concept_name=exercise.concept_name,
        concept_idx=exercise.concept_idx,
        bloom_level=exercise.bloom_level,
        bloom_label=BLOOM_LABELS.get(exercise.bloom_level, "Unknown"),
        question=exercise.question,
        options=exercise.options,
        step=env_stats["step"],
        max_steps=env_stats["max_steps"],
        theory=exercise.theory,
    )


@router.post("/{session_id}/submit", response_model=SubmitAnswerResponse)
async def submit_answer(
    session_id: str,
    req: SubmitAnswerRequest,
    manager=Depends(get_session_manager),
    exercise_svc=Depends(get_session_service),
    user_id: str = Depends(get_current_user),
):
    session = manager.get_session(session_id)
    if not session or getattr(session, "user_id", None) != user_id:
        raise SessionNotFoundError(session_id)

    result = await exercise_svc.submit_answer(session, req.answer)
    if not result:
        raise ExerciseGenerationError("No pending exercise or session not found")

    return SubmitAnswerResponse(**result)


@router.get("/{session_id}/status", response_model=SessionStatusResponse)
async def session_status(
    session_id: str,
    manager=Depends(get_session_manager),
    user_id: str = Depends(get_current_user),
):
    session = manager.get_session(session_id)
    if not session or getattr(session, "user_id", None) != user_id:
        raise SessionNotFoundError(session_id)
    status = manager.get_session_status(session_id)
    if not status:
        raise SessionNotFoundError(session_id)

    return SessionStatusResponse(**status)
