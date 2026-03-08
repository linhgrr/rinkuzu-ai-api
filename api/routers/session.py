"""
Session router — Session lifecycle endpoints
"""

from fastapi import APIRouter, HTTPException

from ..schemas import (
    SessionCreateRequest, SessionCreateResponse,
    NextConceptResponse, TheoryResponse, ExerciseResponse, 
    SubmitAnswerRequest, SubmitAnswerResponse,
    SessionStatusResponse,
)

router = APIRouter(prefix="/api/session", tags=["session"])

# Will be set by main.py on startup
session_manager = None

BLOOM_LABELS = {
    1: "Remember", 2: "Understand", 3: "Apply",
    4: "Analyze", 5: "Evaluate", 6: "Create",
}


@router.post("/start", response_model=SessionCreateResponse)
async def start_session(req: SessionCreateRequest):
    if session_manager is None:
        raise HTTPException(500, "Server not initialized")

    session = session_manager.create_session(max_steps=req.max_steps)

    id_to_concept = {v: k for k, v in session.concept_map.items()}
    concepts = [
        {
            "id": id_to_concept.get(i, str(i)),
            "name": session.concept_names.get(id_to_concept.get(i, str(i)), str(i)),
            "index": i,
        }
        for i in range(len(session.concept_map))
    ]

    return SessionCreateResponse(
        session_id=session.session_id,
        n_concepts=len(session.concept_map),
        concepts=concepts,
        status="active",
    )


@router.post("/{session_id}/next-concept", response_model=NextConceptResponse)
async def next_concept(session_id: str):
    if session_manager is None:
        raise HTTPException(500, "Server not initialized")

    session = session_manager.get_session(session_id)
    if not session:
        raise HTTPException(404, "Session not found")
    if session.status != "active":
        raise HTTPException(400, "Session is completed")

    concept_info = await session_manager.get_next_concept(session_id)
    if not concept_info:
        raise HTTPException(500, "Failed to determine next concept")

    return NextConceptResponse(**concept_info)


@router.get("/{session_id}/theory", response_model=TheoryResponse)
async def theory(session_id: str):
    if session_manager is None:
        raise HTTPException(500, "Server not initialized")

    session = session_manager.get_session(session_id)
    if not session:
        raise HTTPException(404, "Session not found")
        
    theory_data = await session_manager.get_theory(session_id)
    if not theory_data:
        raise HTTPException(404, "Theory not available for this concept")
        
    return TheoryResponse(**theory_data)


@router.post("/{session_id}/exercise", response_model=ExerciseResponse)
async def generate_exercise(session_id: str):
    if session_manager is None:
        raise HTTPException(500, "Server not initialized")

    session = session_manager.get_session(session_id)
    if not session:
        raise HTTPException(404, "Session not found")
    if session.status != "active":
        raise HTTPException(400, "Session is completed")

    exercise = await session_manager.generate_exercise(session_id)
    if not exercise:
        raise HTTPException(500, "Failed to generate exercise")

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
async def submit_answer(session_id: str, req: SubmitAnswerRequest):
    if session_manager is None:
        raise HTTPException(500, "Server not initialized")

    result = await session_manager.submit_answer(session_id, req.answer)
    if not result:
        raise HTTPException(400, "No pending exercise or session not found")

    return SubmitAnswerResponse(**result)


@router.get("/{session_id}/status", response_model=SessionStatusResponse)
async def session_status(session_id: str):
    if session_manager is None:
        raise HTTPException(500, "Server not initialized")

    status = session_manager.get_session_status(session_id)
    if not status:
        raise HTTPException(404, "Session not found")

    return SessionStatusResponse(**status)
