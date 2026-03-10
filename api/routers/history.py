"""
history.py — Endpoints for querying persisted sessions and pipeline jobs from MongoDB.
"""

from fastapi import APIRouter, Depends

from ..core import mongo_store
from ..dependencies import get_session_manager
from ..exceptions import SessionNotFoundError, PipelineNotFoundError

import numpy as np

router = APIRouter(prefix="/api/history", tags=["history"])


@router.get("/subjects")
async def list_subjects(limit: int = 100):
    """List all completed pipeline jobs (= subjects)."""
    jobs = await mongo_store.list_pipeline_jobs(limit=limit)
    subjects = [j for j in jobs if j.get("status") == "completed"]
    return {"subjects": subjects, "count": len(subjects)}


@router.get("/sessions")
async def list_sessions(limit: int = 50):
    """List recent adaptive learning sessions."""
    sessions = await mongo_store.list_sessions(limit=limit)
    return {"sessions": sessions, "count": len(sessions)}


@router.get("/sessions/{session_id}")
async def get_session_history(session_id: str, manager=Depends(get_session_manager)):
    """Get full persisted state for a session."""
    doc = await mongo_store.load_session_doc(session_id)
    if not doc:
        # Fallback to active memory session
        mem_session = manager.get_session(session_id)
        if mem_session:
            env = mem_session.env
            bloom_mastery = env.get_mastery_matrix()
            concept_mastery = env.get_concept_mastery()
            env_stats = env.get_session_stats()

            history = [
                {
                    "exercise_id": ex.exercise_id,
                    "concept_idx": ex.concept_idx,
                    "concept_name": ex.concept_name,
                    "bloom_level": ex.bloom_level,
                    "question": ex.question,
                    "options": ex.options,
                    "correct_option": ex.correct_option,
                    "explanation": ex.explanation,
                    "user_answer": ex.user_answer,
                    "is_correct": ex.is_correct,
                    "timestamp": ex.timestamp,
                }
                for ex in mem_session.exercise_history
            ]

            return {
                "session_id": mem_session.session_id,
                "job_id": getattr(mem_session, "job_id", None),
                "status": mem_session.status,
                "total_correct": mem_session.total_correct,
                "total_answered": mem_session.total_answered,
                "accuracy": mem_session.total_correct / max(mem_session.total_answered, 1),
                "step": env_stats.get("step", 0),
                "max_steps": env_stats.get("max_steps", 50),
                "avg_mastery": float(np.mean(concept_mastery)),
                "concept_mastery": concept_mastery.tolist(),
                "bloom_mastery": bloom_mastery.tolist(),
                "concept_names": list(mem_session.concept_names.values()),
                "exercise_history": history,
                "created_at": mem_session.created_at,
                "updated_at": mem_session.created_at,
            }

        raise SessionNotFoundError(session_id)

    return doc


@router.get("/pipeline-jobs")
async def list_pipeline_jobs(limit: int = 20):
    """List recent pipeline jobs."""
    jobs = await mongo_store.list_pipeline_jobs(limit=limit)
    return {"jobs": jobs, "count": len(jobs)}


@router.get("/pipeline-jobs/{job_id}")
async def get_pipeline_job(job_id: str):
    """Get full pipeline job result."""
    doc = await mongo_store.load_pipeline_job(job_id)
    if not doc:
        raise PipelineNotFoundError(job_id)
    return doc


@router.delete("/subjects/{job_id}")
async def delete_subject(job_id: str, delete_sessions: bool = True):
    """Delete a subject (pipeline job) and optionally its sessions."""
    result = await mongo_store.delete_pipeline_job(
        job_id=job_id,
        delete_sessions=delete_sessions,
    )
    if result.get("deleted_job", 0) == 0:
        raise PipelineNotFoundError(job_id)

    return {
        "job_id": job_id,
        "deleted_job": result.get("deleted_job", 0),
        "deleted_sessions": result.get("deleted_sessions", 0),
        "status": "deleted",
    }
