"""
history.py — Endpoints for querying persisted sessions and pipeline jobs from MongoDB.
"""

from fastapi import APIRouter, HTTPException
from ..core import mongo_store

router = APIRouter(prefix="/api/history", tags=["history"])


@router.get("/subjects")
async def list_subjects(limit: int = 100):
    """
    List all completed pipeline jobs (= subjects) from MongoDB.
    Used by the Dashboard to show the 'My Subjects' grid.
    """
    jobs = await mongo_store.list_pipeline_jobs(limit=limit)
    subjects = [j for j in jobs if j.get("status") == "completed"]
    return {"subjects": subjects, "count": len(subjects)}


@router.get("/sessions")
async def list_sessions(limit: int = 50):
    """List recent adaptive learning sessions stored in MongoDB."""
    sessions = await mongo_store.list_sessions(limit=limit)
    return {"sessions": sessions, "count": len(sessions)}


@router.get("/sessions/{session_id}")
async def get_session_history(session_id: str):
    """
    Get full persisted state for a session from MongoDB.
    Includes mastery matrix, bloom mastery per concept, and full exercise history.
    """
    doc = await mongo_store.load_session_doc(session_id)
    if not doc:
        # Fallback to active memory session if not yet persisted (or MongoDB is disabled)
        from .session import session_manager
        if session_manager:
            mem_session = session_manager.get_session(session_id)
            if mem_session:
                import numpy as np
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
                    "concept_names": list(mem_session.concept_names.values()), # Ensure array-like if type asks for string[]
                    "exercise_history": history,
                    "created_at": mem_session.created_at,
                    "updated_at": mem_session.created_at,
                }

        raise HTTPException(
            404,
            detail=f"Session {session_id} not found in MongoDB or Memory.",
        )
    
    # Ensure concept_names is properly formatted for frontend (array of strings vs obj)
    if doc and isinstance(doc.get("concept_names"), dict):
        # We'll just return it as is since frontend will handle or we just provide what it expects.
        # Although type says string[], some data might be dict.
        pass

    return doc


@router.get("/pipeline-jobs")
async def list_pipeline_jobs(limit: int = 20):
    """List recent pipeline jobs stored in MongoDB."""
    jobs = await mongo_store.list_pipeline_jobs(limit=limit)
    return {"jobs": jobs, "count": len(jobs)}


@router.get("/pipeline-jobs/{job_id}")
async def get_pipeline_job(job_id: str):
    """Get full pipeline job result from MongoDB."""
    doc = await mongo_store.load_pipeline_job(job_id)
    if not doc:
        raise HTTPException(404, detail=f"Pipeline job {job_id} not found in MongoDB.")
    return doc


@router.delete("/subjects/{job_id}")
async def delete_subject(job_id: str, delete_sessions: bool = True):
    """
    Delete a subject (pipeline job) from MongoDB.
    Optionally delete all learning sessions linked to this job_id.
    """
    result = await mongo_store.delete_pipeline_job(
        job_id=job_id,
        delete_sessions=delete_sessions,
    )
    if result.get("deleted_job", 0) == 0:
        raise HTTPException(404, detail=f"Subject/job {job_id} not found.")

    return {
        "job_id": job_id,
        "deleted_job": result.get("deleted_job", 0),
        "deleted_sessions": result.get("deleted_sessions", 0),
        "status": "deleted",
    }
