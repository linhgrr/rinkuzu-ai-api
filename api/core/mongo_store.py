"""
mongo_store.py — MongoDB persistence layer for adaptive learning sessions and pipeline jobs.

Collections:
  - al_sessions      : full session state snapshot (saved on every answer submission)
  - al_pipeline_jobs : completed pipeline job results (concepts, graph, prereq edges)

Usage:
    from .mongo_store import MongoStore
    store = MongoStore()             # call once at startup via init()
    await store.init()
    await store.save_session(session_state)
    session = await store.load_session(session_id)
"""

import os
import time
import logging
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

_mongo_available = False
_client = None
_db = None


async def init_mongo() -> bool:
    """Connect to MongoDB. Returns True if successful, False otherwise."""
    global _mongo_available, _client, _db

    mongo_url = os.getenv("MONGO_URL")
    if not mongo_url:
        logger.warning("[MongoDB] MONGO_URL not set — persistence disabled")
        return False

    try:
        import motor.motor_asyncio
        _client = motor.motor_asyncio.AsyncIOMotorClient(mongo_url, serverSelectionTimeoutMS=5000)
        # Verify connectivity
        await _client.admin.command("ping")
        _db = _client["adaptive_learning"]
        _mongo_available = True

        # Create indexes
        await _db["al_sessions"].create_index("session_id", unique=True)
        await _db["al_sessions"].create_index("job_id")  # for finding sessions by subject
        await _db["al_pipeline_jobs"].create_index("job_id", unique=True)
        logger.info("[MongoDB] ✓ Connected to adaptive_learning database")
        print("[MongoDB] ✓ Connected to adaptive_learning database")
        return True
    except Exception as e:
        logger.warning(f"[MongoDB] ✗ Could not connect: {e} — persistence disabled")
        print(f"[MongoDB] ✗ Could not connect: {e} — persistence disabled")
        _mongo_available = False
        return False


def is_available() -> bool:
    return _mongo_available


# ---------------------------------------------------------------------------
# Session persistence
# ---------------------------------------------------------------------------

async def save_session(session) -> bool:
    """
    Persist a SessionState snapshot to MongoDB.
    Saves: session_id, concept_map, mastery (per concept×bloom),
           exercise_history, stats, status, timestamps.
    """
    if not _mongo_available:
        return False
    try:
        import numpy as np
        env = session.env
        bloom_mastery = env.get_mastery_matrix()   # shape (n_concepts, 6)
        concept_mastery = env.get_concept_mastery()  # shape (n_concepts,)

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
            for ex in session.exercise_history
        ]

        env_stats = env.get_session_stats()

        doc = {
            "session_id": session.session_id,
            "job_id": getattr(session, "job_id", None),
            "status": session.status,
            "total_correct": session.total_correct,
            "total_answered": session.total_answered,
            "accuracy": session.total_correct / max(session.total_answered, 1),
            "step": env_stats.get("step", 0),
            "max_steps": env_stats.get("max_steps", 50),
            "avg_mastery": float(np.mean(concept_mastery)),
            # Mastery vectors (stored as lists for BSON compatibility)
            "concept_mastery": concept_mastery.tolist(),
            "bloom_mastery": bloom_mastery.tolist(),
            # Concept metadata
            "concept_names": session.concept_names,
            # History
            "exercise_history": history,
            # Timestamps
            "created_at": session.created_at,
            "updated_at": time.time(),
        }

        await _db["al_sessions"].update_one(
            {"session_id": session.session_id},
            {"$set": doc},
            upsert=True,
        )
        return True
    except Exception as e:
        logger.error(f"[MongoDB] save_session error: {e}")
        return False


async def load_session_doc(session_id: str) -> Optional[Dict[str, Any]]:
    """Load a raw session document from MongoDB (for display / resume)."""
    if not _mongo_available:
        return None
    try:
        doc = await _db["al_sessions"].find_one(
            {"session_id": session_id}, {"_id": 0}
        )
        return doc
    except Exception as e:
        logger.error(f"[MongoDB] load_session_doc error: {e}")
        return None


async def list_sessions(limit: int = 50) -> list:
    """List recent sessions (summary only)."""
    if not _mongo_available:
        return []
    try:
        cursor = _db["al_sessions"].find(
            {},
            {
                "_id": 0,
                "session_id": 1,
                "status": 1,
                "total_correct": 1,
                "total_answered": 1,
                "accuracy": 1,
                "avg_mastery": 1,
                "step": 1,
                "max_steps": 1,
                "updated_at": 1,
                "created_at": 1,
            },
        ).sort("updated_at", -1).limit(limit)
        return await cursor.to_list(length=limit)
    except Exception as e:
        logger.error(f"[MongoDB] list_sessions error: {e}")
        return []


async def find_latest_session_for_job(job_id: str) -> Optional[Dict[str, Any]]:
    """Find the most recent session for a given pipeline job_id.
    
    This enables resuming learning from where the student left off.
    Returns the full session document including exercise_history.
    """
    if not _mongo_available:
        return None
    try:
        doc = await _db["al_sessions"].find_one(
            {"job_id": job_id},
            {"_id": 0},
            sort=[("updated_at", -1)],
        )
        return doc
    except Exception as e:
        logger.error(f"[MongoDB] find_latest_session_for_job error: {e}")
        return None


# ---------------------------------------------------------------------------
# Pipeline job persistence
# ---------------------------------------------------------------------------

async def save_pipeline_job(job) -> bool:
    """Persist a completed PipelineJob's result to MongoDB."""
    if not _mongo_available:
        return False
    try:
        doc = {
            "job_id": job.job_id,
            "filename": job.filename,
            "subject_id": job.subject_id,
            "status": job.status.value,
            "total_chunks": job.total_chunks,
            "concepts_extracted": job.concepts_extracted,
            "concepts_after_merge": job.concepts_after_merge,
            "relations_verified": job.relations_verified,
            "graph_stats": job.graph_stats if isinstance(job.graph_stats, dict) else {},
            "result": job.result,
            "created_at": job.created_at,
            "completed_at": job.completed_at or time.time(),
        }
        await _db["al_pipeline_jobs"].update_one(
            {"job_id": job.job_id},
            {"$set": doc},
            upsert=True,
        )
        print(f"[MongoDB] ✓ Pipeline job {job.job_id} saved to DB")
        return True
    except Exception as e:
        logger.error(f"[MongoDB] save_pipeline_job error: {e}")
        return False


async def load_pipeline_job(job_id: str) -> Optional[Dict[str, Any]]:
    """Load a pipeline job result from MongoDB."""
    if not _mongo_available:
        return None
    try:
        doc = await _db["al_pipeline_jobs"].find_one({"job_id": job_id}, {"_id": 0})
        return doc
    except Exception as e:
        logger.error(f"[MongoDB] load_pipeline_job error: {e}")
        return None


async def list_pipeline_jobs(limit: int = 20) -> list:
    """List recent pipeline jobs."""
    if not _mongo_available:
        return []
    try:
        cursor = _db["al_pipeline_jobs"].find(
            {},
            {
                "_id": 0,
                "job_id": 1,
                "filename": 1,
                "subject_id": 1,
                "status": 1,
                "concepts_after_merge": 1,
                "relations_verified": 1,
                "completed_at": 1,
            },
        ).sort("completed_at", -1).limit(limit)
        return await cursor.to_list(length=limit)
    except Exception as e:
        logger.error(f"[MongoDB] list_pipeline_jobs error: {e}")
        return []


async def delete_pipeline_job(job_id: str, delete_sessions: bool = True) -> Dict[str, Any]:
    """
    Delete a pipeline job by job_id.

    Optionally deletes all adaptive learning sessions tied to this job_id.
    """
    if not _mongo_available:
        return {"deleted_job": 0, "deleted_sessions": 0}

    try:
        job_result = await _db["al_pipeline_jobs"].delete_one({"job_id": job_id})
        deleted_sessions = 0

        if delete_sessions:
            session_result = await _db["al_sessions"].delete_many({"job_id": job_id})
            deleted_sessions = session_result.deleted_count

        return {
            "deleted_job": int(job_result.deleted_count),
            "deleted_sessions": int(deleted_sessions),
        }
    except Exception as e:
        logger.error(f"[MongoDB] delete_pipeline_job error: {e}")
        return {"deleted_job": 0, "deleted_sessions": 0, "error": str(e)}
