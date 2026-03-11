"""
mongo_store.py — MongoDB connection management and backward-compatible API.

Wraps the repository classes and provides module-level functions
for backward compatibility with content_pipeline.py.
"""

from typing import Optional, Dict, Any

from loguru import logger

from ..repositories.session_repo import SessionRepository
from ..repositories.pipeline_repo import PipelineRepository


_mongo_available = False
_session_repo: Optional[SessionRepository] = None
_pipeline_repo: Optional[PipelineRepository] = None


async def init_mongo(mongo_url: Optional[str] = None) -> bool:
    """Connect to MongoDB and initialize repositories.

    Returns True if successful, False otherwise.
    """
    global _mongo_available, _session_repo, _pipeline_repo

    if not mongo_url:
        import os
        mongo_url = os.getenv("MONGO_URL")

    if not mongo_url:
        logger.warning("[MongoDB] MONGO_URL not set — persistence disabled")
        return False

    try:
        import motor.motor_asyncio
        client = motor.motor_asyncio.AsyncIOMotorClient(mongo_url, serverSelectionTimeoutMS=5000)
        await client.admin.command("ping")
        db = client["adaptive_learning"]

        _session_repo = SessionRepository(db)
        _pipeline_repo = PipelineRepository(db)

        await _session_repo.ensure_indexes()
        await _pipeline_repo.ensure_indexes()

        _mongo_available = True
        logger.info("[MongoDB] ✓ Connected to adaptive_learning database")
        return True
    except Exception as e:
        logger.warning(f"[MongoDB] ✗ Could not connect: {e} — persistence disabled")
        _mongo_available = False
        return False


def is_available() -> bool:
    return _mongo_available


def get_session_repo() -> Optional[SessionRepository]:
    return _session_repo


def get_pipeline_repo() -> Optional[PipelineRepository]:
    return _pipeline_repo


# ── Backward-compatible module-level functions ──────────────
# Used by content_pipeline.py which imports from this module directly.

async def save_session(session) -> bool:
    if not _session_repo:
        return False
    return await _session_repo.save(session)


async def load_session_doc(session_id: str) -> Optional[Dict[str, Any]]:
    if not _session_repo:
        return None
    return await _session_repo.load(session_id)


async def list_sessions(limit: int = 50, user_id: str = None) -> list:
    if not _session_repo:
        return []
    return await _session_repo.list_recent(limit, user_id)


async def find_latest_session_for_job(job_id: str) -> Optional[Dict[str, Any]]:
    if not _session_repo:
        return None
    return await _session_repo.find_latest_for_job(job_id)


async def save_pipeline_job(job) -> bool:
    if not _pipeline_repo:
        return False
    return await _pipeline_repo.save(job)


async def load_pipeline_job(job_id: str) -> Optional[Dict[str, Any]]:
    if not _pipeline_repo:
        return None
    return await _pipeline_repo.load(job_id)


async def list_pipeline_jobs(limit: int = 20, user_id: str = None) -> list:
    if not _pipeline_repo:
        return []
    return await _pipeline_repo.list_recent(limit, user_id)


async def delete_pipeline_job(job_id: str, delete_sessions: bool = True) -> Dict[str, Any]:
    if not _pipeline_repo:
        return {"deleted_job": 0, "deleted_sessions": 0}
    return await _pipeline_repo.delete(job_id, delete_sessions)
