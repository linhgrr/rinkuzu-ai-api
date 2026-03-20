"""
mongo_store.py — MongoDB connection management and backward-compatible API.

Wraps the repository classes and provides module-level functions
for backward compatibility with content_pipeline.py.
"""

from typing import Optional, Dict, Any

from loguru import logger

from ..repositories.pipeline_repo import PipelineRepository
from ..repositories.subject_progress_repo import SubjectProgressRepository


_mongo_available = False
_pipeline_repo: Optional[PipelineRepository] = None
_subject_progress_repo: Optional[SubjectProgressRepository] = None


async def init_mongo(mongo_url: Optional[str] = None) -> bool:
    """Connect to MongoDB and initialize repositories.

    Returns True if successful, False otherwise.
    """
    global _mongo_available, _pipeline_repo, _subject_progress_repo

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

        _pipeline_repo = PipelineRepository(db)
        _subject_progress_repo = SubjectProgressRepository(db)

        await _pipeline_repo.ensure_indexes()
        await _subject_progress_repo.ensure_indexes()

        _mongo_available = True
        logger.info("[MongoDB] ✓ Connected to adaptive_learning database")
        return True
    except Exception as e:
        logger.warning(f"[MongoDB] ✗ Could not connect: {e} — persistence disabled")
        _mongo_available = False
        return False


def is_available() -> bool:
    return _mongo_available


def get_pipeline_repo() -> Optional[PipelineRepository]:
    return _pipeline_repo


def get_subject_progress_repo() -> Optional[SubjectProgressRepository]:
    return _subject_progress_repo


# ── Backward-compatible module-level functions ──────────────
# Used by content_pipeline.py which imports from this module directly.

async def save_subject_progress(job_id: str, user_id: str, doc: Dict[str, Any]) -> bool:
    if not _subject_progress_repo:
        return False
    return await _subject_progress_repo.save_snapshot(job_id, user_id, doc)


async def load_subject_progress_for_user(job_id: str, user_id: str) -> Optional[Dict[str, Any]]:
    if not _subject_progress_repo:
        return None
    return await _subject_progress_repo.load_for_user(job_id, user_id)


async def load_subject_progress_by_session_for_user(
    session_id: str,
    user_id: str,
) -> Optional[Dict[str, Any]]:
    if not _subject_progress_repo:
        return None
    return await _subject_progress_repo.load_by_session_for_user(session_id, user_id)


async def load_subject_progress_map(job_ids: list[str], user_id: str) -> Dict[str, Dict[str, Any]]:
    if not _subject_progress_repo:
        return {}
    return await _subject_progress_repo.load_many_for_user(job_ids, user_id)


async def list_subject_progress(limit: int = 50, user_id: str = None) -> list:
    if not _subject_progress_repo:
        return []
    return await _subject_progress_repo.list_recent(limit, user_id)


async def delete_subject_progress_for_user(job_id: str, user_id: str) -> int:
    if not _subject_progress_repo:
        return 0
    return await _subject_progress_repo.delete_for_user(job_id, user_id)


async def save_pipeline_job(job) -> bool:
    if not _pipeline_repo:
        return False
    return await _pipeline_repo.save(job)


async def load_pipeline_job(job_id: str) -> Optional[Dict[str, Any]]:
    if not _pipeline_repo:
        return None
    return await _pipeline_repo.load(job_id)


async def load_pipeline_job_for_user(job_id: str, user_id: str) -> Optional[Dict[str, Any]]:
    if not _pipeline_repo:
        return None
    return await _pipeline_repo.load_for_user(job_id, user_id)


async def load_pipeline_job_map_for_user(
    job_ids: list[str],
    user_id: str,
    projection: Optional[Dict[str, int]] = None,
) -> Dict[str, Dict[str, Any]]:
    if not _pipeline_repo:
        return {}
    return await _pipeline_repo.load_many_for_user(job_ids, user_id, projection=projection)


async def list_pipeline_jobs(limit: int = 20, user_id: str = None, status: str = None) -> list:
    if not _pipeline_repo:
        return []
    return await _pipeline_repo.list_recent(limit=limit, user_id=user_id, status=status)


async def delete_pipeline_job(job_id: str, delete_sessions: bool = True) -> Dict[str, Any]:
    if not _pipeline_repo:
        return {"deleted_job": 0, "deleted_sessions": 0}
    return await _pipeline_repo.delete(job_id, delete_sessions)


async def delete_pipeline_job_for_user(
    job_id: str,
    user_id: str,
    delete_sessions: bool = True,
) -> Dict[str, Any]:
    if not _pipeline_repo:
        return {"deleted_job": 0, "deleted_sessions": 0}
    return await _pipeline_repo.delete_for_user(job_id, user_id, delete_sessions)
