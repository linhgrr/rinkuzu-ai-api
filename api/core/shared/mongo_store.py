"""
mongo_store.py — MongoDB connection management and backward-compatible API.

Wraps the repository classes and provides module-level functions
for backward compatibility with content_pipeline.py.
"""

from typing import Any

from loguru import logger

from api.config import get_settings
from api.repositories.pipeline_repo import PipelineRepository
from api.repositories.subject_progress_repo import SubjectProgressRepository

try:
    import motor.motor_asyncio as _motor

    _MOTOR_AVAILABLE = True
except ImportError:
    _motor = None  # type: ignore[assignment]
    _MOTOR_AVAILABLE = False

# Module-level state stored in a dict to avoid `global` statements.
_state: dict[str, Any] = {
    "available": False,
    "pipeline_repo": None,
    "subject_progress_repo": None,
    "client": None,
}


async def init_mongo(mongo_url: str | None = None) -> bool:
    """Connect to MongoDB and initialize repositories.

    Returns True if successful, False otherwise.
    """
    if not mongo_url:
        mongo_url = get_settings().mongo_url

    if not mongo_url:
        logger.warning("[MongoDB] MONGO_URL not set — persistence disabled")
        return False

    if not _MOTOR_AVAILABLE or _motor is None:
        logger.warning("[MongoDB] motor package not available — persistence disabled")
        return False

    try:
        client = _motor.AsyncIOMotorClient(mongo_url, serverSelectionTimeoutMS=5000)
        await client.admin.command("ping")
        db = client["adaptive_learning"]
        _state["client"] = client

        pipeline_repo = PipelineRepository(db)
        subject_progress_repo = SubjectProgressRepository(db)

        await pipeline_repo.ensure_indexes()
        await subject_progress_repo.ensure_indexes()

        _state["pipeline_repo"] = pipeline_repo
        _state["subject_progress_repo"] = subject_progress_repo
        _state["available"] = True
        logger.info("[MongoDB] ✓ Connected to adaptive_learning database")
    except Exception as e:
        logger.warning(f"[MongoDB] ✗ Could not connect: {e} — persistence disabled")
        _state["available"] = False
        return False
    else:
        return True


def is_available() -> bool:
    return bool(_state["available"])


def _get_db():
    """Return the adaptive_learning database if MongoDB is connected, else None."""
    client = _state["client"]
    if client is None:
        return None
    return client["adaptive_learning"]


def get_pipeline_repo() -> PipelineRepository | None:
    return _state["pipeline_repo"]


def get_subject_progress_repo() -> SubjectProgressRepository | None:
    return _state["subject_progress_repo"]


# ── Backward-compatible module-level functions ──────────────
# Used by content_pipeline.py which imports from this module directly.


async def save_subject_progress(job_id: str, user_id: str, doc: dict[str, Any]) -> bool:
    repo = _state["subject_progress_repo"]
    if not repo:
        return False
    return await repo.save_snapshot(job_id, user_id, doc)


async def load_subject_progress_for_user(job_id: str, user_id: str) -> dict[str, Any] | None:
    repo = _state["subject_progress_repo"]
    if not repo:
        return None
    return await repo.load_for_user(job_id, user_id)


async def load_subject_progress_by_session_for_user(
    session_id: str,
    user_id: str,
) -> dict[str, Any] | None:
    repo = _state["subject_progress_repo"]
    if not repo:
        return None
    return await repo.load_by_session_for_user(session_id, user_id)


async def load_session_doc_for_user(
    session_id: str,
    user_id: str,
) -> dict[str, Any] | None:
    """Compatibility alias for persisted adaptive-learning session docs."""
    return await load_subject_progress_by_session_for_user(session_id, user_id)


async def load_subject_progress_for_job(
    job_id: str,
    user_id: str,
) -> dict[str, Any] | None:
    """Load the saved subject-progress snapshot for a user's pipeline job."""
    return await load_subject_progress_for_user(job_id, user_id)


async def find_latest_session_for_job(
    job_id: str,
    user_id: str,
) -> dict[str, Any] | None:
    """Backward-compatible alias for subject-progress loading."""
    return await load_subject_progress_for_job(job_id, user_id)


async def load_subject_progress_map(job_ids: list[str], user_id: str) -> dict[str, dict[str, Any]]:
    repo = _state["subject_progress_repo"]
    if not repo:
        return {}
    return await repo.load_many_for_user(job_ids, user_id)


async def list_subject_progress(limit: int = 50, user_id: str | None = None) -> list:
    repo = _state["subject_progress_repo"]
    if not repo:
        return []
    return await repo.list_recent(limit, user_id)


async def delete_subject_progress_for_user(job_id: str, user_id: str) -> int:
    repo = _state["subject_progress_repo"]
    if not repo:
        return 0
    return await repo.delete_for_user(job_id, user_id)


async def save_pipeline_job(job) -> bool:
    repo = _state["pipeline_repo"]
    if not repo:
        return False
    return await repo.save(job)


async def load_pipeline_job(job_id: str) -> dict[str, Any] | None:
    repo = _state["pipeline_repo"]
    if not repo:
        return None
    return await repo.load(job_id)


async def load_pipeline_job_for_user(job_id: str, user_id: str) -> dict[str, Any] | None:
    repo = _state["pipeline_repo"]
    if not repo:
        return None
    return await repo.load_for_user(job_id, user_id)


async def load_pipeline_job_map_for_user(
    job_ids: list[str],
    user_id: str,
    projection: dict[str, int] | None = None,
) -> dict[str, dict[str, Any]]:
    repo = _state["pipeline_repo"]
    if not repo:
        return {}
    return await repo.load_many_for_user(job_ids, user_id, projection=projection)


async def list_pipeline_jobs(
    limit: int = 20, user_id: str | None = None, status: str | None = None
) -> list:
    repo = _state["pipeline_repo"]
    if not repo:
        return []
    return await repo.list_recent(limit=limit, user_id=user_id, status=status)


async def delete_pipeline_job(job_id: str, *, delete_sessions: bool = True) -> dict[str, Any]:
    repo = _state["pipeline_repo"]
    if not repo:
        return {"deleted_job": 0, "deleted_sessions": 0}
    return await repo.delete(job_id, delete_sessions=delete_sessions)


async def delete_pipeline_job_for_user(
    job_id: str,
    user_id: str,
    *,
    delete_sessions: bool = True,
) -> dict[str, Any]:
    repo = _state["pipeline_repo"]
    if not repo:
        return {"deleted_job": 0, "deleted_sessions": 0}
    return await repo.delete_for_user(job_id, user_id, delete_sessions=delete_sessions)
