"""
mongo_store.py — MongoDB connection management.

Centralizes MongoDB client lifecycle and provides repository accessors
for direct use by application code.
"""

from typing import Any, TypedDict

from loguru import logger

from api.config import get_settings
from api.repositories.pipeline_repo import PipelineRepository
from api.repositories.quiz_draft_repo import QuizDraftRepository
from api.repositories.subject_progress_repo import SubjectProgressRepository

try:
    import motor.motor_asyncio as _motor

    _MOTOR_AVAILABLE = True
except ImportError:
    _motor = None  # type: ignore[assignment]
    _MOTOR_AVAILABLE = False

class _MongoState(TypedDict):
    available: bool
    pipeline_repo: PipelineRepository | None
    quiz_draft_repo: QuizDraftRepository | None
    subject_progress_repo: SubjectProgressRepository | None
    client: Any | None


# Module-level state stored in a dict to avoid `global` statements.
_state: _MongoState = {
    "available": False,
    "pipeline_repo": None,
    "quiz_draft_repo": None,
    "subject_progress_repo": None,
    "client": None,
}


async def init_mongo(mongodb_uri: str | None = None) -> bool:
    """Connect to MongoDB and initialize repositories.

    Returns True if successful, False otherwise.
    """
    if not mongodb_uri:
        mongodb_uri = get_settings().mongodb_uri

    if not mongodb_uri:
        logger.warning("[MongoDB] MONGODB_URI not set — persistence disabled")
        return False

    if not _MOTOR_AVAILABLE or _motor is None:
        logger.warning("[MongoDB] motor package not available — persistence disabled")
        return False

    try:
        client: Any = _motor.AsyncIOMotorClient(
            mongodb_uri,
            serverSelectionTimeoutMS=5000,
            connectTimeoutMS=5000,
        )
        await client.admin.command("ping")
        db = client["adaptive_learning"]
        _state["client"] = client

        pipeline_repo = PipelineRepository(db)
        quiz_draft_repo = QuizDraftRepository(db)
        subject_progress_repo = SubjectProgressRepository(db)

        await pipeline_repo.ensure_indexes()
        await quiz_draft_repo.ensure_indexes()
        await subject_progress_repo.ensure_indexes()

        _state["pipeline_repo"] = pipeline_repo
        _state["quiz_draft_repo"] = quiz_draft_repo
        _state["subject_progress_repo"] = subject_progress_repo
        _state["available"] = True
        logger.info("[MongoDB] ✓ Connected to adaptive_learning database")
    except Exception:
        logger.exception("[MongoDB] ✗ Could not connect — persistence disabled")
        _state["available"] = False
        return False
    else:
        return True


def is_available() -> bool:
    return bool(_state["available"])


def _get_db() -> Any | None:
    """Return the adaptive_learning database if MongoDB is connected, else None."""
    client = _state["client"]
    if client is None:
        return None
    return client["adaptive_learning"]


def get_pipeline_repo() -> PipelineRepository | None:
    return _state["pipeline_repo"]


def get_subject_progress_repo() -> SubjectProgressRepository | None:
    return _state["subject_progress_repo"]


def get_quiz_draft_repo() -> QuizDraftRepository | None:
    return _state["quiz_draft_repo"]


def require_pipeline_repo() -> PipelineRepository:
    repo = _state["pipeline_repo"]
    if repo is None:
        raise RuntimeError("Pipeline repository not initialized — MongoDB may be unavailable")
    return repo


def require_subject_progress_repo() -> SubjectProgressRepository:
    repo = _state["subject_progress_repo"]
    if repo is None:
        raise RuntimeError("Subject progress repository not initialized — MongoDB may be unavailable")
    return repo


def require_quiz_draft_repo() -> QuizDraftRepository:
    repo = _state["quiz_draft_repo"]
    if repo is None:
        raise RuntimeError("Quiz draft repository not initialized — MongoDB may be unavailable")
    return repo
