"""
dependencies.py — FastAPI dependency injection functions.
"""

from fastapi import Header, HTTPException, Request

from .config import Settings, get_settings
from .exceptions import ServiceUnavailableError, SessionNotFoundError


def get_current_user(
    x_user_id: str | None = Header(default=None),
    x_service_token: str | None = Header(default=None),
):
    """Extract user ID from headers."""
    settings = get_settings()
    required_service_token = settings.internal_service_token
    if settings.environment != "dev" and not required_service_token:
        raise HTTPException(status_code=500, detail="Internal service token is not configured")

    if required_service_token and x_service_token != required_service_token:
        raise HTTPException(status_code=401, detail="Invalid service token")

    if not x_user_id:
        raise HTTPException(status_code=401, detail="Missing x-user-id header")
    return x_user_id


def get_app_settings() -> Settings:
    """Provide application settings."""
    return get_settings()


def _resolve_state(request: Request, attr: str, label: str) -> object:
    """Return ``request.app.state.<attr>`` or raise 503 if unavailable."""
    obj = getattr(request.app.state, attr, None)
    if obj is None:
        raise ServiceUnavailableError(label)
    return obj


def get_session_manager(request: Request):
    """Provide SessionManager from app state, raise 503 if not ready."""
    return _resolve_state(request, "session_manager", "SessionManager")


def get_session_service(request: Request):
    """Provide ExerciseService from app state, raise 503 if not ready."""
    return _resolve_state(request, "exercise_service", "ExerciseService")


def get_content_pipeline_service(request: Request):
    """Provide PipelineService from app state, raise 503 if not ready."""
    return _resolve_state(request, "content_pipeline_service", "ContentPipelineService")


async def resolve_user_session(manager, session_id: str, user_id: str):
    """Resolve a session for the authenticated user, raising 404 if not found."""
    session = await manager.get_or_recover_session(session_id, user_id)
    if not session:
        raise SessionNotFoundError(session_id)
    return session


def get_chunk_chroma_store(request: Request):
    """Provide ChunkChromaStore from app state (may be None if unavailable)."""
    return getattr(request.app.state, "chunk_chroma_store", None)


def get_content_pipeline_availability(request: Request) -> dict[str, bool | str | None]:
    """Expose runtime availability of the unified content pipeline modules."""
    return {
        "available": bool(getattr(request.app.state, "content_processor_available", False)),
        "error": getattr(request.app.state, "content_processor_error", None),
        "src": getattr(request.app.state, "content_processor_src", None),
        "service_initialized": getattr(request.app.state, "content_pipeline_service", None)
        is not None,
    }
