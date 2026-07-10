"""
dependencies.py — FastAPI dependency injection functions.
"""

from typing import TYPE_CHECKING, Any, cast

from fastapi import Header, Request

from .config import Settings, get_settings
from .exceptions import AppError, ServiceUnavailableError, SessionNotFoundError
from .shared.llm_usage import current_user_id

if TYPE_CHECKING:
    from .domains.learning.exercise_service import ExerciseService
    from .domains.learning.session import SessionManager, SessionState


def get_current_user(
    x_user_id: str | None = Header(default=None),
    x_service_token: str | None = Header(default=None),
) -> str:
    """Verify the proxy-issued service token and surface the forwarded user id.

    The Next.js proxy is the only trusted issuer of `x-user-id`. Backend MUST
    always validate `x-service-token` — including in dev — to prevent a
    misconfigured deployment from inadvertently exposing the API.
    """
    settings = get_settings()
    required_service_token = settings.internal_service_token
    if not required_service_token:
        raise AppError(
            code="service_unavailable",
            message="Service unavailable",
            detail="Internal service token is not configured",
            status_code=500,
        )

    if not x_service_token or x_service_token != required_service_token:
        raise AppError(
            code="unauthorized",
            message="Unauthorized",
            detail="Invalid service token",
            status_code=401,
        )

    if not x_user_id:
        raise AppError(
            code="unauthorized",
            message="Unauthorized",
            detail="Missing x-user-id header",
            status_code=401,
        )
    current_user_id.set(x_user_id)
    return x_user_id


def get_current_admin_user(
    x_user_id: str | None = Header(default=None),
    x_user_role: str | None = Header(default=None),
    x_service_token: str | None = Header(default=None),
) -> str:
    user_id = get_current_user(x_user_id=x_user_id, x_service_token=x_service_token)
    if x_user_role != "admin":
        raise AppError(
            code="forbidden",
            message="Admin access required",
            detail="x-user-role must be admin",
            status_code=403,
        )
    return user_id


def get_app_settings() -> Settings:
    """Provide application settings."""
    return get_settings()


def _resolve_state(request: Request, attr: str, label: str) -> object:
    """Return ``request.app.state.<attr>`` or raise 503 if unavailable."""
    obj = getattr(request.app.state, attr, None)
    if obj is None:
        raise ServiceUnavailableError(label)
    return obj


def get_session_manager(request: Request) -> "SessionManager":
    """Provide SessionManager from app state, raise 503 if not ready."""
    return cast("SessionManager", _resolve_state(request, "session_manager", "SessionManager"))


def get_session_service(request: Request) -> "ExerciseService":
    """Provide ExerciseService from app state, raise 503 if not ready."""
    return cast("ExerciseService", _resolve_state(request, "exercise_service", "ExerciseService"))


def get_content_pipeline_service(request: Request) -> Any:
    """Provide PipelineService from app state, raise 503 if not ready."""
    return _resolve_state(request, "content_pipeline_service", "ContentPipelineService")


async def resolve_user_session(
    manager: "SessionManager", session_id: str, user_id: str
) -> "SessionState":
    """Resolve a session for the authenticated user, raising 404 if not found."""
    session = await manager.get_or_recover_session(session_id, user_id)
    if not session:
        raise SessionNotFoundError(session_id)
    return session


def get_chunk_chroma_store(request: Request) -> Any:
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
