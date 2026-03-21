"""
dependencies.py — FastAPI dependency injection functions.
"""

import os

from fastapi import Request, Header, HTTPException

from .config import Settings, get_settings
from .exceptions import ServiceUnavailableError


def get_current_user(
    x_user_id: str = Header(default=None),
    x_service_token: str = Header(default=None),
):
    """Extract user ID from headers."""
    required_service_token = os.getenv("INTERNAL_SERVICE_TOKEN")
    if required_service_token and x_service_token != required_service_token:
        raise HTTPException(status_code=401, detail="Invalid service token")

    if not x_user_id:
        raise HTTPException(status_code=401, detail="Missing x-user-id header")
    return x_user_id


def get_app_settings() -> Settings:
    """Provide application settings."""
    return get_settings()


def get_session_manager(request: Request):
    """Provide SessionManager from app state, raise 503 if not ready."""
    manager = getattr(request.app.state, "session_manager", None)
    if manager is None:
        raise ServiceUnavailableError("SessionManager")
    return manager


def get_session_service(request: Request):
    """Provide ExerciseService from app state, raise 503 if not ready."""
    service = getattr(request.app.state, "exercise_service", None)
    if service is None:
        raise ServiceUnavailableError("ExerciseService")
    return service


def get_content_pipeline_service(request: Request):
    """Provide PipelineService from app state, raise 503 if not ready."""
    service = getattr(request.app.state, "content_pipeline_service", None)
    if service is None:
        raise ServiceUnavailableError("ContentPipelineService")
    return service


def get_content_pipeline_availability(request: Request) -> dict:
    """Expose runtime availability of the legacy content-processor bindings."""
    return {
        "available": bool(getattr(request.app.state, "content_processor_available", False)),
        "error": getattr(request.app.state, "content_processor_error", None),
        "src": getattr(request.app.state, "content_processor_src", None),
    }
