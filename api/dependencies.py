"""
dependencies.py — FastAPI dependency injection functions.
"""

from fastapi import Request

from .config import Settings, get_settings
from .exceptions import ServiceUnavailableError


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
