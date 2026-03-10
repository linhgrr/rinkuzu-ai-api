"""
exceptions.py — Domain exceptions and global FastAPI exception handlers.
"""

from fastapi import Request
from fastapi.responses import JSONResponse


# ── Domain Exceptions ───────────────────────────────────────

class AppError(Exception):
    """Base exception for application errors."""

    def __init__(self, detail: str, status_code: int = 500):
        self.detail = detail
        self.status_code = status_code
        super().__init__(detail)


class SessionNotFoundError(AppError):
    def __init__(self, session_id: str):
        super().__init__(f"Session {session_id} not found", status_code=404)


class SessionCompletedError(AppError):
    def __init__(self, session_id: str):
        super().__init__(f"Session {session_id} is completed", status_code=400)


class ExerciseGenerationError(AppError):
    def __init__(self, detail: str = "Failed to generate exercise"):
        super().__init__(detail, status_code=500)


class ServiceUnavailableError(AppError):
    def __init__(self, service: str = "Service"):
        super().__init__(f"{service} not initialized", status_code=503)


class PipelineNotFoundError(AppError):
    def __init__(self, job_id: str):
        super().__init__(f"Pipeline job {job_id} not found", status_code=404)


class PipelineNotCompletedError(AppError):
    def __init__(self, job_id: str, status: str):
        super().__init__(
            f"Pipeline job {job_id} is not completed. Status: {status}",
            status_code=400,
        )


# ── Global Exception Handler ───────────────────────────────

async def app_error_handler(_request: Request, exc: AppError) -> JSONResponse:
    """Map domain exceptions to JSON error responses."""
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail},
    )


def register_exception_handlers(app) -> None:
    """Register all custom exception handlers on the FastAPI app."""
    app.add_exception_handler(AppError, app_error_handler)
