"""
exceptions.py — Domain exceptions and global FastAPI exception handlers.
"""

from fastapi import HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from loguru import logger

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


def _build_error_response(code: str, message: str, detail: str | None = None, meta: list | dict | None = None) -> dict:
    return {
        "success": False,
        "error": {
            "code": code,
            "message": message,
            "detail": detail,
            "meta": meta,
        },
    }

async def app_error_handler(_request: Request, exc: AppError) -> JSONResponse:
    """Map domain exceptions to JSON error responses."""
    code = exc.__class__.__name__
    return JSONResponse(
        status_code=exc.status_code,
        content=_build_error_response(code, "Application error", exc.detail),
    )


async def http_exception_handler(_request: Request, exc: HTTPException) -> JSONResponse:
    """Normalize FastAPI HTTPException responses into a stable JSON envelope."""
    detail_str = exc.detail if isinstance(exc.detail, str) else str(exc.detail)
    meta = exc.detail if not isinstance(exc.detail, str) else None

    return JSONResponse(
        status_code=exc.status_code,
        content=_build_error_response("HTTPException", "HTTP error occurred", detail_str, meta),
        headers=exc.headers,
    )


async def validation_exception_handler(
    request: Request,
    exc: RequestValidationError,
) -> JSONResponse:
    """Normalize request validation failures into a stable JSON envelope."""
    logger.warning(
        "[ValidationError] method={} path={} errors={}",
        request.method,
        request.url.path,
        exc.errors(),
    )
    safe_errors = [
        {"loc": e.get("loc"), "msg": e.get("msg"), "type": e.get("type")} for e in exc.errors()
    ]
    return JSONResponse(
        status_code=422,
        content=_build_error_response("ValidationError", "Invalid request body", str(exc), safe_errors),
    )


async def unexpected_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Catch unexpected errors and avoid leaking internal details to clients."""
    logger.exception(
        "[UnhandledError] method={} path={} error_type={}",
        request.method,
        request.url.path,
        exc.__class__.__name__,
    )
    return JSONResponse(
        status_code=500,
        content=_build_error_response("InternalServerError", "Internal server error", None),
    )


def register_exception_handlers(app) -> None:
    """Register all custom exception handlers on the FastAPI app."""
    app.add_exception_handler(AppError, app_error_handler)
    app.add_exception_handler(HTTPException, http_exception_handler)
    app.add_exception_handler(RequestValidationError, validation_exception_handler)
    app.add_exception_handler(Exception, unexpected_exception_handler)
