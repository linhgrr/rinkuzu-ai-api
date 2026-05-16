"""
exceptions.py — Domain exceptions and global FastAPI exception handlers.
"""

from collections.abc import Mapping
from typing import Any, cast

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from loguru import logger
from slowapi.errors import RateLimitExceeded

from api.error_codes import (
    get_default_api_error_message,
    get_http_error_code,
    get_http_error_message,
)
from api.schemas.common import ErrorDetail, StandardErrorResponse

ErrorMeta = dict[str, object] | list[dict[str, object]] | None

HTTP_BAD_REQUEST = 400
HTTP_UNAUTHORIZED = 401
HTTP_FORBIDDEN = 403
HTTP_NOT_FOUND = 404
HTTP_CONFLICT = 409
HTTP_UNPROCESSABLE_ENTITY = 422
HTTP_TOO_MANY_REQUESTS = 429
HTTP_INTERNAL_SERVER_ERROR = 500
HTTP_SERVICE_UNAVAILABLE = 503


class AppError(Exception):
    """Base exception for application errors."""

    def __init__(
        self,
        *,
        code: str,
        message: str,
        detail: str | None = None,
        status_code: int = 500,
        meta: ErrorMeta = None,
    ):
        self.code = code
        self.message = message
        self.detail = detail
        self.status_code = status_code
        self.meta = meta
        super().__init__(detail or message)


class SessionNotFoundError(AppError):
    def __init__(self, session_id: str):
        super().__init__(
            code="not_found",
            message="Session not found",
            detail=f"Session {session_id} not found",
            status_code=404,
        )


class SessionCompletedError(AppError):
    def __init__(self, session_id: str):
        super().__init__(
            code="conflict",
            message="Session already completed",
            detail=f"Session {session_id} is completed",
            status_code=409,
        )


class ExerciseGenerationError(AppError):
    def __init__(self, detail: str = "Failed to generate exercise"):
        super().__init__(
            code="internal_error",
            message="Exercise generation failed",
            detail=detail,
            status_code=500,
        )


class ServiceUnavailableError(AppError):
    def __init__(self, service: str = "Service"):
        super().__init__(
            code="service_unavailable",
            message=f"{service} unavailable",
            detail=f"{service} not initialized",
            status_code=503,
        )


class PipelineNotFoundError(AppError):
    def __init__(self, job_id: str):
        super().__init__(
            code="pipeline_not_found",
            message="Pipeline job not found",
            detail=f"Pipeline job {job_id} not found",
            status_code=404,
        )


class PipelineNotCompletedError(AppError):
    def __init__(self, job_id: str, status: str):
        super().__init__(
            code="pipeline_not_completed",
            message="Pipeline job is not completed",
            detail=f"Pipeline job {job_id} is not completed. Status: {status}",
            status_code=409,
            meta={"job_id": job_id, "status": status},
        )


def _build_error_body(
    *,
    code: str,
    message: str,
    detail: str | None = None,
    meta: ErrorMeta = None,
) -> dict[str, object]:
    payload = StandardErrorResponse(
        error=ErrorDetail(
            code=code,
            message=message,
            detail=detail,
            meta=meta,
        )
    )
    return cast("dict[str, object]", payload.model_dump(exclude_none=True))


def error_json_response(
    *,
    code: str,
    message: str,
    detail: str | None,
    status_code: int,
    meta: ErrorMeta = None,
    headers: Mapping[str, str] | None = None,
) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content=_build_error_body(code=code, message=message, detail=detail, meta=meta),
        headers=headers,
    )


def _http_error_code(status_code: int) -> str:
    default_code = (
        "upstream_error"
        if HTTP_BAD_REQUEST <= status_code < HTTP_INTERNAL_SERVER_ERROR
        else "internal_error"
    )
    return get_http_error_code(status_code, default_code)


def _http_error_message(status_code: int) -> str:
    default_message = (
        "Request failed"
        if HTTP_BAD_REQUEST <= status_code < HTTP_INTERNAL_SERVER_ERROR
        else "Internal server error"
    )
    return get_http_error_message(status_code, default_message)


async def app_error_handler(_request: Request, exc: AppError) -> JSONResponse:
    """Map domain exceptions to JSON error responses."""
    return error_json_response(
        code=exc.code,
        message=exc.message,
        detail=exc.detail,
        status_code=exc.status_code,
        meta=exc.meta,
    )


async def http_exception_handler(_request: Request, exc: HTTPException) -> JSONResponse:
    """Normalize FastAPI HTTPException responses into the standard JSON envelope."""
    status_code = exc.status_code
    raw_detail: Any = exc.detail
    detail = raw_detail if isinstance(raw_detail, str) else str(raw_detail)
    meta = raw_detail if isinstance(raw_detail, (dict, list)) else None
    return error_json_response(
        code=_http_error_code(status_code),
        message=_http_error_message(status_code),
        detail=detail,
        status_code=status_code,
        meta=meta,
        headers=exc.headers,
    )


async def validation_exception_handler(
    request: Request,
    exc: RequestValidationError,
) -> JSONResponse:
    """Normalize request validation failures into the standard JSON envelope."""
    logger.warning(
        "[ValidationError] method={} path={} errors={}",
        request.method,
        request.url.path,
        exc.errors(),
    )
    safe_errors = [
        {"loc": e.get("loc"), "msg": e.get("msg"), "type": e.get("type")} for e in exc.errors()
    ]
    return error_json_response(
        code="validation_error",
        message="Invalid request",
        detail="Request validation failed",
        status_code=HTTP_UNPROCESSABLE_ENTITY,
        meta=safe_errors,
    )


async def unexpected_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Catch unexpected errors and avoid leaking internal details to clients."""
    logger.exception(
        "[UnhandledError] method={} path={} error_type={}",
        request.method,
        request.url.path,
        exc.__class__.__name__,
    )
    return error_json_response(
        code="internal_error",
        message=get_default_api_error_message("internal_error", "Internal server error"),
        detail=None,
        status_code=HTTP_INTERNAL_SERVER_ERROR,
    )


async def rate_limit_exceeded_handler(request: Request, exc: RateLimitExceeded) -> JSONResponse:
    """Normalize SlowAPI 429 responses into the standard error envelope."""
    detail = str(exc.detail)
    response = error_json_response(
        code="rate_limit_exceeded",
        message="Rate limit exceeded",
        detail=detail,
        status_code=HTTP_TOO_MANY_REQUESTS,
    )
    limiter = getattr(request.app.state, "limiter", None)
    view_limit: Any = getattr(request.state, "view_rate_limit", None)
    if limiter is not None and view_limit is not None:
        response = limiter._inject_headers(response, view_limit)
    return response


def register_exception_handlers(app: FastAPI) -> None:
    """Register all custom exception handlers on the FastAPI app."""
    app.add_exception_handler(AppError, app_error_handler)  # type: ignore[arg-type]
    app.add_exception_handler(HTTPException, http_exception_handler)  # type: ignore[arg-type]
    app.add_exception_handler(RequestValidationError, validation_exception_handler)  # type: ignore[arg-type]
    app.add_exception_handler(Exception, unexpected_exception_handler)
    app.add_exception_handler(RateLimitExceeded, rate_limit_exceeded_handler)  # type: ignore[arg-type]
