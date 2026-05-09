"""Request context middleware — attaches a unique ID and logs every request."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING
import uuid

from loguru import logger
from starlette.middleware.base import BaseHTTPMiddleware

from api.rate_limit import reset_current_rate_limit_request, set_current_rate_limit_request

if TYPE_CHECKING:
    from fastapi import Request


class RequestContextMiddleware(BaseHTTPMiddleware):
    """Attach X-Request-ID to every request and emit a structured access log line."""

    async def dispatch(self, request: Request, call_next):
        request_id = request.headers.get("x-request-id") or uuid.uuid4().hex
        request.state.request_id = request_id
        start = time.perf_counter()
        rate_limit_token = set_current_rate_limit_request(request)

        try:
            with logger.contextualize(request_id=request_id):
                response = await call_next(request)
                duration_ms = (time.perf_counter() - start) * 1000
                logger.info(
                    "request method={} path={} status={} duration_ms={:.1f}",
                    request.method,
                    request.url.path,
                    response.status_code,
                    duration_ms,
                )
                response.headers["x-request-id"] = request_id
                return response
        finally:
            reset_current_rate_limit_request(rate_limit_token)
