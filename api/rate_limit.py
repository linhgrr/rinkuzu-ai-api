"""Shared API rate-limit primitives.

Keep this outside `api.main` so routers can use decorators without importing the
application entrypoint and creating circular imports during tests.
"""

from contextvars import ContextVar, Token

from fastapi import Request
from slowapi import Limiter
from slowapi.util import get_remote_address

from api.config import get_settings

_current_request: ContextVar[Request | None] = ContextVar("rate_limit_request", default=None)


def rate_limit_key(request: Request) -> str:
    """Prefer authenticated user IDs, then fall back to client IP."""
    return request.headers.get("x-user-id") or get_remote_address(request)


def set_current_rate_limit_request(request: Request) -> Token[Request | None]:
    return _current_request.set(request)


def reset_current_rate_limit_request(token: Token[Request | None]) -> None:
    _current_request.reset(token)


def is_admin_request() -> bool:
    """Exempt internal service-to-service calls from rate limits.

    Only exempts requests that present a valid x-service-token matching the
    configured internal_service_token — NOT requests that self-declare an
    admin role via the x-user-role header (which any client can forge).
    """
    settings = get_settings()
    required_token = settings.internal_service_token
    if not required_token:
        return False
    request = _current_request.get()
    if request is None:
        return False
    return request.headers.get("x-service-token") == required_token


limiter = Limiter(key_func=rate_limit_key)
