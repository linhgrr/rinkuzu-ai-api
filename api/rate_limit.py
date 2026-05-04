"""Shared API rate-limit primitives.

Keep this outside `api.main` so routers can use decorators without importing the
application entrypoint and creating circular imports during tests.
"""

from fastapi import Request
from slowapi import Limiter
from slowapi.util import get_remote_address


def rate_limit_key(request: Request) -> str:
    """Prefer authenticated user IDs, then fall back to client IP."""
    return request.headers.get("x-user-id") or get_remote_address(request)


limiter = Limiter(key_func=rate_limit_key)
