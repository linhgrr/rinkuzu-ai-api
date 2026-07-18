"""Shared authentication helpers for internal service-token checks."""

from __future__ import annotations

import hmac


def service_tokens_match(provided: str | None, expected: str | None) -> bool:
    """Return True when both tokens are present and equal (constant-time).

    Fail closed: None or empty on either side is a mismatch. Both values are
    encoded as UTF-8 bytes and compared only when lengths match so
    ``hmac.compare_digest`` never receives unequal inputs. Token values are
    never raised or logged by this helper.
    """
    if not provided or not expected:
        return False
    provided_bytes = provided.encode("utf-8")
    expected_bytes = expected.encode("utf-8")
    if len(provided_bytes) != len(expected_bytes):
        return False
    return hmac.compare_digest(provided_bytes, expected_bytes)
