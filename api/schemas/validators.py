"""Shared FastAPI path-parameter validators.

Rejects path segments with characters outside the allowed set to
guard against path traversal and injection via URL parameters.
"""

from typing import Annotated

from fastapi import Path

_ID_PATTERN = r"^[a-zA-Z0-9_\-]+$"
_EXERCISE_CONTEXT_ID_PATTERN = r"^[a-zA-Z0-9_:\-]+$"

PathID = Annotated[
    str,
    Path(
        min_length=1,
        max_length=64,
        pattern=_ID_PATTERN,
        description="Resource identifier (alphanumeric, hyphens, underscores)",
    ),
]

ExerciseContextPathID = Annotated[
    str,
    Path(
        min_length=1,
        max_length=160,
        pattern=_EXERCISE_CONTEXT_ID_PATTERN,
        description="Namespaced exercise context identifier",
    ),
]
