"""Backward-compatible shim for legacy root imports."""

from api.core.content_pipeline.infrastructure.utils import (  # noqa: F401
    clean_text,
    get_file_type,
    guess_mime_type,
    timeit,
)

__all__ = ["timeit", "clean_text", "guess_mime_type", "get_file_type"]
