"""Serialization helpers for content pipeline persistence."""

from __future__ import annotations

from typing import Any

from api.core.shared.persistence.common import normalize_for_bson
from api.core.shared.persistence.pipeline_jobs import pipeline_job_to_document

__all__ = ["normalize_for_bson", "pipeline_job_to_document"]


def _to_bson_safe(value: Any) -> Any:
    """Backward-compatible alias for legacy imports/tests."""
    return normalize_for_bson(value)
