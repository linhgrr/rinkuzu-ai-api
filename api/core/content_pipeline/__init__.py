"""Unified content pipeline package.

This package is the landing zone for merging the legacy content pipeline
service into the main backend. It intentionally preserves the old import
surface of `api.core.content_pipeline` while allowing internal modules to move
into domain/application/infrastructure/interfaces layers.
"""

from .domain.jobs import PipelineJob, PipelineStatus

__all__ = [
    "CONTENT_PROCESSOR_AVAILABLE",
    "CONTENT_PROCESSOR_ERROR",
    "CONTENT_PROCESSOR_SRC",
    "PipelineJob",
    "PipelineStatus",
    "calculate_file_hash",
    "get_s3_client",
    "process_pdf",
]


def __getattr__(name: str):
    if name in {
        "CONTENT_PROCESSOR_AVAILABLE",
        "CONTENT_PROCESSOR_ERROR",
        "CONTENT_PROCESSOR_SRC",
        "calculate_file_hash",
        "get_s3_client",
        "process_pdf",
    }:
        from . import orchestrator

        return getattr(orchestrator, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
