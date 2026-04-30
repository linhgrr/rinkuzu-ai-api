"""Unified content pipeline package.

This package is the landing zone for merging the legacy content pipeline
service into the main backend. It intentionally preserves the old import
surface of `api.core.content_pipeline` while allowing internal modules to move
into domain/application/infrastructure/interfaces layers.
"""

from .domain.jobs import PipelineJob, PipelineStatus
from .infrastructure.runtime import (
    CONTENT_PROCESSOR_AVAILABLE,
    CONTENT_PROCESSOR_ERROR,
    CONTENT_PROCESSOR_SRC,
    calculate_file_hash,
    get_s3_client,
)
from .orchestrator import process_pdf

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
