from typing import Any

"""Unified content pipeline package.

The public package import stays lightweight. Heavy runtime collaborators
(LLM/vector stores/S3/ML) are imported lazily by `__getattr__` only when callers
explicitly ask for them.
"""

from importlib import import_module

from .domain.jobs import PipelineJob, PipelineStatus

__all__ = [
    "CONTENT_PROCESSOR_AVAILABLE",
    "CONTENT_PROCESSOR_ERROR",
    "CONTENT_PROCESSOR_SRC",
    "PipelineJob",
    "PipelineStatus",
    "calculate_file_hash",
    "get_s3_client",
]


def __getattr__(name: str) -> Any:
    if name in {
        "CONTENT_PROCESSOR_AVAILABLE",
        "CONTENT_PROCESSOR_ERROR",
        "CONTENT_PROCESSOR_SRC",
        "calculate_file_hash",
        "get_s3_client",
    }:
        runtime = import_module(".infrastructure.runtime", __name__)
        return getattr(runtime, name)
    raise AttributeError(name)
