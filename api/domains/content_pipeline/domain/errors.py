"""Domain error types for the content pipeline."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .relations import PipelineQualityReport


class PipelineExecutionError(Exception):
    """Base error for pipeline execution failures."""


class PipelineStageTimeoutError(PipelineExecutionError):
    """Raised when a blocking pipeline stage exceeds its configured timeout."""

    def __init__(self, stage_name: str, timeout_sec: float):
        self.stage_name = stage_name
        self.timeout_sec = timeout_sec
        super().__init__(f"Stage '{stage_name}' exceeded timeout after {timeout_sec:.0f} seconds")


class PipelineQualityGateError(PipelineExecutionError):
    """Raised when the completed graph is not reliable enough to publish."""

    def __init__(self, message: str, report: PipelineQualityReport):
        self.report = report
        super().__init__(message)
