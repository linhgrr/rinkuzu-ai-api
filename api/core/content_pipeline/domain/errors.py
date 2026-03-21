"""Domain error types for the content pipeline."""

from __future__ import annotations


class PipelineExecutionError(Exception):
    """Base error for pipeline execution failures."""


class PipelineStageTimeoutError(PipelineExecutionError):
    """Raised when a blocking pipeline stage exceeds its configured timeout."""

    def __init__(self, stage_name: str, timeout_sec: float):
        self.stage_name = stage_name
        self.timeout_sec = timeout_sec
        super().__init__(
            f"Stage '{stage_name}' exceeded timeout after {timeout_sec:.0f} seconds"
        )
