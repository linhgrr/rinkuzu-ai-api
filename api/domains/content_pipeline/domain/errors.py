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


class PipelineCacheRebuildError(PipelineExecutionError):
    """Raised when an S3 result cache cannot be made usable for retrieval."""


class PipelineQualityGateError(PipelineExecutionError):
    """Raised when the completed graph is not reliable enough to publish."""

    def __init__(self, message: str, report: PipelineQualityReport):
        self.report = report
        super().__init__(message)


class PipelineStaleWorkerError(PipelineExecutionError):
    """Stop a stale worker cleanly without overwriting terminal/newer generation state.

    Raised when a save CAS misses with STALE_GENERATION or ALREADY_TERMINAL.
    Must be caught before generic terminal-failure handling so the job is not
    forced to FAILED.
    """

    def __init__(self, job_id: str, outcome: str):
        self.job_id = job_id
        self.outcome = outcome
        super().__init__(f"Stale pipeline worker stop for job {job_id}: {outcome}")


class PipelineJobIdCollisionError(PipelineExecutionError):
    """Raised after bounded UUID collision retries are exhausted."""


class PipelineSchedulingUnavailableError(PipelineExecutionError):
    """A retry/recovery job cannot be scheduled during runtime shutdown."""


class PipelineSourceDownloadError(PipelineExecutionError):
    """A source could not be downloaded because its provider is unavailable."""


class PipelineInvalidSourceError(PipelineExecutionError):
    """A persisted source violates the pipeline's PDF invariant."""
