"""Job models for the unified content pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
import time
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .relations import PipelineDebugTraceEntry, PipelineQualityReport


class PipelineProgress:
    """Named progress checkpoints for the content pipeline (0.0 -> 1.0)."""

    INIT = 0.01
    CACHE_RESTORE = 0.02
    PDF_LOADED = 0.05
    PDF_CHUNKED = 0.10
    CHUNKS_PERSISTING = 0.11
    CHUNKS_PERSISTED = 0.12
    CONCEPT_EXTRACTION_START = 0.15
    CONCEPT_EXTRACTION_DONE = 0.30
    EMBEDDING_START = 0.35
    EMBEDDING_DONE = 0.45
    MERGING_START = 0.50
    MERGING_DONE = 0.55
    RANKING_START = 0.60
    RANKING_DONE = 0.65
    RELATION_VERIFICATION_START = 0.70
    RELATION_VERIFICATION_DONE = 0.80
    GRAPH_BUILT = 0.85
    GRAPH_OPTIMIZATION_START = 0.90
    SAINT_EMBEDDINGS = 0.92
    THEORIES_GENERATED = 0.93
    GRAPH_OPTIMIZATION_DONE = 0.95
    COMPLETE = 1.0


class PipelineStatus(StrEnum):
    """Authoritative lifecycle states for content pipeline jobs."""

    PENDING = "pending"
    QUEUED = "queued"
    LOADING = "loading"
    CHUNKING = "chunking"
    EXTRACTING = "extracting"
    EMBEDDING = "embedding"
    MERGING = "merging"
    RANKING = "ranking"
    VERIFYING = "verifying"
    BUILDING_GRAPH = "building_graph"
    OPTIMIZING = "optimizing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

    @property
    def is_terminal(self) -> bool:
        return self in {
            PipelineStatus.COMPLETED,
            PipelineStatus.FAILED,
            PipelineStatus.CANCELLED,
        }


@dataclass(slots=True)
class PipelineJob:
    """Persistent job state shared across orchestrator and repository layers."""

    job_id: str
    filename: str
    subject_id: str
    user_id: str | None = None
    status: PipelineStatus = PipelineStatus.PENDING
    current_step: str = ""
    progress: float = 0.0
    total_chunks: int = 0
    total_pages: int = 0
    page_batch_size: int = 10
    batch_count: int = 0
    failed_batch_count: int = 0
    partial_success: bool = False
    concepts_extracted: int = 0
    concepts_after_merge: int = 0
    relations_verified: int = 0
    graph_stats: dict[str, Any] = field(default_factory=dict)
    quality_report: PipelineQualityReport | None = None
    debug_trace: list[PipelineDebugTraceEntry] = field(default_factory=list)
    error_message: str | None = None
    error_code: str | None = None
    user_message: str | None = None
    retryable: bool = False
    result: dict[str, Any] | None = None
    partial_graph: dict[str, Any] | None = None
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    heartbeat_at: float = field(default_factory=time.time)
    completed_at: float | None = None
    source_s3_key: str | None = None
    prs_threshold: float | None = None
    min_confidence: float = 0.6
    apply_reduction: bool = True
    retry_count: int = 0
    cancel_requested: bool = False
    eta_seconds: float | None = None

    def mark_completed(self) -> None:
        now = time.time()
        self.status = PipelineStatus.COMPLETED
        self.progress = PipelineProgress.COMPLETE
        self.updated_at = now
        self.heartbeat_at = now
        self.completed_at = self.completed_at or now

    def mark_failed(self, message: str) -> None:
        now = time.time()
        self.status = PipelineStatus.FAILED
        self.error_message = message
        self.updated_at = now
        self.heartbeat_at = now

    def mark_cancelled(self, message: str) -> None:
        now = time.time()
        self.status = PipelineStatus.CANCELLED
        self.error_message = message
        self.updated_at = now
        self.heartbeat_at = now

    def reset_for_retry(self) -> None:
        now = time.time()
        self.status = PipelineStatus.QUEUED
        self.current_step = "Queued for retry"
        self.progress = 0.0
        self.error_message = None
        self.error_code = None
        self.user_message = None
        self.retryable = False
        self.quality_report = None
        self.debug_trace = []
        self.cancel_requested = False
        self.completed_at = None
        self.retry_count += 1
        self.updated_at = now
        self.heartbeat_at = now

    def request_cancel(self) -> None:
        self.cancel_requested = True
        self.updated_at = time.time()
