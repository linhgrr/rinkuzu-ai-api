"""Job models for the unified content pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
import time
from typing import Any


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

    def mark_completed(self) -> None:
        now = time.time()
        self.status = PipelineStatus.COMPLETED
        self.progress = 1.0
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
