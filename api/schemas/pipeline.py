"""
schemas/pipeline.py — Pipeline-related Pydantic models.
"""

from typing import Any, Literal

from pydantic import BaseModel, Field

PipelineStatus = Literal[
    "pending",
    "queued",
    "loading",
    "chunking",
    "extracting",
    "embedding",
    "merging",
    "ranking",
    "verifying",
    "building_graph",
    "optimizing",
    "completed",
    "failed",
    "cancelled",
]


class PipelineProcessResponse(BaseModel):
    job_id: str
    filename: str
    file_size: int
    subject_id: str
    status: PipelineStatus
    status_url: str
    page_batch_size: int = 10
    retry_after_seconds: int = 2
    message: str


class PipelineFailedBatchResponse(BaseModel):
    batch_index: int
    page_start: int
    page_end: int
    reason: str


class PipelinePartialGraphNodeResponse(BaseModel):
    id: str
    name: str


class PipelinePartialGraphEdgeResponse(BaseModel):
    source: str
    target: str


class PipelinePartialGraphResponse(BaseModel):
    nodes: list[PipelinePartialGraphNodeResponse] = Field(default_factory=list)
    edges: list[PipelinePartialGraphEdgeResponse] = Field(default_factory=list)


class PipelineGraphNodeResponse(BaseModel):
    id: str
    index: int
    name: str
    definition: str = ""


class PipelineGraphResponse(BaseModel):
    nodes: list[PipelineGraphNodeResponse] = Field(default_factory=list)
    edges: list[PipelinePartialGraphEdgeResponse] = Field(default_factory=list)


class PipelineJobResultResponse(BaseModel):
    graph: PipelineGraphResponse = Field(
        default_factory=lambda: PipelineGraphResponse(nodes=[], edges=[])
    )
    stats: dict[str, Any] = Field(default_factory=dict)
    n_concepts: int = 0


class PipelineJobStatusResponse(BaseModel):
    job_id: str
    filename: str = ""
    subject_id: str = ""
    status: PipelineStatus
    current_step: str = ""
    progress: float = 0.0
    total_chunks: int = 0
    page_batch_size: int = 10
    batch_count: int = 0
    failed_batch_count: int = 0
    partial_success: bool = False
    concepts_extracted: int = 0
    concepts_after_merge: int = 0
    relations_verified: int = 0
    graph_stats: dict[str, Any] = Field(default_factory=dict)
    error_message: str | None = None
    error_code: str | None = None
    user_message: str | None = None
    retryable: bool = False
    failed_batches: list[PipelineFailedBatchResponse] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    is_terminal: bool = False
    is_delayed: bool = False
    created_at: float = 0.0
    updated_at: float = 0.0
    heartbeat_at: float = 0.0
    retry_after_seconds: int = 2
    partial_graph: PipelinePartialGraphResponse | None = None
    result: PipelineJobResultResponse | None = None


class PipelineSessionCreateResponse(BaseModel):
    session_id: str
    n_concepts: int
    source: str
    job_id: str
    status: Literal["active"]
