"""
schemas/pipeline.py — Pipeline-related Pydantic models.
"""

from typing import Literal

from pydantic import BaseModel, Field

from api.core.content_pipeline.domain.jobs import PipelineStatus

from .enums import PipelineSessionSource


class PipelineProcessResponse(BaseModel):
    job_id: str
    filename: str
    file_size: int
    subject_id: str
    status: PipelineStatus
    status_url: str
    page_batch_size: int = 10
    retry_after_seconds: int = 3


class PipelineRuntimeStatusResponse(BaseModel):
    available: bool
    service_initialized: bool


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
    confidence: float | None = None
    evidence: list[str] | None = None
    reasoning: str | None = None
    sources: list[str] | None = None
    ranker_score: float | None = None
    extraction_confidence: float | None = None


class PipelineQualityChecksResponse(BaseModel):
    has_concepts: bool = False
    edges_reference_known_concepts: bool = False
    is_dag: bool = False
    extraction_failure_ratio_ok: bool = False
    has_verified_relation_when_multi_concept: bool = False


class PipelineQualityReportResponse(BaseModel):
    passed: bool = False
    checks: PipelineQualityChecksResponse = Field(default_factory=PipelineQualityChecksResponse)
    concept_count: int = 0
    candidate_relation_count: int = 0
    verified_relation_count: int = 0
    extraction_failure_ratio: float = 0.0
    invalid_edge_count: int = 0


class PipelineDebugArtifactResponse(BaseModel):
    artifact_id: str = ""
    kind: str = ""
    label: str = ""
    index: int = 0
    page_start: int | None = None
    page_end: int | None = None
    input: dict[str, object] = Field(default_factory=dict)
    output: dict[str, object] = Field(default_factory=dict)
    content_type: str = "text/plain"
    content: str = ""
    truncated: bool = False


class PipelineDebugTraceEntryResponse(BaseModel):
    step_id: str = ""
    label: str = ""
    status: str = ""
    started_at: float = 0.0
    completed_at: float | None = None
    duration_ms: float | None = None
    input: dict[str, object] = Field(default_factory=dict)
    output: dict[str, object] | None = None
    error: str | None = None
    artifacts: list[PipelineDebugArtifactResponse] = Field(default_factory=list)


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


class PipelineCycleStatsResponse(BaseModel):
    removed_cycles: int | None = None


class PipelineGraphStatsResponse(BaseModel):
    num_nodes: int | None = None
    num_edges: int | None = None
    is_dag: bool | None = None
    relation_candidates: int | None = None
    relation_candidates_from_extraction: int | None = None
    relation_candidates_from_mlp: int | None = None
    relations_inserted_after_verification: int | None = None
    relations_extraction_candidates_dropped: int | None = None
    relations_verified: int | None = None
    quality_report: PipelineQualityReportResponse | None = None
    builder_subject_id: str | None = None
    cycle_stats: PipelineCycleStatsResponse | None = None


class PipelineJobResultResponse(BaseModel):
    graph: PipelineGraphResponse = Field(
        default_factory=lambda: PipelineGraphResponse(nodes=[], edges=[])
    )
    stats: PipelineGraphStatsResponse = Field(default_factory=PipelineGraphStatsResponse)
    quality_report: PipelineQualityReportResponse | None = None
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
    graph_stats: PipelineGraphStatsResponse = Field(default_factory=PipelineGraphStatsResponse)
    quality_report: PipelineQualityReportResponse | None = None
    debug_trace: list[PipelineDebugTraceEntryResponse] = Field(default_factory=list)
    error_message: str | None = None
    error_code: str | None = None
    user_message: str | None = None
    eta_seconds: float | None = None
    retry_count: int = 0
    retryable: bool = False
    failed_batches: list[PipelineFailedBatchResponse] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    is_terminal: bool = False
    is_delayed: bool = False
    created_at: float = 0.0
    updated_at: float = 0.0
    heartbeat_at: float = 0.0
    retry_after_seconds: int = 3
    partial_graph: PipelinePartialGraphResponse | None = None
    result: PipelineJobResultResponse | None = None


class PipelineJobListItemResponse(BaseModel):
    job_id: str
    filename: str = ""
    subject_id: str = ""
    status: PipelineStatus
    current_step: str = ""
    progress: float = 0.0
    page_batch_size: int = 10
    batch_count: int = 0
    failed_batch_count: int = 0
    partial_success: bool = False
    concepts_extracted: int = 0
    concepts_after_merge: int = 0
    relations_verified: int = 0
    quality_report: PipelineQualityReportResponse | None = None
    error_code: str | None = None
    user_message: str | None = None
    retryable: bool = False
    retry_count: int = 0
    eta_seconds: float | None = None
    created_at: float = 0.0
    updated_at: float = 0.0
    heartbeat_at: float = 0.0
    completed_at: float | None = None
    is_terminal: bool = False
    is_delayed: bool = False
    retry_after_seconds: int = 3


class PipelineJobListResponse(BaseModel):
    jobs: list[PipelineJobListItemResponse] = Field(default_factory=list)
    count: int = 0


class PipelineJobCancelResponse(BaseModel):
    job_id: str
    status: PipelineStatus | Literal["cancelling"]


class PipelineJobRetryResponse(BaseModel):
    job_id: str
    status: PipelineStatus
    status_url: str
    retry_count: int = 0


class PipelineSessionCreateResponse(BaseModel):
    session_id: str
    n_concepts: int
    source: PipelineSessionSource
    job_id: str
    status: Literal["active"]
