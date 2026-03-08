"""Pydantic models for API requests and responses."""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from enum import Enum


class ProcessingStatus(str, Enum):
    """Status of processing job."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class ConceptResponse(BaseModel):
    """Response model for concept."""
    concept_id: str
    subject_id: str
    name: str
    definition: str
    examples: List[str] = []
    num_relations: int = 0


class GraphStatsResponse(BaseModel):
    """Response model for graph statistics."""
    num_nodes: int
    num_edges: int
    density: float
    has_cycle: bool
    edge_types: Dict[str, int] = {}
    is_dag: bool = True
    optimization_stats: Optional[Dict[str, Any]] = None


class ProcessingResult(BaseModel):
    """Response model for processing results."""
    job_id: str
    subject_id: str
    status: ProcessingStatus

    # Stats
    total_chunks: Optional[int] = None
    concepts_extracted: Optional[int] = None
    concepts_after_merge: Optional[int] = None
    relations_verified: Optional[int] = None

    # Graph
    graph_stats: Optional[GraphStatsResponse] = None

    # ChromaDB
    concepts_in_chromadb: Optional[int] = None

    # Errors
    error_message: Optional[str] = None

    # Timing
    processing_time_seconds: Optional[float] = None


class UploadResponse(BaseModel):
    """Response model for file upload."""
    job_id: str
    filename: str
    file_size: int
    subject_id: str
    message: str


class SearchRequest(BaseModel):
    """Request model for concept search."""
    query: str = Field(..., min_length=1, description="Search query")
    subject_id: Optional[str] = Field(None, description="Filter by subject ID")
    k: int = Field(5, ge=1, le=50, description="Number of results")


class SearchResult(BaseModel):
    """Search result item."""
    concept_id: str
    name: str
    definition: str
    subject_id: str
    score: float
    content: str


class SearchResponse(BaseModel):
    """Response model for concept search."""
    query: str
    results: List[SearchResult]
    total_results: int


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    version: str
    components: Dict[str, str]


class ErrorResponse(BaseModel):
    """Error response model."""
    error: str
    detail: Optional[str] = None
    job_id: Optional[str] = None
