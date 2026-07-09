"""Domain types for prerequisite relation discovery."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypedDict

JsonPrimitive = str | int | float | bool | None
JsonObject = dict[str, "JsonValue"]
JsonValue = JsonPrimitive | list["JsonValue"] | JsonObject


class QualityGateChecks(TypedDict):
    has_concepts: bool
    edges_reference_known_concepts: bool
    is_dag: bool
    extraction_failure_ratio_ok: bool
    has_verified_relation_when_multi_concept: bool


class PipelineQualityReport(TypedDict):
    passed: bool
    checks: QualityGateChecks
    concept_count: int
    candidate_relation_count: int
    verified_relation_count: int
    extraction_failure_ratio: float
    invalid_edge_count: int


class PipelineDebugArtifact(TypedDict, total=False):
    artifact_id: str
    kind: str
    label: str
    index: int
    page_start: int
    page_end: int
    input: JsonObject
    output: JsonObject
    content_type: str
    content: str
    truncated: bool


class PipelineDebugTraceEntry(TypedDict, total=False):
    step_id: str
    label: str
    status: str
    started_at: float
    completed_at: float | None
    duration_ms: float | None
    input: JsonObject
    output: JsonObject | None
    error: str | None
    artifacts: list[PipelineDebugArtifact]


@dataclass(frozen=True, slots=True)
class RelationCandidate:
    """A prerequisite edge proposed by extraction, ranking, or both."""

    source_id: str
    target_id: str
    sources: frozenset[str]
    ranker_score: float | None = None
    extraction_confidence: float | None = None
    extracted_evidences: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class VerifiedRelation:
    """A prerequisite edge accepted by the LLM verifier."""

    source_id: str
    target_id: str
    confidence: float
    evidences: tuple[str, ...] = ()
    reasoning: str | None = None
    sources: frozenset[str] = frozenset()
    ranker_score: float | None = None
    extraction_confidence: float | None = None
