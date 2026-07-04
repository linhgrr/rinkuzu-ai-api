"""Production quality gate for completed PDF-to-KG pipeline output."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import networkx as nx

from api.core.content_pipeline.domain.errors import PipelineQualityGateError

if TYPE_CHECKING:
    from api.core.content_pipeline.domain.jobs import PipelineJob
    from api.core.content_pipeline.domain.relations import (
        PipelineQualityReport,
        QualityGateChecks,
    )


def validate_pipeline_quality(
    job: PipelineJob,
    *,
    graph: Any,
    concepts: list[Any],
    concept_ids: set[str],
    extraction_failure_ratio: float,
    max_extraction_failure_ratio: float,
    candidate_count: int,
    verified_relation_count: int,
) -> PipelineQualityReport:
    """Return a quality report or raise before unreliable KG output is published."""
    invalid_edges = [
        (source_id, target_id)
        for source_id, target_id in graph.edges()
        if source_id not in concept_ids or target_id not in concept_ids
    ]
    is_dag = nx.is_directed_acyclic_graph(graph)
    concept_count = len(concepts)
    checks: QualityGateChecks = {
        "has_concepts": concept_count > 0,
        "edges_reference_known_concepts": not invalid_edges,
        "is_dag": is_dag,
        "extraction_failure_ratio_ok": extraction_failure_ratio <= max_extraction_failure_ratio,
        "has_verified_relation_when_multi_concept": (
            concept_count <= 1 or verified_relation_count > 0
        ),
    }
    report: PipelineQualityReport = {
        "passed": all(checks.values()),
        "checks": checks,
        "concept_count": concept_count,
        "candidate_relation_count": candidate_count,
        "verified_relation_count": verified_relation_count,
        "extraction_failure_ratio": extraction_failure_ratio,
        "invalid_edge_count": len(invalid_edges),
    }
    job.quality_report = report
    if report["passed"]:
        return report

    failed_checks = [name for name, passed in checks.items() if not passed]
    raise PipelineQualityGateError(
        "Pipeline quality gate failed: " + ", ".join(failed_checks),
        report,
    )
