"""Graph building stage for the content pipeline."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any

from api.core.content_pipeline.domain.jobs import PipelineJob, PipelineProgress, PipelineStatus

from .execution import run_blocking_stage

if TYPE_CHECKING:
    from api.core.content_pipeline.domain.relations import VerifiedRelation

PersistJobStateFn = Callable[[PipelineJob, PipelineStatus, str, float], Awaitable[None]]


def build_partial_graph(graph: Any, all_concepts: list[Any]) -> dict[str, Any]:
    """Build the partial graph payload exposed while processing is in progress."""
    concept_name_map = {
        getattr(concept, "concept_id", ""): getattr(concept, "name", "") for concept in all_concepts
    }
    return {
        "nodes": [
            {"id": node_id, "name": concept_name_map.get(node_id, str(node_id))}
            for node_id in graph.nodes()
        ],
        "edges": [{"source": src, "target": tgt} for src, tgt in graph.edges()],
    }


def remove_invalid_graph_members(graph: Any, concept_ids: set[str]) -> None:
    """Drop non-prerequisite edges and orphan nodes from the graph."""
    edges_to_remove = []
    for source_id, target_id, data in list(graph.edges(data=True)):
        relation_type = str(data.get("relation_type", "PREREQUISITE")).upper()
        if (
            relation_type != "PREREQUISITE"
            or source_id not in concept_ids
            or target_id not in concept_ids
        ):
            edges_to_remove.append((source_id, target_id))
    if edges_to_remove:
        graph.remove_edges_from(edges_to_remove)

    orphan_nodes = [node_id for node_id in list(graph.nodes()) if node_id not in concept_ids]
    if orphan_nodes:
        graph.remove_nodes_from(orphan_nodes)


async def build_knowledge_graph(
    job: PipelineJob,
    *,
    concepts: list[Any],
    verified_relations: list[VerifiedRelation],
    knowledge_graph_builder_factory: Callable[[str], Any],
    persist_job_state: PersistJobStateFn,
) -> tuple[Any, dict[str, Any]]:
    """Build the knowledge graph from concepts and verified prerequisite relations."""
    await persist_job_state(
        job,
        PipelineStatus.BUILDING_GRAPH,
        "Building knowledge graph...",
        PipelineProgress.GRAPH_BUILT,
    )

    concept_ids = {
        str(getattr(concept, "concept_id", "")).strip()
        for concept in concepts
        if getattr(concept, "concept_id", None)
    }
    builder = knowledge_graph_builder_factory(job.subject_id)
    add_nodes = (
        builder.add_concept_nodes if hasattr(builder, "add_concept_nodes") else builder.add_concepts
    )
    await run_blocking_stage(
        add_nodes,
        concepts,
        stage_name="graph_building",
    )
    graph = builder.get_graph()
    remove_invalid_graph_members(graph, concept_ids)

    existing_edges = set(graph.edges())

    verified_relation_count = 0
    for relation in verified_relations:
        if relation.source_id not in concept_ids or relation.target_id not in concept_ids:
            continue
        edge = (relation.source_id, relation.target_id)
        if edge not in existing_edges:
            await run_blocking_stage(
                builder.add_relation,
                edge[0],
                edge[1],
                "PREREQUISITE",
                evidence=list(relation.evidences),
                confidence=relation.confidence,
                reasoning=relation.reasoning,
                sources=sorted(relation.sources),
                ranker_score=relation.ranker_score,
                extraction_confidence=relation.extraction_confidence,
                stage_name="graph_relation_insertion",
            )
            existing_edges.add(edge)
            verified_relation_count += 1

    remove_invalid_graph_members(graph, concept_ids)
    job.partial_graph = build_partial_graph(graph, concepts)

    builder_stats: dict[str, Any] = {}
    if hasattr(builder, "get_stats"):
        builder_stats = dict(
            await run_blocking_stage(
                builder.get_stats,
                stage_name="graph_stats_collection",
            )
        )

    return graph, {
        "base_graph_stats": builder_stats,
        "verified_relation_count": verified_relation_count,
    }
