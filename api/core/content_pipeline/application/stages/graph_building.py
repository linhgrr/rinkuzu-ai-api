"""Graph building stage for the content pipeline."""

from __future__ import annotations

from typing import Any, Awaitable, Callable

from loguru import logger

from ...domain.jobs import PipelineJob, PipelineStatus
from .execution import run_blocking_stage


PersistJobStateFn = Callable[[PipelineJob, PipelineStatus, str, float], Awaitable[None]]
VerifiedRelation = tuple[str, str, Any]


def sanitize_concept_relations(all_concepts: list[Any]) -> tuple[int, int]:
    """Remove invalid or duplicate prerequisite relations attached to concepts."""
    concept_ids = {
        str(getattr(concept, "concept_id", "")).strip()
        for concept in all_concepts
        if getattr(concept, "concept_id", None)
    }
    kept = 0
    dropped = 0

    for concept in all_concepts:
        source_id = str(getattr(concept, "concept_id", "")).strip()
        seen_targets = set()
        cleaned_relations = []

        for relation in getattr(concept, "relations", []) or []:
            relation_type = str(getattr(relation, "type", "")).strip().upper()
            target_id = str(getattr(relation, "target_id", "")).strip()

            if relation_type != "PREREQUISITE" or not target_id or target_id == source_id:
                dropped += 1
                continue
            if target_id not in concept_ids:
                dropped += 1
                continue
            if target_id in seen_targets:
                continue

            seen_targets.add(target_id)
            cleaned_relations.append(relation)
            kept += 1

        concept.relations = cleaned_relations

    return kept, dropped


def build_partial_graph(graph, all_concepts: list[Any]) -> dict[str, Any]:
    """Build the partial graph payload exposed while processing is in progress."""
    concept_name_map = {
        getattr(concept, "concept_id", ""): getattr(concept, "name", "")
        for concept in all_concepts
    }
    return {
        "nodes": [
            {"id": node_id, "name": concept_name_map.get(node_id, str(node_id))}
            for node_id in graph.nodes()
        ],
        "edges": [{"source": src, "target": tgt} for src, tgt in graph.edges()],
    }


def remove_invalid_graph_members(graph, concept_ids: set[str]) -> None:
    """Drop non-prerequisite edges and orphan nodes from the graph."""
    edges_to_remove = []
    for source_id, target_id, data in list(graph.edges(data=True)):
        relation_type = str(data.get("relation_type", "PREREQUISITE")).upper()
        if relation_type != "PREREQUISITE" or source_id not in concept_ids or target_id not in concept_ids:
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
        0.85,
    )

    concept_ids = {
        str(getattr(concept, "concept_id", "")).strip()
        for concept in concepts
        if getattr(concept, "concept_id", None)
    }
    extracted_relation_count, dropped_relation_count = sanitize_concept_relations(concepts)
    if dropped_relation_count:
        logger.info(f"[Pipeline] Dropped {dropped_relation_count} invalid extracted relations")

    builder = knowledge_graph_builder_factory(job.subject_id)
    await run_blocking_stage(
        builder.add_concepts,
        concepts,
        stage_name="graph_building",
    )
    graph = builder.get_graph()
    remove_invalid_graph_members(graph, concept_ids)

    existing_edges = set(graph.edges())
    logger.debug(f"[Pipeline] Added {extracted_relation_count} relations from extraction")

    verified_relation_count = 0
    for source_id, target_id, evaluation in verified_relations:
        if source_id not in concept_ids or target_id not in concept_ids:
            continue
        if not hasattr(evaluation, "direction"):
            continue

        edge = None
        if evaluation.direction == "A_to_B":
            edge = (source_id, target_id)
        elif evaluation.direction == "B_to_A":
            edge = (target_id, source_id)

        if edge and edge not in existing_edges:
            await run_blocking_stage(
                builder.add_relation,
                edge[0],
                edge[1],
                "PREREQUISITE",
                stage_name="graph_relation_insertion",
            )
            existing_edges.add(edge)
            verified_relation_count += 1

    remove_invalid_graph_members(graph, concept_ids)
    job.partial_graph = build_partial_graph(graph, concepts)

    builder_stats = {}
    if hasattr(builder, "get_stats"):
        builder_stats = dict(
            await run_blocking_stage(
                builder.get_stats,
                stage_name="graph_stats_collection",
            )
        )

    return graph, {
        "base_graph_stats": builder_stats,
        "extracted_relation_count": extracted_relation_count,
        "verified_relation_count": verified_relation_count,
    }
