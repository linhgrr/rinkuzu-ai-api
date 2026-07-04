"""Result assembly helpers for the content pipeline."""

from __future__ import annotations

from typing import Any


def serialize_concepts(concepts: list[Any]) -> tuple[dict[str, dict[str, Any]], dict[str, int]]:
    """Convert extracted concept objects into serializable API payloads."""
    concepts_data: dict[str, dict[str, Any]] = {}
    concept_map: dict[str, int] = {}

    for index, concept in enumerate(concepts):
        concept_id = concept.concept_id
        concept_map[concept_id] = index
        concepts_data[concept_id] = {
            "name": concept.name,
            "definition": concept.definition,
            "examples": concept.examples if hasattr(concept, "examples") else [],
            "relations": [],
        }

    return concepts_data, concept_map


def serialize_prerequisite_edges(graph: Any, concept_map: dict[str, int]) -> list[dict[str, Any]]:
    """Extract prerequisite edges from the final graph payload."""
    prereq_edges = []
    for source_id, target_id, data in graph.edges(data=True):
        relation_type = data.get("relation_type", "PREREQUISITE")
        if (
            relation_type == "PREREQUISITE"
            and source_id in concept_map
            and target_id in concept_map
        ):
            edge: dict[str, Any] = {"source": source_id, "target": target_id}
            for key in (
                "confidence",
                "evidence",
                "reasoning",
                "sources",
                "ranker_score",
                "extraction_confidence",
            ):
                value = data.get(key)
                if value not in (None, [], ""):
                    edge[key] = value
            prereq_edges.append(edge)
    return prereq_edges


def build_graph_nodes(
    concepts_data: dict[str, dict[str, Any]],
    concept_map: dict[str, int],
) -> list[dict[str, Any]]:
    """Build the node list used in the final graph result payload."""
    return [
        {
            "id": concept_id,
            "index": index,
            "name": concepts_data[concept_id]["name"],
            "definition": concepts_data[concept_id].get("definition", ""),
        }
        for concept_id, index in concept_map.items()
    ]


def assemble_pipeline_result(
    *,
    concepts_data: dict[str, dict[str, Any]],
    concept_map: dict[str, int],
    prereq_edges: list[dict[str, Any]],
    concept_embeddings: list[list[float]] | None,
    stats: dict[str, Any],
) -> dict[str, Any]:
    """Build the final persisted pipeline result payload."""
    return {
        "concepts_data": concepts_data,
        "concept_map": concept_map,
        "prereq_edges": prereq_edges,
        "concept_embeddings": concept_embeddings,
        "graph": {
            "nodes": build_graph_nodes(concepts_data, concept_map),
            "edges": prereq_edges,
        },
        "stats": stats,
    }
