from types import SimpleNamespace

import networkx as nx

from api.core.content_pipeline.application.stages.result_assembly import (
    assemble_pipeline_result,
    build_graph_nodes,
    serialize_concepts,
    serialize_prerequisite_edges,
)


def test_serialize_concepts_returns_serializable_payloads_and_index_map():
    concepts = [
        SimpleNamespace(
            concept_id="c1",
            name="Alpha",
            definition="alpha def",
            examples=["ex1"],
            relations=[
                SimpleNamespace(
                    type="PREREQUISITE",
                    target_id="c2",
                    confidence=0.8,
                    evidence="because",
                )
            ],
        ),
        SimpleNamespace(
            concept_id="c2",
            name="Beta",
            definition="beta def",
            relations=[],
        ),
    ]

    concepts_data, concept_map = serialize_concepts(concepts)

    assert concept_map == {"c1": 0, "c2": 1}
    assert concepts_data == {
        "c1": {
            "name": "Alpha",
            "definition": "alpha def",
            "examples": ["ex1"],
            "relations": [],
        },
        "c2": {
            "name": "Beta",
            "definition": "beta def",
            "examples": [],
            "relations": [],
        },
    }


def test_serialize_prerequisite_edges_keeps_only_known_prerequisite_edges():
    graph = nx.DiGraph()
    graph.add_edge(
        "c1",
        "c2",
        relation_type="PREREQUISITE",
        confidence=0.8,
        evidence=["because"],
        sources=["mlp"],
    )
    graph.add_edge("c2", "ghost", relation_type="PREREQUISITE")
    graph.add_edge("c1", "c2x", relation_type="RELATED")

    prereq_edges = serialize_prerequisite_edges(graph, {"c1": 0, "c2": 1})

    assert prereq_edges == [
        {
            "source": "c1",
            "target": "c2",
            "confidence": 0.8,
            "evidence": ["because"],
            "sources": ["mlp"],
        }
    ]


def test_assemble_pipeline_result_builds_final_payload_shape():
    concepts_data = {
        "c1": {"name": "Alpha", "definition": "alpha def", "relations": [], "examples": []},
        "c2": {"name": "Beta", "definition": "beta def", "relations": [], "examples": []},
    }
    concept_map = {"c1": 0, "c2": 1}
    prereq_edges = [{"source": "c1", "target": "c2"}]

    result = assemble_pipeline_result(
        concepts_data=concepts_data,
        concept_map=concept_map,
        prereq_edges=prereq_edges,
        concept_embeddings=[[0.1], [0.2]],
        stats={"num_nodes": 2},
    )

    assert result == {
        "concepts_data": concepts_data,
        "concept_map": concept_map,
        "prereq_edges": prereq_edges,
        "concept_embeddings": [[0.1], [0.2]],
        "graph": {
            "nodes": build_graph_nodes(concepts_data, concept_map),
            "edges": prereq_edges,
        },
        "stats": {"num_nodes": 2},
    }
