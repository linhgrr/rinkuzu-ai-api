import asyncio
from types import SimpleNamespace

import networkx as nx

from api.core.content_pipeline.application.stages.graph_building import (
    build_knowledge_graph,
    build_partial_graph,
    remove_invalid_graph_members,
    sanitize_concept_relations,
)
from api.core.content_pipeline.domain.jobs import PipelineJob, PipelineStatus


def test_sanitize_concept_relations_drops_invalid_and_duplicate_relations():
    concepts = [
        SimpleNamespace(
            concept_id="c1",
            name="Alpha",
            relations=[
                SimpleNamespace(type="PREREQUISITE", target_id="c2"),
                SimpleNamespace(type="PREREQUISITE", target_id="c2"),
                SimpleNamespace(type="RELATED", target_id="c2"),
                SimpleNamespace(type="PREREQUISITE", target_id="missing"),
                SimpleNamespace(type="PREREQUISITE", target_id="c1"),
            ],
        ),
        SimpleNamespace(concept_id="c2", name="Beta", relations=[]),
    ]

    kept, dropped = sanitize_concept_relations(concepts)

    assert kept == 1
    assert dropped == 3
    assert len(concepts[0].relations) == 1
    assert concepts[0].relations[0].target_id == "c2"


def test_remove_invalid_graph_members_keeps_only_valid_prerequisite_edges():
    graph = nx.DiGraph()
    graph.add_edge("c1", "c2", relation_type="PREREQUISITE")
    graph.add_edge("c2", "ghost", relation_type="PREREQUISITE")
    graph.add_edge("c1", "c2x", relation_type="RELATED")
    graph.add_node("orphan")

    remove_invalid_graph_members(graph, {"c1", "c2"})

    assert set(graph.nodes()) == {"c1", "c2"}
    assert list(graph.edges()) == [("c1", "c2")]


def test_build_partial_graph_serializes_nodes_and_edges():
    graph = nx.DiGraph()
    graph.add_edge("c1", "c2")
    concepts = [
        SimpleNamespace(concept_id="c1", name="Alpha"),
        SimpleNamespace(concept_id="c2", name="Beta"),
    ]

    partial = build_partial_graph(graph, concepts)

    assert partial == {
        "nodes": [
            {"id": "c1", "name": "Alpha"},
            {"id": "c2", "name": "Beta"},
        ],
        "edges": [{"source": "c1", "target": "c2"}],
    }


class _BuilderStub:
    def __init__(self, subject_id: str) -> None:
        self.subject_id = subject_id
        self._graph = nx.DiGraph()

    def add_concepts(self, concepts):
        for concept in concepts:
            self._graph.add_node(concept.concept_id)
            for relation in getattr(concept, "relations", []) or []:
                self._graph.add_edge(
                    concept.concept_id,
                    relation.target_id,
                    relation_type=relation.type,
                )

    def get_graph(self):
        return self._graph

    def add_relation(self, source_id: str, target_id: str, relation_type: str):
        self._graph.add_edge(source_id, target_id, relation_type=relation_type)


def test_build_knowledge_graph_updates_partial_graph_and_stats():
    concepts = [
        SimpleNamespace(
            concept_id="c1",
            name="Alpha",
            relations=[SimpleNamespace(type="PREREQUISITE", target_id="c2")],
        ),
        SimpleNamespace(
            concept_id="c2",
            name="Beta",
            relations=[],
        ),
    ]
    verified_relations = [
        ("c2", "c1", SimpleNamespace(direction="A_to_B")),
        ("ghost", "c1", SimpleNamespace(direction="A_to_B")),
    ]
    job = PipelineJob(job_id="job-1", filename="lesson.pdf", subject_id="algebra")
    calls = []

    async def persist_job_state(job_arg, status, step, progress):
        assert job_arg is job
        calls.append((status, step, progress))

    graph, stats = asyncio.run(
        build_knowledge_graph(
            job,
            concepts=concepts,
            verified_relations=verified_relations,
            knowledge_graph_builder_factory=_BuilderStub,
            persist_job_state=persist_job_state,
        )
    )

    assert isinstance(graph, nx.DiGraph)
    assert set(graph.edges()) == {("c1", "c2"), ("c2", "c1")}
    assert stats == {
        "extracted_relation_count": 1,
        "verified_relation_count": 1,
    }
    assert job.partial_graph == {
        "nodes": [
            {"id": "c1", "name": "Alpha"},
            {"id": "c2", "name": "Beta"},
        ],
        "edges": [
            {"source": "c1", "target": "c2"},
            {"source": "c2", "target": "c1"},
        ],
    }
    assert calls == [
        (PipelineStatus.BUILDING_GRAPH, "Building knowledge graph...", 0.85),
    ]
