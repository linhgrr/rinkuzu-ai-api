import asyncio
from types import SimpleNamespace

import networkx as nx

from api.core.content_pipeline.application.stages.graph_building import (
    build_knowledge_graph,
    build_partial_graph,
    remove_invalid_graph_members,
)
from api.core.content_pipeline.domain.jobs import PipelineJob, PipelineStatus
from api.core.content_pipeline.domain.relations import VerifiedRelation
from api.core.content_pipeline.infrastructure.graph.builder import KnowledgeGraphBuilder
from api.core.content_pipeline.infrastructure.llm.schemas import Concept, Relation


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


def test_build_knowledge_graph_does_not_insert_unverified_extracted_relations():
    concepts = [
        Concept(
            concept_id="ohms_law",
            subject_id="physics",
            name="Ohm's law",
            definition="Voltage equals current multiplied by resistance.",
            relations=[
                Relation(
                    type="PREREQUISITE",
                    target_id="electric_current",
                    confidence=0.9,
                    evidence="Ohm's law uses electric current in its formula.",
                )
            ],
        ),
        Concept(
            concept_id="electric_current",
            subject_id="physics",
            name="Electric current",
            definition="The rate of flow of electric charge.",
            relations=[],
        ),
    ]
    job = PipelineJob(job_id="job-1", filename="lesson.pdf", subject_id="physics")

    async def persist_job_state(job_arg, status, step, progress):
        assert job_arg is job

    graph, _stats = asyncio.run(
        build_knowledge_graph(
            job,
            concepts=concepts,
            verified_relations=[],
            knowledge_graph_builder_factory=KnowledgeGraphBuilder,
            persist_job_state=persist_job_state,
        )
    )

    assert set(graph.edges()) == set()
    assert job.partial_graph == {
        "nodes": [
            {"id": "ohms_law", "name": "Ohm's law"},
            {"id": "electric_current", "name": "Electric current"},
        ],
        "edges": [],
    }


class _BuilderStub:
    def __init__(self, subject_id: str) -> None:
        self.subject_id = subject_id
        self._graph = nx.DiGraph()

    def add_concept_nodes(self, concepts):
        for concept in concepts:
            self._graph.add_node(concept.concept_id)

    def get_graph(self):
        return self._graph

    def add_relation(self, source_id: str, target_id: str, relation_type: str, **kwargs):
        self._graph.add_edge(source_id, target_id, relation_type=relation_type, **kwargs)

    def get_stats(self):
        return {"builder_subject_id": self.subject_id}


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
        VerifiedRelation(source_id="c2", target_id="c1", confidence=0.8),
        VerifiedRelation(source_id="ghost", target_id="c1", confidence=0.8),
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
    assert set(graph.edges()) == {("c2", "c1")}
    assert stats == {
        "base_graph_stats": {"builder_subject_id": "algebra"},
        "verified_relation_count": 1,
    }
    assert job.partial_graph == {
        "nodes": [
            {"id": "c1", "name": "Alpha"},
            {"id": "c2", "name": "Beta"},
        ],
        "edges": [{"source": "c2", "target": "c1"}],
    }
    assert calls == [
        (PipelineStatus.BUILDING_GRAPH, "Building knowledge graph...", 0.85),
    ]
