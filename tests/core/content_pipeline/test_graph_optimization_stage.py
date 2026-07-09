import asyncio
from types import SimpleNamespace

import networkx as nx

from api.domains.content_pipeline.application.stages import graph_optimization
from api.domains.content_pipeline.application.stages.graph_optimization import optimize_graph
from api.domains.content_pipeline.domain.jobs import PipelineJob, PipelineStatus


def test_optimize_graph_runs_cycle_removal_and_reduction_when_needed():
    graph = nx.DiGraph()
    graph.add_edge("c1", "c2")
    graph.add_edge("c2", "c1")
    concepts = [
        SimpleNamespace(concept_id="c1", name="Alpha"),
        SimpleNamespace(concept_id="c2", name="Beta"),
    ]
    job = PipelineJob(job_id="job-1", filename="lesson.pdf", subject_id="algebra")
    calls = []
    cycle_calls = []
    reduction_calls = []

    async def persist_job_state(job_arg, status, step, progress):
        assert job_arg is job
        calls.append((status, step, progress))

    async def make_dag_with_llm(input_graph):
        cycle_calls.append(set(input_graph.edges()))
        dag = nx.DiGraph()
        dag.add_edge("c1", "c2")
        return dag, {"removed_cycles": 1}

    def apply_transitive_reduction(input_graph):
        reduction_calls.append(set(input_graph.edges()))
        return input_graph

    optimized_graph, stats = asyncio.run(
        optimize_graph(
            job,
            graph=graph,
            concepts=concepts,
            apply_reduction=True,
            make_dag_with_llm=make_dag_with_llm,
            apply_transitive_reduction=apply_transitive_reduction,
            persist_job_state=persist_job_state,
        )
    )

    assert set(optimized_graph.edges()) == {("c1", "c2")}
    assert cycle_calls == [{("c1", "c2"), ("c2", "c1")}]
    assert reduction_calls == [{("c1", "c2")}]
    assert stats == {
        "num_nodes": 2,
        "num_edges": 1,
        "is_dag": True,
        "cycle_stats": {"removed_cycles": 1},
    }
    assert job.partial_graph == {
        "nodes": [
            {"id": "c1", "name": "Alpha"},
            {"id": "c2", "name": "Beta"},
        ],
        "edges": [{"source": "c1", "target": "c2"}],
    }
    assert calls == [
        (PipelineStatus.OPTIMIZING, "Removing cycles, building DAG...", 0.90),
        (PipelineStatus.OPTIMIZING, "Removing cycles, building DAG...", 0.95),
    ]


def test_optimize_graph_skips_cycle_removal_for_existing_dag():
    graph = nx.DiGraph()
    graph.add_edge("c1", "c2")
    concepts = [
        SimpleNamespace(concept_id="c1", name="Alpha"),
        SimpleNamespace(concept_id="c2", name="Beta"),
    ]
    job = PipelineJob(job_id="job-2", filename="lesson.pdf", subject_id="algebra")

    async def persist_job_state(job_arg, status, step, progress):
        pass

    async def make_dag_with_llm(_):
        raise AssertionError("cycle removal should not run for a DAG")

    def apply_transitive_reduction(input_graph):
        return input_graph

    optimized_graph, stats = asyncio.run(
        optimize_graph(
            job,
            graph=graph,
            concepts=concepts,
            apply_reduction=False,
            make_dag_with_llm=make_dag_with_llm,
            apply_transitive_reduction=apply_transitive_reduction,
            persist_job_state=persist_job_state,
        )
    )

    assert set(optimized_graph.edges()) == {("c1", "c2")}
    assert stats["is_dag"] is True
    assert stats["cycle_stats"] is None


def test_optimize_graph_uses_extended_cycle_timeout(monkeypatch):
    graph = nx.DiGraph()
    graph.add_edge("c1", "c2")
    graph.add_edge("c2", "c1")
    concepts = [
        SimpleNamespace(concept_id="c1", name="Alpha"),
        SimpleNamespace(concept_id="c2", name="Beta"),
    ]
    job = PipelineJob(job_id="job-3", filename="lesson.pdf", subject_id="algebra")
    captured = {}

    async def persist_job_state(*_args, **_kwargs):
        return None

    monkeypatch.setattr(
        graph_optimization,
        "get_settings",
        lambda: SimpleNamespace(content_pipeline_graph_cycle_timeout_sec=900),
    )

    async def make_dag_with_llm(_):
        captured["timeout_sec"] = graph_optimization._resolve_cycle_removal_timeout()
        dag = nx.DiGraph()
        dag.add_edge("c1", "c2")
        return dag, {"removed_cycles": 1}

    optimized_graph, stats = asyncio.run(
        optimize_graph(
            job,
            graph=graph,
            concepts=concepts,
            apply_reduction=False,
            make_dag_with_llm=make_dag_with_llm,
            apply_transitive_reduction=lambda input_graph: input_graph,
            persist_job_state=persist_job_state,
        )
    )

    assert set(optimized_graph.edges()) == {("c1", "c2")}
    assert stats["cycle_stats"] == {"removed_cycles": 1}
    assert captured == {
        "timeout_sec": 900.0,
    }
