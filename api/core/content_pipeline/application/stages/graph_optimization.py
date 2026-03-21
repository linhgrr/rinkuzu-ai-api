"""Graph optimization stage for the content pipeline."""

from __future__ import annotations

from typing import Any, Awaitable, Callable

from ...domain.jobs import PipelineJob, PipelineStatus
from .graph_building import build_partial_graph
from .execution import run_blocking_stage


PersistJobStateFn = Callable[[PipelineJob, PipelineStatus, str, float], Awaitable[None]]


async def optimize_graph(
    job: PipelineJob,
    *,
    graph,
    concepts: list[Any],
    apply_reduction: bool,
    make_dag_with_llm: Callable[[Any], tuple[Any, Any]],
    apply_transitive_reduction: Callable[[Any], Any],
    persist_job_state: PersistJobStateFn,
) -> tuple[Any, dict[str, Any]]:
    """Make the graph a DAG, optionally apply transitive reduction, and derive stats."""
    await persist_job_state(
        job,
        PipelineStatus.OPTIMIZING,
        "Removing cycles, building DAG...",
        0.90,
    )

    import networkx as nx

    cycle_stats: dict[str, Any] | None = None
    if not nx.is_directed_acyclic_graph(graph):
        graph, cycle_stats = await run_blocking_stage(
            make_dag_with_llm,
            graph,
            stage_name="graph_cycle_removal",
        )
        job.partial_graph = build_partial_graph(graph, concepts)

    if apply_reduction:
        graph = await run_blocking_stage(
            apply_transitive_reduction,
            graph,
            stage_name="graph_reduction",
        )
        job.partial_graph = build_partial_graph(graph, concepts)

    await persist_job_state(
        job,
        PipelineStatus.OPTIMIZING,
        "Removing cycles, building DAG...",
        0.95,
    )

    return graph, {
        "num_nodes": graph.number_of_nodes(),
        "num_edges": graph.number_of_edges(),
        "is_dag": nx.is_directed_acyclic_graph(graph),
        "cycle_stats": cycle_stats,
    }
