"""Graph optimization stage for the content pipeline."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any

import networkx as nx

from api.config import get_settings
from api.core.content_pipeline.domain.errors import PipelineStageTimeoutError
from api.core.content_pipeline.domain.jobs import PipelineJob, PipelineProgress, PipelineStatus

from .execution import run_blocking_stage
from .graph_building import build_partial_graph

PersistJobStateFn = Callable[[PipelineJob, PipelineStatus, str, float], Awaitable[None]]


def _resolve_cycle_removal_timeout() -> float | None:
    settings = get_settings()
    return float(settings.content_pipeline_graph_cycle_timeout_sec)


async def optimize_graph(
    job: PipelineJob,
    *,
    graph: Any,
    concepts: list[Any],
    apply_reduction: bool,
    make_dag_with_llm: Callable[[Any], Awaitable[tuple[Any, Any]]],
    apply_transitive_reduction: Callable[[Any], Any],
    persist_job_state: PersistJobStateFn,
) -> tuple[Any, dict[str, Any]]:
    """Make the graph a DAG, optionally apply transitive reduction, and derive stats."""
    await persist_job_state(
        job,
        PipelineStatus.OPTIMIZING,
        "Removing cycles, building DAG...",
        PipelineProgress.GRAPH_OPTIMIZATION_START,
    )

    cycle_stats: dict[str, Any] | None = None
    if not nx.is_directed_acyclic_graph(graph):
        try:
            graph, cycle_stats = await asyncio.wait_for(
                make_dag_with_llm(graph),
                timeout=_resolve_cycle_removal_timeout(),
            )
        except TimeoutError as exc:
            raise PipelineStageTimeoutError(
                "graph_cycle_removal",
                float(_resolve_cycle_removal_timeout() or 0.0),
            ) from exc
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
        PipelineProgress.GRAPH_OPTIMIZATION_DONE,
    )

    return graph, {
        "num_nodes": graph.number_of_nodes(),
        "num_edges": graph.number_of_edges(),
        "is_dag": nx.is_directed_acyclic_graph(graph),
        "cycle_stats": cycle_stats,
    }
