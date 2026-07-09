"""LLM-based cycle removal for knowledge graphs."""

from __future__ import annotations

from typing import Any

from loguru import logger
import networkx as nx

from api.domains.content_pipeline.infrastructure.llm.schemas import CycleRemovalDecision
from api.domains.content_pipeline.infrastructure.prompts import CYCLE_REMOVAL_PROMPT
from api.shared.llm import ainvoke_structured_completion
from api.shared.retry import llm_async_retry


class CycleRemover:
    """LLM-based cycle remover for knowledge graphs."""

    def __init__(self, model: str | None = None) -> None:
        self.model = model
        logger.info("CycleRemover initialized with shared structured output")

    async def remove_cycles(self, graph: nx.DiGraph) -> tuple[nx.DiGraph, dict[str, Any]]:
        """Remove all cycles from graph using LLM decisions."""
        if nx.is_directed_acyclic_graph(graph):
            logger.info("Graph is already a DAG, no cycles to remove")
            return graph, {
                "had_cycles": False,
                "cycles_removed": 0,
                "edges_removed": 0,
                "iterations": 0,
            }

        logger.info("Graph contains cycles, starting LLM-based removal")

        dag_graph = graph.copy()
        total_edges_removed = 0
        cycles_removed = 0
        iteration = 0
        max_iterations = 100

        while not nx.is_directed_acyclic_graph(dag_graph) and iteration < max_iterations:
            iteration += 1

            try:
                cycle_edges_raw = list(nx.find_cycle(dag_graph, orientation="original"))
            except nx.NetworkXNoCycle:
                logger.info("No cycle found during iteration {}", iteration)
                break
            except Exception:
                logger.exception("Error finding cycles")
                break

            if not cycle_edges_raw:
                break

            logger.info(
                "Iteration {}: Found cycle with {} edge(s)",
                iteration,
                len(cycle_edges_raw),
            )

            cycle = [cycle_edges_raw[0][0], *[edge[1] for edge in cycle_edges_raw[:-1]]]
            cycle_edges = []
            for edge in cycle_edges_raw:
                source = edge[0]
                target = edge[1]
                if dag_graph.has_edge(source, target):
                    cycle_edges.append(
                        {"source": source, "target": target, "data": dag_graph[source][target]}
                    )

            edges_removed = await self._remove_cycle_with_llm(dag_graph, cycle, cycle_edges)
            total_edges_removed += edges_removed
            cycles_removed += 1
            logger.info("Removed {} edge(s) from cycle", edges_removed)

        is_dag = nx.is_directed_acyclic_graph(dag_graph)
        if not is_dag:
            logger.warning(
                "Graph still has cycles after {} iterations. Consider manual intervention.",
                iteration,
            )

        stats = {
            "had_cycles": True,
            "cycles_removed": cycles_removed,
            "edges_removed": total_edges_removed,
            "iterations": iteration,
            "is_dag": is_dag,
        }

        logger.info(
            "Cycle removal complete: {} cycles, {} edges removed in {} iterations",
            cycles_removed,
            total_edges_removed,
            iteration,
        )
        return dag_graph, stats

    async def _remove_cycle_with_llm(
        self,
        graph: nx.DiGraph,
        cycle: list[str],
        cycle_edges: list[dict[str, Any]],
    ) -> int:
        result = 0
        try:
            cycle_info = self._format_cycle_info(graph, cycle, cycle_edges)
            logger.debug("Asking LLM to analyze cycle: {}", " → ".join([*cycle, cycle[0]]))
            logger.info("Cycle removal LLM Input: {}", cycle_info)
            decision = await self._invoke_cycle_removal_decision(cycle_info)
            logger.info("Cycle removal LLM Output: {}", decision.model_dump())

            edges_removed = 0
            for edge_decision in decision.edges_to_remove:
                if not edge_decision.should_remove:
                    continue
                source = edge_decision.source_id
                target = edge_decision.target_id
                if graph.has_edge(source, target):
                    logger.info(
                        "Removing edge: {} → {} (confidence: {:.2f})",
                        source,
                        target,
                        edge_decision.confidence,
                    )
                    logger.debug("Reasoning: {}", edge_decision.reasoning)
                    graph.remove_edge(source, target)
                    edges_removed += 1

            if edges_removed == 0 and cycle_edges:
                logger.warning("LLM didn't remove any edges, removing first edge as fallback")
                first_edge = cycle_edges[0]
                graph.remove_edge(first_edge["source"], first_edge["target"])
                result = 1
            else:
                result = edges_removed
        except Exception:
            logger.exception("Error in LLM cycle removal")
            if cycle_edges:
                first_edge = cycle_edges[0]
                logger.warning(
                    "Fallback: removing {} → {}",
                    first_edge["source"],
                    first_edge["target"],
                )
                graph.remove_edge(first_edge["source"], first_edge["target"])
                result = 1
        return result

    @llm_async_retry(label="cycle removal")
    async def _invoke_cycle_removal_decision(self, cycle_info: str) -> CycleRemovalDecision:
        return await ainvoke_structured_completion(
            model=self.model,
            temperature=0.1,
            max_tokens=None,
            schema=CycleRemovalDecision,
            messages=[
                {"role": "system", "content": CYCLE_REMOVAL_PROMPT},
                {"role": "user", "content": f"## CYCLE INFORMATION\n\n{cycle_info}"},
            ],
        )

    def _format_cycle_info(
        self,
        graph: nx.DiGraph,
        cycle: list[str],
        cycle_edges: list[dict[str, Any]],
    ) -> str:
        lines = [f"**Cycle Path**: {' → '.join(cycle)} → {cycle[0]}\n", "## Concepts in Cycle\n"]

        for node_id in cycle:
            node_data = graph.nodes.get(node_id, {})
            name = node_data.get("name", node_id)
            definition = node_data.get("definition", "")
            lines.append(f"### {name} (ID: {node_id})")
            if definition:
                lines.append(f"Definition: {definition[:200]}...")
            lines.append("")

        lines.append("## Edges in Cycle\n")
        for edge in cycle_edges:
            source = edge["source"]
            target = edge["target"]
            edge_data = edge["data"]
            source_name = graph.nodes.get(source, {}).get("name", source)
            target_name = graph.nodes.get(target, {}).get("name", target)

            lines.append(f"### {source_name} → {target_name}")
            lines.append(f"- Source ID: {source}")
            lines.append(f"- Target ID: {target}")
            lines.append(f"- Relation Type: {edge_data.get('relation_type', 'UNKNOWN')}")

            evidence = edge_data.get("evidence")
            if evidence:
                if isinstance(evidence, list):
                    lines.append(f"- Evidence: {evidence[0][:150]}..." if evidence else "")
                else:
                    lines.append(f"- Evidence: {str(evidence)[:150]}...")

            lines.append("")

        return "\n".join(lines)


async def make_dag_with_llm(
    graph: nx.DiGraph,
    model: str | None = None,
) -> tuple[nx.DiGraph, dict[str, Any]]:
    """Convert graph to DAG by removing cycles using LLM decisions."""
    remover = CycleRemover(model=model)
    return await remover.remove_cycles(graph)
