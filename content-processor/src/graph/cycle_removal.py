"""LLM-based cycle removal for knowledge graphs."""

from typing import List, Tuple, Dict, Any
import networkx as nx
from loguru import logger

from llm import get_llm
from llm.schemas import CycleRemovalDecision
from prompts import CYCLE_REMOVAL_PROMPT
from langchain_core.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate
)
from langchain.agents import create_agent


class CycleRemover:
    """LLM-based cycle remover for knowledge graphs."""

    def __init__(self, llm = None):
        """
        Initialize cycle remover with LLM.

        Args:
            llm: ChatOpenAI instance
        """
        if llm is None:
            self.llm = get_llm(
                temperature=0.1,
                max_tokens=None,
                timeout=None
            )
        else:
            self.llm = llm

        self.cycle_removal_prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(CYCLE_REMOVAL_PROMPT),
            HumanMessagePromptTemplate.from_template(
                "## CYCLE INFORMATION\n\n{cycle_info}"
            )
        ])

        self.cycle_removal_agent = create_agent(
            self.llm,
            tools=[],
            response_format=CycleRemovalDecision
        )

        self.cycle_removal_chain = self.cycle_removal_prompt | self.cycle_removal_agent

        logger.info("CycleRemover initialized with LLM-based decision making")

    def remove_cycles(self, graph: nx.DiGraph) -> Tuple[nx.DiGraph, Dict[str, Any]]:
        """
        Remove all cycles from graph using LLM decisions.

        Args:
            graph: Knowledge graph (may contain cycles)

        Returns:
            Tuple of (dag_graph, stats_dict)
            - dag_graph: Graph with cycles removed
            - stats_dict: Statistics about cycle removal
        """
        if nx.is_directed_acyclic_graph(graph):
            logger.info("Graph is already a DAG, no cycles to remove")
            return graph, {
                'had_cycles': False,
                'cycles_removed': 0,
                'edges_removed': 0,
                'iterations': 0
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
                cycles = list(nx.simple_cycles(dag_graph))
            except Exception as e:
                logger.error(f"Error finding cycles: {e}")
                break

            if not cycles:
                break

            logger.info(f"Iteration {iteration}: Found {len(cycles)} cycle(s)")

            cycle = cycles[0]

            cycle_edges = []
            for i in range(len(cycle)):
                source = cycle[i]
                target = cycle[(i + 1) % len(cycle)]
                if dag_graph.has_edge(source, target):
                    edge_data = dag_graph[source][target]
                    cycle_edges.append({
                        'source': source,
                        'target': target,
                        'data': edge_data
                    })

            edges_removed = self._remove_cycle_with_llm(dag_graph, cycle, cycle_edges)

            total_edges_removed += edges_removed
            cycles_removed += 1

            logger.info(f"Removed {edges_removed} edge(s) from cycle")

        is_dag = nx.is_directed_acyclic_graph(dag_graph)

        if not is_dag:
            logger.warning(
                f"Graph still has cycles after {iteration} iterations. "
                f"Consider manual intervention."
            )

        stats = {
            'had_cycles': True,
            'cycles_removed': cycles_removed,
            'edges_removed': total_edges_removed,
            'iterations': iteration,
            'is_dag': is_dag
        }

        logger.info(
            f"Cycle removal complete: {cycles_removed} cycles, "
            f"{total_edges_removed} edges removed in {iteration} iterations"
        )

        return dag_graph, stats

    def _remove_cycle_with_llm(
        self,
        graph: nx.DiGraph,
        cycle: List[str],
        cycle_edges: List[Dict[str, Any]]
    ) -> int:
        """
        Use LLM to decide which edges to remove from a cycle.

        Args:
            graph: The graph to modify
            cycle: List of node IDs forming the cycle
            cycle_edges: List of edge dictionaries with source, target, data

        Returns:
            Number of edges removed
        """
        try:
            cycle_info = self._format_cycle_info(graph, cycle, cycle_edges)
            logger.debug(f"Asking LLM to analyze cycle: {' → '.join(cycle)}")
            
            logger.info(f"Cycle removal LLM Input: {cycle_info}")

            response = self.cycle_removal_chain.invoke({
                "cycle_info": cycle_info
            })
            
            logger.info(f"Cycle removal LLM Output: {response}")

            decision: CycleRemovalDecision = response["structured_response"]

            edges_removed = 0
            for edge_decision in decision.edges_to_remove:
                if edge_decision.should_remove:
                    source = edge_decision.source_id
                    target = edge_decision.target_id

                    if graph.has_edge(source, target):
                        logger.info(
                            f"Removing edge: {source} → {target} "
                            f"(confidence: {edge_decision.confidence:.2f})"
                        )
                        logger.debug(f"Reasoning: {edge_decision.reasoning}")

                        graph.remove_edge(source, target)
                        edges_removed += 1

            if edges_removed == 0:
                logger.warning("LLM didn't remove any edges, removing first edge as fallback")
                if cycle_edges:
                    first_edge = cycle_edges[0]
                    graph.remove_edge(first_edge['source'], first_edge['target'])
                    edges_removed = 1

            return edges_removed

        except Exception as e:
            logger.error(f"Error in LLM cycle removal: {e}")
            if cycle_edges:
                first_edge = cycle_edges[0]
                logger.warning(f"Fallback: removing {first_edge['source']} → {first_edge['target']}")
                graph.remove_edge(first_edge['source'], first_edge['target'])
                return 1
            return 0

    def _format_cycle_info(
        self,
        graph: nx.DiGraph,
        cycle: List[str],
        cycle_edges: List[Dict[str, Any]]
    ) -> str:
        """
        Format cycle information for LLM prompt.

        Args:
            graph: The graph
            cycle: List of node IDs in cycle
            cycle_edges: List of edge dictionaries

        Returns:
            Formatted string with cycle information
        """
        lines = []

        # Cycle overview
        lines.append(f"**Cycle Path**: {' → '.join(cycle)} → {cycle[0]}\n")

        # Node details
        lines.append("## Concepts in Cycle\n")
        for node_id in cycle:
            node_data = graph.nodes.get(node_id, {})
            name = node_data.get('name', node_id)
            definition = node_data.get('definition', '')

            lines.append(f"### {name} (ID: {node_id})")
            if definition:
                lines.append(f"Definition: {definition[:200]}...")
            lines.append("")

        # Edge details
        lines.append("## Edges in Cycle\n")
        for edge in cycle_edges:
            source = edge['source']
            target = edge['target']
            edge_data = edge['data']

            source_name = graph.nodes.get(source, {}).get('name', source)
            target_name = graph.nodes.get(target, {}).get('name', target)

            lines.append(f"### {source_name} → {target_name}")
            lines.append(f"- Source ID: {source}")
            lines.append(f"- Target ID: {target}")
            lines.append(f"- Relation Type: {edge_data.get('relation_type', 'UNKNOWN')}")

            evidence = edge_data.get('evidence')
            if evidence:
                if isinstance(evidence, list):
                    lines.append(f"- Evidence: {evidence[0][:150]}..." if evidence else "")
                else:
                    lines.append(f"- Evidence: {str(evidence)[:150]}...")

            lines.append("")

        return "\n".join(lines)


def make_dag_with_llm(
    graph: nx.DiGraph,
    llm=None
) -> Tuple[nx.DiGraph, Dict[str, Any]]:
    """
    Convert graph to DAG by removing cycles using LLM decisions.

    This is a convenience function that creates a CycleRemover and uses it.

    Args:
        graph: Knowledge graph (may contain cycles)
        llm: Optional LLM instance

    Returns:
        Tuple of (dag_graph, stats)
    """
    remover = CycleRemover(llm=llm)
    return remover.remove_cycles(graph)

