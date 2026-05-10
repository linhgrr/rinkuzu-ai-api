"""Transitive reduction for knowledge graphs."""

from loguru import logger
import networkx as nx


def apply_transitive_reduction(graph: nx.DiGraph) -> nx.DiGraph:
    """
    Apply transitive reduction to remove redundant edges.

    Args:
        graph: Knowledge graph

    Returns:
        Graph with transitive reduction applied
    """
    prereq_graph = _extract_prerequisite_subgraph(graph)

    if prereq_graph.number_of_edges() == 0:
        logger.info("No prerequisite edges to reduce")
        return graph

    try:
        reduced_prereq = nx.transitive_reduction(prereq_graph)
    except Exception:
        logger.exception("Error applying transitive reduction")
        return graph

    removed_edges = _find_removed_edges(prereq_graph, reduced_prereq)

    logger.info(
        "Transitive reduction complete",
        original_edges=prereq_graph.number_of_edges(),
        reduced_edges=reduced_prereq.number_of_edges(),
        removed=len(removed_edges),
    )

    return _rebuild_graph(graph, reduced_prereq, prereq_graph)


def _extract_prerequisite_subgraph(graph: nx.DiGraph) -> nx.DiGraph:
    """Extract only PREREQUISITE edges."""
    subgraph = nx.DiGraph()

    subgraph.add_nodes_from(graph.nodes(data=True))

    for u, v, data in graph.edges(data=True):
        if data.get("relation_type") == "PREREQUISITE":
            subgraph.add_edge(u, v, **data)

    return subgraph


def _find_removed_edges(
    original: nx.DiGraph,
    reduced: nx.DiGraph,
) -> list[tuple[str, str]]:
    """Find edges that were removed."""
    original_edges = set(original.edges())
    reduced_edges = set(reduced.edges())

    return list(original_edges - reduced_edges)


def _rebuild_graph(
    original: nx.DiGraph,
    reduced_prereq: nx.DiGraph,
    original_prereq: nx.DiGraph,
) -> nx.DiGraph:
    """Rebuild graph with reduced prerequisites."""
    result = nx.DiGraph()

    # Copy all nodes
    result.add_nodes_from(original.nodes(data=True))

    # Add non-prerequisite edges from original
    for u, v, data in original.edges(data=True):
        if data.get("relation_type") != "PREREQUISITE":
            result.add_edge(u, v, **data)

    # Add reduced prerequisite edges (with original data)
    for u, v in reduced_prereq.edges():
        # Get original edge data
        original_data = original_prereq[u][v]
        result.add_edge(u, v, **original_data)

    return result
