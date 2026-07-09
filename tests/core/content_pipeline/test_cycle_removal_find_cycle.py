import asyncio

import networkx as nx
import pytest

from api.domains.content_pipeline.infrastructure.graph import cycle_removal as cycle_removal_module
from api.domains.content_pipeline.infrastructure.graph.cycle_removal import CycleRemover
from api.domains.content_pipeline.infrastructure.llm.schemas import (
    CycleRemovalDecision,
    EdgeDecision,
)


def test_cycle_remover_uses_find_cycle_for_three_node_cycle(monkeypatch):
    graph = nx.DiGraph()
    graph.add_node("a", name="A", definition="Khái niệm A")
    graph.add_node("b", name="B", definition="Khái niệm B")
    graph.add_node("c", name="C", definition="Khái niệm C")
    graph.add_edge("a", "b", relation_type="PREREQUISITE")
    graph.add_edge("b", "c", relation_type="PREREQUISITE")
    graph.add_edge("c", "a", relation_type="PREREQUISITE")

    decision = CycleRemovalDecision(
        cycle_nodes=["a", "b", "c"],
        edges_to_remove=[
            EdgeDecision(
                source_id="c",
                target_id="a",
                should_remove=True,
                reasoning="Cạnh đóng vòng nên loại bỏ.",
                confidence=0.9,
            )
        ],
        reasoning="Giữ A -> B -> C.",
    )

    original_find_cycle = nx.find_cycle
    find_cycle_calls = {"count": 0}

    def wrapped_find_cycle(*args, **kwargs):
        find_cycle_calls["count"] += 1
        return original_find_cycle(*args, **kwargs)

    def fail_simple_cycles(*_args, **_kwargs):
        raise AssertionError("simple_cycles should not be used")

    monkeypatch.setattr(nx, "find_cycle", wrapped_find_cycle)
    monkeypatch.setattr(nx, "simple_cycles", fail_simple_cycles)

    async def fake_ainvoke_structured_completion(**kwargs):
        assert kwargs["schema"] is CycleRemovalDecision
        return decision

    patcher = pytest.MonkeyPatch()
    patcher.setattr(
        cycle_removal_module,
        "ainvoke_structured_completion",
        fake_ainvoke_structured_completion,
    )
    try:
        remover = CycleRemover()
        dag, stats = asyncio.run(remover.remove_cycles(graph))
    finally:
        patcher.undo()

    assert find_cycle_calls["count"] >= 1
    assert nx.is_directed_acyclic_graph(dag)
    assert dag.has_edge("a", "b")
    assert dag.has_edge("b", "c")
    assert not dag.has_edge("c", "a")
    assert stats["edges_removed"] == 1
    assert stats["iterations"] <= 3
