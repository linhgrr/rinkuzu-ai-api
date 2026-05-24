import asyncio

import networkx as nx
import pytest

from api.core.content_pipeline.infrastructure.graph import cycle_removal as cycle_removal_module
from api.core.content_pipeline.infrastructure.graph.cycle_removal import CycleRemover
from api.core.content_pipeline.infrastructure.llm.schemas import CycleRemovalDecision, EdgeDecision


def test_cycle_remover_uses_langchain_structured_output():
    graph = nx.DiGraph()
    graph.add_node("a", name="A", definition="Khái niệm A")
    graph.add_node("b", name="B", definition="Khái niệm B")
    graph.add_edge("a", "b", relation_type="PREREQUISITE")
    graph.add_edge("b", "a", relation_type="PREREQUISITE")

    decision = CycleRemovalDecision(
        cycle_nodes=["a", "b"],
        edges_to_remove=[
            EdgeDecision(
                source_id="b",
                target_id="a",
                should_remove=True,
                reasoning="B phụ thuộc A nên cạnh ngược nên bỏ.",
                confidence=0.9,
            )
        ],
        reasoning="Giữ hướng A -> B để bảo toàn prerequisite chính.",
    )

    async def fake_ainvoke_structured_completion(**kwargs):
        assert kwargs["schema"] is CycleRemovalDecision
        return decision

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(
        cycle_removal_module,
        "ainvoke_structured_completion",
        fake_ainvoke_structured_completion,
    )

    try:
        remover = CycleRemover()
        dag, stats = asyncio.run(remover.remove_cycles(graph))
    finally:
        monkeypatch.undo()

    assert nx.is_directed_acyclic_graph(dag)
    assert dag.has_edge("a", "b")
    assert not dag.has_edge("b", "a")
    assert stats["edges_removed"] == 1


def test_make_dag_with_llm_is_async_compatible():
    graph = nx.DiGraph()
    graph.add_node("a", name="A")
    graph.add_node("b", name="B")
    graph.add_edge("a", "b", relation_type="PREREQUISITE")
    graph.add_edge("b", "a", relation_type="PREREQUISITE")

    decision = CycleRemovalDecision(
        cycle_nodes=["a", "b"],
        edges_to_remove=[
            EdgeDecision(
                source_id="b",
                target_id="a",
                should_remove=True,
                reasoning="B phụ thuộc A nên bỏ cạnh ngược.",
                confidence=0.9,
            )
        ],
        reasoning="Giữ A -> B.",
    )

    from api.core.content_pipeline.infrastructure.graph.cycle_removal import make_dag_with_llm

    async def fake_ainvoke_structured_completion(**kwargs):
        assert kwargs["schema"] is CycleRemovalDecision
        return decision

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(
        cycle_removal_module,
        "ainvoke_structured_completion",
        fake_ainvoke_structured_completion,
    )
    try:
        dag, stats = asyncio.run(make_dag_with_llm(graph))
    finally:
        monkeypatch.undo()

    assert nx.is_directed_acyclic_graph(dag)
    assert stats["is_dag"] is True
