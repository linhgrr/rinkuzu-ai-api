from langchain_core.runnables import RunnableLambda
import networkx as nx

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

    class FakeLLM:
        def with_structured_output(self, schema, *, method="json_schema", strict=True, **kwargs):
            assert schema is CycleRemovalDecision
            assert method == "json_schema"
            assert strict is True
            return RunnableLambda(lambda _: decision)

    remover = CycleRemover(llm=FakeLLM())
    dag, stats = remover.remove_cycles(graph)

    assert nx.is_directed_acyclic_graph(dag)
    assert dag.has_edge("a", "b")
    assert not dag.has_edge("b", "a")
    assert stats["edges_removed"] == 1
