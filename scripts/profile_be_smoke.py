from __future__ import annotations

import asyncio
from types import SimpleNamespace

from langchain_core.runnables import RunnableLambda
import networkx as nx

import api.core.content_pipeline.infrastructure.embed as embed_module
from api.core.content_pipeline.infrastructure.embed import compute_embeddings_batch
from api.core.content_pipeline.infrastructure.graph.cycle_removal import make_dag_with_llm
from api.core.content_pipeline.infrastructure.llm.schemas import CycleRemovalDecision, EdgeDecision
import api.core.content_pipeline.infrastructure.utils.text as text_utils


class FakeTokenizer:
    def encode(self, text, *, max_length, truncation, add_special_tokens):
        assert truncation is True
        assert add_special_tokens is False
        return list(range(min(len(text.split()), max_length)))

    def decode(self, token_ids):
        return " ".join(f"tok{index}" for index in token_ids)


class FakeEmbeddingClient:
    def __init__(self, *args, **kwargs):
        self._model_handle = SimpleNamespace(model=SimpleNamespace(tokenizer=FakeTokenizer()))

    def embed_documents(self, texts):
        return [[float(len(text))] for text in texts]


class FakeLLM:
    def with_structured_output(self, schema, *, method="json_schema", strict=True, **kwargs):
        decision = CycleRemovalDecision(
            cycle_nodes=["a", "b"],
            edges_to_remove=[
                EdgeDecision(
                    source_id="b",
                    target_id="a",
                    should_remove=True,
                    reasoning="B depends on A, drop reverse edge.",
                    confidence=0.9,
                )
            ],
            reasoning="Keep A -> B.",
        )
        return RunnableLambda(lambda _: decision)


def build_cycle_graph():
    graph = nx.DiGraph()
    graph.add_node("a", name="A")
    graph.add_node("b", name="B")
    graph.add_edge("a", "b", relation_type="PREREQUISITE")
    graph.add_edge("b", "a", relation_type="PREREQUISITE")
    return graph


def main() -> None:
    original_embedding_client = embed_module.EmbeddingClient
    original_settings = embed_module.settings
    original_debug = embed_module.logger.debug

    embed_module.EmbeddingClient = FakeEmbeddingClient
    embed_module.settings = SimpleNamespace(embedding_model="model-x")
    embed_module.logger.debug = lambda *args, **kwargs: None
    text_utils._text_normalize = None

    try:
        for _ in range(500):
            text_utils.clean_text("ﬁle   nội   dung!!")

        for _ in range(200):
            compute_embeddings_batch(["one two three", "alpha beta"], batch_size=8, max_length=2)

        for _ in range(10):
            asyncio.run(make_dag_with_llm(build_cycle_graph(), llm=FakeLLM()))
    finally:
        embed_module.EmbeddingClient = original_embedding_client
        embed_module.settings = original_settings
        embed_module.logger.debug = original_debug


if __name__ == "__main__":
    main()
