from __future__ import annotations

import asyncio
from types import SimpleNamespace

import networkx as nx

import api.domains.content_pipeline.infrastructure.embed as embed_module
from api.domains.content_pipeline.infrastructure.embed import compute_embeddings_batch
from api.domains.content_pipeline.infrastructure.graph import cycle_removal
from api.domains.content_pipeline.infrastructure.graph.cycle_removal import (
    CycleRemover,
    make_dag_with_llm,
)
from api.domains.content_pipeline.infrastructure.llm.schemas import (
    CycleRemovalDecision,
    EdgeDecision,
)
import api.domains.content_pipeline.infrastructure.utils.text as text_utils


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


async def fake_cycle_removal_decision(self, cycle_info):
    del self, cycle_info
    return CycleRemovalDecision(
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
    original_cycle_logger_info = cycle_removal.logger.info
    original_cycle_logger_debug = cycle_removal.logger.debug
    original_invoke_cycle_removal_decision = CycleRemover._invoke_cycle_removal_decision

    embed_module.EmbeddingClient = FakeEmbeddingClient
    embed_module.settings = SimpleNamespace(embedding_model="model-x")
    embed_module.logger.debug = lambda *args, **kwargs: None
    cycle_removal.logger.info = lambda *args, **kwargs: None
    cycle_removal.logger.debug = lambda *args, **kwargs: None
    CycleRemover._invoke_cycle_removal_decision = fake_cycle_removal_decision
    text_utils._text_normalize = None

    try:
        for _ in range(500):
            text_utils.clean_text("ﬁle   nội   dung!!")

        for _ in range(200):
            compute_embeddings_batch(["one two three", "alpha beta"], batch_size=8, max_length=2)

        for _ in range(10):
            asyncio.run(make_dag_with_llm(build_cycle_graph()))
    finally:
        embed_module.EmbeddingClient = original_embedding_client
        embed_module.settings = original_settings
        embed_module.logger.debug = original_debug
        cycle_removal.logger.info = original_cycle_logger_info
        cycle_removal.logger.debug = original_cycle_logger_debug
        CycleRemover._invoke_cycle_removal_decision = original_invoke_cycle_removal_decision


if __name__ == "__main__":
    main()
