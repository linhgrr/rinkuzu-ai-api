from __future__ import annotations

import asyncio
from types import SimpleNamespace

import networkx as nx

from api.domains.content_pipeline.infrastructure.embed import compute_embeddings_batch
from api.domains.content_pipeline.infrastructure.graph import cycle_removal as cycle_removal_module
from api.domains.content_pipeline.infrastructure.graph.cycle_removal import make_dag_with_llm
from api.domains.content_pipeline.infrastructure.llm.schemas import (
    CycleRemovalDecision,
    EdgeDecision,
)
import api.domains.content_pipeline.infrastructure.utils.text as text_utils


def test_clean_text_smoke(monkeypatch, benchmark):
    monkeypatch.setattr(text_utils, "_text_normalize", None)
    result = benchmark(text_utils.clean_text, "ﬁle   nội   dung!!")
    assert result == "file nội dung!!"


def test_compute_embeddings_batch_smoke(monkeypatch, benchmark):
    recorded_texts: list[str] = []

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
            recorded_texts.extend(texts)
            return [[float(len(text))] for text in texts]

    monkeypatch.setattr(
        "api.domains.content_pipeline.infrastructure.embed.EmbeddingClient",
        FakeEmbeddingClient,
    )
    monkeypatch.setattr(
        "api.domains.content_pipeline.infrastructure.embed.settings",
        SimpleNamespace(embedding_model="model-x"),
    )
    monkeypatch.setattr(
        "api.domains.content_pipeline.infrastructure.embed.logger.debug",
        lambda *args, **kwargs: None,
    )

    result = benchmark(
        compute_embeddings_batch,
        ["one two three", "alpha beta"],
        batch_size=8,
        max_length=2,
    )

    assert len(recorded_texts) >= 2
    assert set(recorded_texts) == {"tok0 tok1"}
    assert result == [[9.0], [9.0]]


def test_make_dag_with_llm_smoke(monkeypatch, benchmark):
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

    async def fake_ainvoke_structured_completion(**kwargs):
        assert kwargs["schema"] is CycleRemovalDecision
        return decision

    monkeypatch.setattr(
        cycle_removal_module,
        "ainvoke_structured_completion",
        fake_ainvoke_structured_completion,
    )

    def run_once():
        return asyncio.run(make_dag_with_llm(graph))

    dag, stats = benchmark(run_once)

    assert nx.is_directed_acyclic_graph(dag)
    assert stats["is_dag"] is True
