import importlib
import sys
from types import ModuleType, SimpleNamespace

pyvi_module = ModuleType("pyvi")
pyvi_module.ViTokenizer = SimpleNamespace(tokenize=lambda text: text)
sys.modules.setdefault("pyvi", pyvi_module)

sentence_transformers_module = ModuleType("sentence_transformers")


class _FakeSentenceTransformer:
    def __init__(self, *args, **kwargs):
        pass


sentence_transformers_module.SentenceTransformer = _FakeSentenceTransformer
sys.modules.setdefault("sentence_transformers", sentence_transformers_module)

torch_module = ModuleType("torch")
torch_module.cuda = SimpleNamespace(is_available=lambda: False)
sys.modules.setdefault("torch", torch_module)


def test_compute_embeddings_batch_truncates_with_tokenizer(monkeypatch):
    embed_module = importlib.import_module("api.core.content_pipeline.infrastructure.embed")
    recorded_texts = []

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

    monkeypatch.setattr(embed_module, "EmbeddingClient", FakeEmbeddingClient)
    monkeypatch.setattr(
        embed_module,
        "settings",
        SimpleNamespace(embedding_model="model-x"),
    )

    result = embed_module.compute_embeddings_batch(
        ["one two three four", "", "alpha beta"],
        batch_size=8,
        max_length=2,
    )

    assert recorded_texts == ["tok0 tok1", "", "tok0 tok1"]
    assert result == [[9.0], [0.0], [9.0]]
