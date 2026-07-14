import asyncio
from pathlib import Path
import sys
from types import SimpleNamespace

from api.domains.content_pipeline.infrastructure import runtime
from api.domains.content_pipeline.infrastructure.runtime import calculate_file_hash


def test_calculate_file_hash_is_stable_for_same_content(tmp_path: Path):
    file_path = tmp_path / "lesson.txt"
    file_path.write_text("algebra", encoding="utf-8")

    first_hash = calculate_file_hash(str(file_path))
    second_hash = calculate_file_hash(str(file_path))

    assert first_hash == second_hash
    assert len(first_hash) == 64


def test_get_content_processor_bindings_uses_cached_bindings():
    runtime.get_content_processor_bindings.cache_clear()
    first = runtime.get_content_processor_bindings()

    second = runtime.get_content_processor_bindings()

    assert second is first


def test_content_processor_bindings_can_build_relation_engine(monkeypatch):
    class _ExtractionChain:
        def verify_relations_batch(self, pairs):
            return pairs

    class _StubRanker:
        @classmethod
        def load(cls, *_args, **_kwargs):
            instance = cls()
            instance.rank = lambda concepts, threshold: []
            return instance

    monkeypatch.setattr(runtime, "_MLPPrerequisiteRanker", _StubRanker)
    monkeypatch.setattr(runtime, "_MLP_RANKING_AVAILABLE", True)

    bindings = runtime.get_content_processor_bindings()
    engine = bindings.relation_engine_factory(extraction_chain=_ExtractionChain())

    assert hasattr(engine, "discover_relations")


def test_build_embedding_client_passes_model_name_and_batch_size(monkeypatch):
    module_name = "api.domains.content_pipeline.infrastructure.embed.embedding_client"
    original_module = sys.modules.get(module_name)

    class _EmbeddingClient:
        def __init__(self, model_name, *, batch_size=None):
            self.model_name = model_name
            self.batch_size = batch_size

    monkeypatch.setitem(sys.modules, module_name, SimpleNamespace(EmbeddingClient=_EmbeddingClient))

    try:
        client = runtime._build_embedding_client("keepitreal/vietnamese-sbert", 32)
    finally:
        if original_module is None:
            sys.modules.pop(module_name, None)
        else:
            sys.modules[module_name] = original_module

    assert client.model_name == "keepitreal/vietnamese-sbert"
    assert client.batch_size == 32


def test_runtime_root_no_longer_points_to_content_processor_folder():
    assert runtime.PROJECT_ROOT.name == "rinkuzu-ai-api"
    assert "content-processor" not in runtime.CONTENT_PROCESSOR_SRC


def test_generate_theory_helper_imports_exercise_gen_lazily(monkeypatch):
    module_name = "api.domains.learning.exercise_gen"
    original_module = sys.modules.get(module_name)

    async def _fake_generate_theory(name, definition):
        return {"content": f"{name}: {definition}", "examples": []}

    fake_module = SimpleNamespace(generate_theory=_fake_generate_theory)
    monkeypatch.setitem(sys.modules, module_name, fake_module)

    try:
        result = asyncio.run(
            runtime._generate_theory_via_exercise_gen("Algebra", "Sets and equations")
        )
    finally:
        if original_module is None:
            sys.modules.pop(module_name, None)
        else:
            sys.modules[module_name] = original_module

    assert result["content"] == "Algebra: Sets and equations"
