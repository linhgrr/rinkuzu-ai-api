import sys
from types import SimpleNamespace
from pathlib import Path

from api.core.content_pipeline.infrastructure import runtime
from api.core.content_pipeline.infrastructure.runtime import calculate_file_hash


def test_calculate_file_hash_is_stable_for_same_content(tmp_path: Path):
    file_path = tmp_path / "lesson.txt"
    file_path.write_text("algebra", encoding="utf-8")

    first_hash = calculate_file_hash(str(file_path))
    second_hash = calculate_file_hash(str(file_path))

    assert first_hash == second_hash
    assert len(first_hash) == 64


def test_get_content_processor_bindings_uses_cached_bindings(monkeypatch):
    sentinel = object()
    monkeypatch.setattr(runtime, "_content_processor_bindings", sentinel)

    bindings = runtime.get_content_processor_bindings()

    assert bindings is sentinel


def test_get_content_processor_llm_factory_uses_cached_factory(monkeypatch):
    sentinel = object()
    monkeypatch.setattr(runtime, "_content_processor_llm_factory", sentinel)

    factory = runtime.get_content_processor_llm_factory()

    assert factory is sentinel


def test_content_processor_bindings_can_build_relation_engine():
    class _ExtractionChain:
        def verify_relations_batch(self, pairs):
            return pairs

    bindings = runtime.get_content_processor_bindings()
    engine = bindings.relation_engine_factory(extraction_chain=_ExtractionChain())

    assert hasattr(engine, "discover_relations")


def test_runtime_root_no_longer_points_to_content_processor_folder():
    assert runtime.PROJECT_ROOT.name == "rinkuzu-ai-api"
    assert "content-processor" not in runtime.CONTENT_PROCESSOR_SRC


def test_generate_theory_helper_imports_exercise_gen_lazily(monkeypatch):
    module_name = "api.core.exercise_gen"
    original_module = sys.modules.get(module_name)
    fake_module = SimpleNamespace(
        generate_theory=lambda name, definition: {
            "content": f"{name}: {definition}",
            "examples": [],
        },
    )
    monkeypatch.setitem(sys.modules, module_name, fake_module)

    try:
        result = runtime._generate_theory_via_exercise_gen("Algebra", "Sets and equations")
    finally:
        if original_module is None:
            sys.modules.pop(module_name, None)
        else:
            sys.modules[module_name] = original_module

    assert result["content"] == "Algebra: Sets and equations"
