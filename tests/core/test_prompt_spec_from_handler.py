import importlib

from api.domains.learning.exercise_types.models import ExerciseType
from api.domains.learning.prompts.registry import get_prompt_spec


def test_prompt_spec_is_built_from_handler():
    spec = get_prompt_spec(ExerciseType.TRUE_FALSE)
    assert spec.schema.__name__ == "TrueFalseOutput"
    assert "Đúng/Sai" in spec.instruction
    assert spec.negative_constraints.strip() != ""
    assert spec.explanation_guidance.strip() != ""


def test_prompt_registry_symbol_is_gone():
    registry = importlib.import_module("api.domains.learning.prompts.registry")
    assert not hasattr(registry, "PROMPT_REGISTRY")


def test_package_import_is_cycle_free():
    # Smoke: a fresh import of both packages must not deadlock on the
    # handlers -> prompts.constants -> exercise_types import chain.
    import api.domains.learning.exercise_types as et
    import api.domains.learning.prompts as pr

    assert hasattr(et, "get_handler")
    assert hasattr(pr, "get_prompt_spec")
