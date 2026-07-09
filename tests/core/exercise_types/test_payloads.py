from pydantic import TypeAdapter, ValidationError
import pytest

from api.domains.learning.exercise_types.payloads import (
    ExercisePayload,
    MCQPayload,
    OrderingPayload,
    TrueFalsePayload,
)

_adapter = TypeAdapter(ExercisePayload)


def test_discriminator_picks_correct_variant():
    p = _adapter.validate_python(
        {"exercise_type": "true_false", "statement": "S", "correct_answer": True}
    )
    assert isinstance(p, TrueFalsePayload)
    assert p.statement == "S"


def test_mcq_round_trips_through_model_dump():
    p = MCQPayload(options={"A": "a", "B": "b", "C": "c", "D": "d"}, correct_option="A")
    again = _adapter.validate_python(p.model_dump())
    assert isinstance(again, MCQPayload)
    assert again.correct_option == "A"


def test_ordering_stores_canonical_only():
    p = OrderingPayload(correct_order=["x", "y", "z"])
    assert p.model_dump()["correct_order"] == ["x", "y", "z"]
    assert "items" not in p.model_dump()


def test_unknown_type_rejected():
    with pytest.raises(ValidationError):
        _adapter.validate_python({"exercise_type": "nope"})
