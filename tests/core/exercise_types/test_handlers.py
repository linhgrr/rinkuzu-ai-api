from types import SimpleNamespace

from api.core.learning.exercise_types import (
    FillBlankOutput,
    MatchingOutput,
    MatchingPair,
    MCQOutput,
    MultiCorrectOutput,
    OrderingOutput,
    ShortAnswerOutput,
)
from api.core.learning.exercise_types.models import (
    ExerciseOptions,
    ExerciseOptionsFive,
    ExerciseType,
)
from api.core.learning.exercise_types.payloads import (
    MatchingPayload,
    MCQPayload,
    OrderingPayload,
    TrueFalsePayload,
)
from api.core.learning.exercise_types.registry import get_handler


def _record(payload, **over):
    base = {
        "exercise_id": "ex-1",
        "concept_idx": 0,
        "concept_name": "C",
        "bloom_level": 1,
        "question": "Q",
        "payload": payload,
        "explanation": "",
        "explanation_correct": "ok",
        "explanation_incorrect": "no",
        "correct_answer_compat": None,
    }
    base.update(over)
    return SimpleNamespace(**base)


def test_mcq_payload_from_output_and_evaluate():
    h = get_handler(ExerciseType.MCQ)
    out = MCQOutput(
        question="Q",
        options=ExerciseOptions(A="a", B="b", C="c", D="d"),
        correct_option="B",
        explanation_correct="ok",
        explanation_incorrect="no",
    )
    payload = h.payload_from_output(out)
    assert isinstance(payload, MCQPayload)
    assert payload.correct_option == "B"
    rec = _record(payload)
    assert h.evaluate(rec, {"choice": "b"}) == (True, "B")
    assert h.evaluate(rec, {"choice": "A"}) == (False, "A")
    assert h.tutor_question(rec) == "Q"
    assert h.tutor_options(rec) == ["a", "b", "c", "d"]


def test_true_false_tutor_surfaces_statement():
    h = get_handler(ExerciseType.TRUE_FALSE)
    payload = TrueFalsePayload(statement="Trời xanh", correct_answer=True)
    rec = _record(payload, question="Đúng hay sai?")
    assert "Trời xanh" in h.tutor_question(rec)
    assert h.tutor_options(rec) == ["True", "False"]
    assert h.evaluate(rec, {"boolean": True}) == (True, "True")


def test_ordering_response_is_permutation_canonical_is_stable():
    h = get_handler(ExerciseType.ORDERING)
    out = OrderingOutput(
        question="Sắp xếp",
        items=["x", "y", "z"],
        correct_order=["a", "b", "c"],
        explanation_correct="ok",
        explanation_incorrect="no",
    )
    payload = h.payload_from_output(out)
    assert isinstance(payload, OrderingPayload)
    assert payload.correct_order == ["a", "b", "c"]
    rec = _record(payload, question="Sắp xếp")
    resp = h.to_response_dict(rec)
    assert sorted(resp["items"]) == ["a", "b", "c"]
    assert resp["correct_answer"] == ["a", "b", "c"]
    assert h.to_response_dict(rec)["items"] == resp["items"]
    assert h.evaluate(rec, {"ordering": ["a", "b", "c"]}) == (True, "a → b → c")


def test_matching_response_shuffles_right_items_deterministically():
    h = get_handler(ExerciseType.MATCHING)
    out = MatchingOutput(
        question="Ghép",
        pairs=[
            MatchingPair(left="L1", right="R1"),
            MatchingPair(left="L2", right="R2"),
            MatchingPair(left="L3", right="R3"),
        ],
        explanation_correct="ok",
        explanation_incorrect="no",
    )
    payload = h.payload_from_output(out)
    assert isinstance(payload, MatchingPayload)
    rec = _record(payload, question="Ghép")
    resp = h.to_response_dict(rec)
    assert resp["left_items"] == ["L1", "L2", "L3"]
    assert sorted(resp["right_items"]) == ["R1", "R2", "R3"]
    assert h.to_response_dict(rec)["right_items"] == resp["right_items"]
    assert h.evaluate(rec, {"matching": {"L1": "R1", "L2": "R2", "L3": "R3"}})[0] is True


def test_fill_blank_evaluate_and_options():
    h = get_handler(ExerciseType.FILL_BLANK)
    out = FillBlankOutput(
        question="Điền",
        sentence="Trời ___",
        blank_answers=["xanh", "xanh lam"],
        hint="màu",
        explanation_correct="ok",
        explanation_incorrect="no",
    )
    payload = h.payload_from_output(out)
    rec = _record(payload, question="Điền")
    assert "Trời ___" in h.tutor_question(rec)
    assert h.evaluate(rec, {"blanks": ["Xanh"]}) == (True, "Xanh")


def test_multi_correct_evaluate_orderless():
    h = get_handler(ExerciseType.MULTI_CORRECT)
    out = MultiCorrectOutput(
        question="Chọn",
        options=ExerciseOptionsFive(A="a", B="b", C="c", D="d", E="e"),
        correct_options=["C", "A"],
        explanation_correct="ok",
        explanation_incorrect="no",
    )
    payload = h.payload_from_output(out)
    rec = _record(payload, question="Chọn")
    assert h.evaluate(rec, {"choices": ["A", "C"]}) == (True, "A, C")
    assert h.evaluate(rec, {"choices": ["A"]})[0] is False


def test_short_answer_uses_injected_grader():
    captured = {}

    def grader(**kw):
        captured.update(kw)
        return {"is_correct": True, "explanation": "đạt", "score": 9}

    h = get_handler(ExerciseType.SHORT_ANSWER, short_answer_grader=grader)
    out = ShortAnswerOutput(
        question="Giải thích",
        rubric=["ý 1", "ý 2"],
        sample_answer="mẫu",
        explanation_correct="ok",
        explanation_incorrect="no",
    )
    payload = h.payload_from_output(out)
    rec = _record(payload, question="Giải thích")
    ok, summary = h.evaluate(rec, {"text": "trả lời"})
    assert ok is True
    assert summary == "trả lời"
    assert rec.explanation_correct == "đạt"


def test_prompt_config_methods_return_per_type_text():
    h = get_handler(ExerciseType.TRUE_FALSE)
    assert "Đúng/Sai" in h.prompt_instruction()
    assert h.negative_constraints().strip() != ""
    assert h.explanation_guidance().strip() != ""
