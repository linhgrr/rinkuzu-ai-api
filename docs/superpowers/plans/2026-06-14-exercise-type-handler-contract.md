# Exercise Type Handler Contract — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the seven scattered `exercise_type` branches with one `ExerciseTypeHandler` contract + a typed `payload`, persisted nested in MongoDB, so adding a future exercise type touches exactly one package.

**Architecture:** A discriminated-union `ExercisePayload` (canonical data only) replaces the optional-bag fields on `ExerciseRecord`. Each type is one `@register`-ed handler owning prompt config, serialization, grading, tutor-context, and answer-history. `ordering`/`matching` display order is derived deterministically from `exercise_id` (never stored). DB stores `payload.model_dump()` nested; a one-time idempotent script migrates old flat rows; the read path validates strictly.

**Tech Stack:** Python 3.12, Pydantic v2 (discriminated unions), Beanie/MongoDB, pytest, mypy, ruff.

**Spec:** `docs/superpowers/specs/2026-06-14-exercise-type-handler-contract-design.md`

**Run tests with:** `.venv/bin/python -m pytest <path> -v`
**Type check:** `.venv/bin/python -m mypy api`

---

## Execution order & cutover note

Tasks 1–15 are code; Task 16 is the migration script; Task 17 is regression cleanup. The **read path is strict** (no legacy-flat fallback), so in any real deploy the migration (Task 16) MUST run against the database **before** the new code serves traffic. Locally, tests use fresh fixtures so order among code tasks is the listed order.

---

## Task 1: Convert `exercise_types.py` into a package skeleton (behavior-preserving)

This task ONLY moves code into a package and keeps every public name importable. No behavior changes yet.

**Files:**
- Create: `api/core/learning/exercise_types/__init__.py`
- Create: `api/core/learning/exercise_types/models.py`
- Create: `api/core/learning/exercise_types/selection.py`
- Delete: `api/core/learning/exercise_types.py` (after moving content)
- Test: `tests/core/test_exercise_types.py` (existing — must stay green)

- [ ] **Step 1: Create the package directory and move enum + output models into `models.py`**

Move these from the old `exercise_types.py` verbatim into `api/core/learning/exercise_types/models.py`: the imports it needs (`StrEnum`, `Literal`, `cast`, `pydantic`), `ExerciseType`, `ExerciseBaseOutput`, `ExerciseOptions`, `MCQOutput`, `TrueFalseOutput`, `FillBlankOutput`, `ExerciseOptionsFive`, `MultiCorrectOutput`, `OrderingOutput`, `MatchingPair`, `MatchingOutput`, `ShortAnswerOutput`, `ShortAnswerEvaluationOutput`. Also move `BLOOM_VERBS`.

- [ ] **Step 2: Move selection + serialization helpers into `selection.py`**

Move into `api/core/learning/exercise_types/selection.py` verbatim: the `_rng = secrets.SystemRandom()` line, `_LOW_MASTERY_THRESHOLD`, `_HIGH_MASTERY_THRESHOLD`, `join_lines`, `shuffle_ordering_items`, `serialize_exercise_result`, `EXERCISE_WEIGHTS`, `select_exercise_type`. Add at top:

```python
from .models import (
    ExerciseBaseOutput, ExerciseType, FillBlankOutput, MatchingOutput, MCQOutput,
    MultiCorrectOutput, OrderingOutput, ShortAnswerOutput, TrueFalseOutput,
)
```

- [ ] **Step 3: Write `__init__.py` re-exporting every public name**

```python
from .models import (
    BLOOM_VERBS,
    ExerciseBaseOutput,
    ExerciseOptions,
    ExerciseOptionsFive,
    ExerciseType,
    FillBlankOutput,
    MatchingOutput,
    MatchingPair,
    MCQOutput,
    MultiCorrectOutput,
    OrderingOutput,
    ShortAnswerEvaluationOutput,
    ShortAnswerOutput,
    TrueFalseOutput,
)
from .selection import (
    EXERCISE_WEIGHTS,
    join_lines,
    select_exercise_type,
    serialize_exercise_result,
    shuffle_ordering_items,
)

__all__ = [
    "BLOOM_VERBS", "EXERCISE_WEIGHTS", "ExerciseBaseOutput", "ExerciseOptions",
    "ExerciseOptionsFive", "ExerciseType", "FillBlankOutput", "MCQOutput",
    "MatchingOutput", "MatchingPair", "MultiCorrectOutput", "OrderingOutput",
    "ShortAnswerEvaluationOutput", "ShortAnswerOutput", "TrueFalseOutput",
    "join_lines", "select_exercise_type", "serialize_exercise_result",
    "shuffle_ordering_items",
]
```

- [ ] **Step 4: Delete the old module file**

Run: `rm api/core/learning/exercise_types.py`

- [ ] **Step 5: Run the existing suite to prove the move is behavior-preserving**

Run: `.venv/bin/python -m pytest tests/core/test_exercise_types.py tests/core/test_exercise_service.py -v`
Expected: PASS (same as before the move).

- [ ] **Step 6: Type check + commit**

Run: `.venv/bin/python -m mypy api`
Expected: no new errors.

```bash
git add api/core/learning/exercise_types/ tests/
git commit -m "refactor(exercise): split exercise_types module into package (no behavior change)"
```

---

## Task 2: Add `payloads.py` — typed discriminated union

**Files:**
- Create: `api/core/learning/exercise_types/payloads.py`
- Test: `tests/core/exercise_types/test_payloads.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/core/exercise_types/test_payloads.py
import pytest
from pydantic import TypeAdapter, ValidationError

from api.core.learning.exercise_types.payloads import (
    ExercisePayload, MCQPayload, OrderingPayload, TrueFalsePayload,
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
```

- [ ] **Step 2: Run it to verify it fails**

Run: `.venv/bin/python -m pytest tests/core/exercise_types/test_payloads.py -v`
Expected: FAIL with `ModuleNotFoundError: ...payloads`.

- [ ] **Step 3: Implement `payloads.py`**

```python
"""
payloads.py — Typed per-exercise content, stored canonical-only and persisted nested.
"""

from __future__ import annotations

from typing import Annotated, Literal, Union

from pydantic import BaseModel, Field

from .models import ExerciseType


class MCQPayload(BaseModel):
    exercise_type: Literal[ExerciseType.MCQ] = ExerciseType.MCQ
    options: dict[str, str]
    correct_option: str


class TrueFalsePayload(BaseModel):
    exercise_type: Literal[ExerciseType.TRUE_FALSE] = ExerciseType.TRUE_FALSE
    statement: str
    correct_answer: bool


class FillBlankPayload(BaseModel):
    exercise_type: Literal[ExerciseType.FILL_BLANK] = ExerciseType.FILL_BLANK
    sentence: str
    hint: str
    blank_answers: list[str]


class MultiCorrectPayload(BaseModel):
    exercise_type: Literal[ExerciseType.MULTI_CORRECT] = ExerciseType.MULTI_CORRECT
    options: dict[str, str]
    correct_options: list[str]


class OrderingPayload(BaseModel):
    exercise_type: Literal[ExerciseType.ORDERING] = ExerciseType.ORDERING
    correct_order: list[str]


class MatchingPayload(BaseModel):
    exercise_type: Literal[ExerciseType.MATCHING] = ExerciseType.MATCHING
    pairs: list[dict[str, str]]


class ShortAnswerPayload(BaseModel):
    exercise_type: Literal[ExerciseType.SHORT_ANSWER] = ExerciseType.SHORT_ANSWER
    rubric: list[str]
    sample_answer: str


ExercisePayload = Annotated[
    Union[
        MCQPayload,
        TrueFalsePayload,
        FillBlankPayload,
        MultiCorrectPayload,
        OrderingPayload,
        MatchingPayload,
        ShortAnswerPayload,
    ],
    Field(discriminator="exercise_type"),
]
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `.venv/bin/python -m pytest tests/core/exercise_types/test_payloads.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add api/core/learning/exercise_types/payloads.py tests/core/exercise_types/test_payloads.py
git commit -m "feat(exercise): add typed ExercisePayload discriminated union"
```

---

## Task 3: Add `shuffle.py` — deterministic display shuffle

**Files:**
- Create: `api/core/learning/exercise_types/shuffle.py`
- Test: `tests/core/exercise_types/test_shuffle.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/core/exercise_types/test_shuffle.py
from api.core.learning.exercise_types.shuffle import deterministic_shuffle


def test_same_seed_same_order():
    items = ["a", "b", "c", "d", "e"]
    assert deterministic_shuffle(items, "ex-1") == deterministic_shuffle(items, "ex-1")


def test_is_a_permutation_not_a_mutation():
    items = ["a", "b", "c", "d"]
    out = deterministic_shuffle(items, "ex-1")
    assert sorted(out) == sorted(items)
    assert items == ["a", "b", "c", "d"]  # input untouched


def test_different_seeds_generally_differ():
    items = [str(i) for i in range(10)]
    assert deterministic_shuffle(items, "ex-1") != deterministic_shuffle(items, "ex-2")
```

- [ ] **Step 2: Run it to verify it fails**

Run: `.venv/bin/python -m pytest tests/core/exercise_types/test_shuffle.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement `shuffle.py`**

```python
"""
shuffle.py — Deterministic display-order shuffle seeded by exercise_id.

The DB stores canonical order only; the shuffled order shown to the learner is
re-derived on every serve from the exercise_id, so it is stable across
generate/tutor/refetch without persisting any extra state.
"""

from __future__ import annotations

import random


def deterministic_shuffle(items: list[str], seed: str) -> list[str]:
    out = list(items)
    random.Random(seed).shuffle(out)
    return out
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `.venv/bin/python -m pytest tests/core/exercise_types/test_shuffle.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add api/core/learning/exercise_types/shuffle.py tests/core/exercise_types/test_shuffle.py
git commit -m "feat(exercise): add deterministic exercise_id-seeded display shuffle"
```

---

## Task 4: Add the `ExerciseTypeHandler` ABC + `registry.py`

The ABC and registry land together so the registry has a base type to store. Concrete handlers come in Task 5. To avoid an import cycle (`ExerciseRecord` lives in `session.py`, which will import handlers), the ABC references the record via `TYPE_CHECKING` only.

**Files:**
- Create: `api/core/learning/exercise_types/base.py`
- Create: `api/core/learning/exercise_types/registry.py`
- Test: `tests/core/exercise_types/test_registry.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/core/exercise_types/test_registry.py
import pytest

from api.core.learning.exercise_types.base import ExerciseTypeHandler
from api.core.learning.exercise_types.models import ExerciseType
from api.core.learning.exercise_types.registry import get_handler, register


def test_register_and_get_handler_returns_fresh_instances():
    @register
    class _DummyHandler(ExerciseTypeHandler):
        exercise_type = ExerciseType.MCQ
        output_model = object
        payload_model = object

        def prompt_instruction(self): return "i"
        def negative_constraints(self): return "n"
        def explanation_guidance(self): return "e"
        def payload_from_output(self, result): return result
        def to_response_dict(self, exercise): return {}
        def evaluate(self, exercise, answer): return (True, "")
        def tutor_question(self, exercise): return ""
        def tutor_options(self, exercise): return []
        def serialize_answer(self, exercise, answer): return None

    a = get_handler(ExerciseType.MCQ)
    b = get_handler(ExerciseType.MCQ)
    assert isinstance(a, _DummyHandler)
    assert a is not b  # fresh per call


def test_get_handler_unknown_type_raises_keyerror():
    class _Fake:
        value = "does-not-exist"
    with pytest.raises(KeyError):
        get_handler(_Fake())
```

- [ ] **Step 2: Run it to verify it fails**

Run: `.venv/bin/python -m pytest tests/core/exercise_types/test_registry.py -v`
Expected: FAIL with `ModuleNotFoundError: ...base`.

- [ ] **Step 3: Implement `base.py`**

```python
"""
base.py — The ExerciseTypeHandler contract every exercise type implements.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, ClassVar

from pydantic import BaseModel

from .models import ExerciseBaseOutput, ExerciseType

if TYPE_CHECKING:
    from collections.abc import Callable

    from api.core.learning.session import ExerciseRecord


class ExerciseTypeHandler(ABC):
    exercise_type: ClassVar[ExerciseType]
    output_model: ClassVar[type[ExerciseBaseOutput]]
    payload_model: ClassVar[type[BaseModel]]

    def __init__(self, *, short_answer_grader: Callable[..., dict] | None = None) -> None:
        self._grader = short_answer_grader

    # 1. generation config (replaces PROMPT_REGISTRY entry)
    @abstractmethod
    def prompt_instruction(self) -> str: ...
    @abstractmethod
    def negative_constraints(self) -> str: ...
    @abstractmethod
    def explanation_guidance(self) -> str: ...

    # 2. LM output model -> payload (canonical; no shuffle)
    @abstractmethod
    def payload_from_output(self, result: ExerciseBaseOutput) -> BaseModel: ...

    # 3. ExerciseRecord -> API response dict (same shape as today; shuffle from exercise_id)
    @abstractmethod
    def to_response_dict(self, exercise: ExerciseRecord) -> dict[str, Any]: ...

    # 4. grading (short_answer uses self._grader; others ignore it)
    @abstractmethod
    def evaluate(self, exercise: ExerciseRecord, answer: dict[str, Any]) -> tuple[bool, str]: ...

    # 5. tutor context for the chatbot
    @abstractmethod
    def tutor_question(self, exercise: ExerciseRecord) -> str: ...
    @abstractmethod
    def tutor_options(self, exercise: ExerciseRecord) -> list[str]: ...

    # 6. user answer -> history string
    @abstractmethod
    def serialize_answer(
        self, exercise: ExerciseRecord, answer: dict[str, Any]
    ) -> str | None: ...
```

- [ ] **Step 4: Implement `registry.py`**

```python
"""
registry.py — Maps each ExerciseType to its handler class.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .base import ExerciseTypeHandler
from .models import ExerciseType

if TYPE_CHECKING:
    from collections.abc import Callable

_HANDLER_CLASSES: dict[ExerciseType, type[ExerciseTypeHandler]] = {}


def register(cls: type[ExerciseTypeHandler]) -> type[ExerciseTypeHandler]:
    _HANDLER_CLASSES[cls.exercise_type] = cls
    return cls


def get_handler(
    exercise_type: ExerciseType,
    *,
    short_answer_grader: Callable[..., dict] | None = None,
) -> ExerciseTypeHandler:
    return _HANDLER_CLASSES[exercise_type](short_answer_grader=short_answer_grader)
```

- [ ] **Step 5: Run the test to verify it passes**

Run: `.venv/bin/python -m pytest tests/core/exercise_types/test_registry.py -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add api/core/learning/exercise_types/base.py api/core/learning/exercise_types/registry.py tests/core/exercise_types/test_registry.py
git commit -m "feat(exercise): add ExerciseTypeHandler ABC and registry"
```

---

## Task 5: Implement the 7 concrete handlers

This is the heart of the refactor. Each handler owns its prompt text (moved from `prompts/constants.py` + `prompts/registry.py`), `payload_from_output`, `to_response_dict`, `evaluate`, `tutor_question`/`tutor_options`, and `serialize_answer`. The behavior must match the current `serialize_exercise_result` (selection.py) and `evaluate_answer`/`serialize_answer_for_history` (answer_eval.py) outputs exactly.

Handlers import the prompt constants that stay in `prompts/constants.py` (the constants are not deleted — only `PROMPT_REGISTRY` is). Each handler method returns the SAME per-type constant the old `PROMPT_REGISTRY` entry used.

**Files:**
- Create: `api/core/learning/exercise_types/handlers.py`
- Modify: `api/core/learning/exercise_types/__init__.py` (import handlers so `@register` runs)
- Test: `tests/core/exercise_types/test_handlers.py`

- [ ] **Step 1: Write the failing tests (per-type behavior, golden against today's output)**

```python
# tests/core/exercise_types/test_handlers.py
from types import SimpleNamespace

from api.core.learning.exercise_types import (
    FillBlankOutput, MatchingOutput, MatchingPair, MCQOutput, MultiCorrectOutput,
    OrderingOutput, ShortAnswerOutput, TrueFalseOutput,
)
from api.core.learning.exercise_types.models import ExerciseOptions, ExerciseOptionsFive, ExerciseType
from api.core.learning.exercise_types.payloads import (
    MatchingPayload, MCQPayload, OrderingPayload, TrueFalsePayload,
)
from api.core.learning.exercise_types.registry import get_handler


def _record(payload, **over):
    base = dict(
        exercise_id="ex-1", concept_idx=0, concept_name="C", bloom_level=1,
        question="Q", payload=payload, explanation="", explanation_correct="ok",
        explanation_incorrect="no", correct_answer_compat=None,
    )
    base.update(over)
    return SimpleNamespace(**base)


def test_mcq_payload_from_output_and_evaluate():
    h = get_handler(ExerciseType.MCQ)
    out = MCQOutput(
        question="Q", options=ExerciseOptions(A="a", B="b", C="c", D="d"),
        correct_option="B", explanation_correct="ok", explanation_incorrect="no",
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
    # the bug fix: the statement MUST be in the tutor question
    assert "Trời xanh" in h.tutor_question(rec)
    assert h.tutor_options(rec) == ["True", "False"]
    assert h.evaluate(rec, {"boolean": True}) == (True, "True")


def test_ordering_response_is_permutation_canonical_is_stable():
    h = get_handler(ExerciseType.ORDERING)
    out = OrderingOutput(
        question="Sắp xếp", items=["x", "y", "z"], correct_order=["a", "b", "c"],
        explanation_correct="ok", explanation_incorrect="no",
    )
    payload = h.payload_from_output(out)
    assert isinstance(payload, OrderingPayload)
    assert payload.correct_order == ["a", "b", "c"]
    rec = _record(payload, question="Sắp xếp")
    resp = h.to_response_dict(rec)
    assert sorted(resp["items"]) == ["a", "b", "c"]          # permutation of canonical
    assert resp["correct_answer"] == ["a", "b", "c"]          # canonical preserved
    assert h.to_response_dict(rec)["items"] == resp["items"]  # deterministic
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
        explanation_correct="ok", explanation_incorrect="no",
    )
    payload = h.payload_from_output(out)
    assert isinstance(payload, MatchingPayload)
    rec = _record(payload, question="Ghép")
    resp = h.to_response_dict(rec)
    assert resp["left_items"] == ["L1", "L2", "L3"]
    assert sorted(resp["right_items"]) == ["R1", "R2", "R3"]
    assert h.to_response_dict(rec)["right_items"] == resp["right_items"]  # deterministic
    assert h.evaluate(rec, {"matching": {"L1": "R1", "L2": "R2", "L3": "R3"}})[0] is True


def test_fill_blank_evaluate_and_options():
    h = get_handler(ExerciseType.FILL_BLANK)
    out = FillBlankOutput(
        question="Điền", sentence="Trời ___", blank_answers=["xanh", "xanh lam"],
        hint="màu", explanation_correct="ok", explanation_incorrect="no",
    )
    payload = h.payload_from_output(out)
    rec = _record(payload, question="Điền")
    assert "Trời ___" in h.tutor_question(rec)
    assert h.evaluate(rec, {"blanks": ["Xanh"]}) == (True, "Xanh")


def test_multi_correct_evaluate_orderless():
    h = get_handler(ExerciseType.MULTI_CORRECT)
    out = MultiCorrectOutput(
        question="Chọn", options=ExerciseOptionsFive(A="a", B="b", C="c", D="d", E="e"),
        correct_options=["C", "A"], explanation_correct="ok", explanation_incorrect="no",
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
        question="Giải thích", rubric=["ý 1", "ý 2"], sample_answer="mẫu",
        explanation_correct="ok", explanation_incorrect="no",
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
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/python -m pytest tests/core/exercise_types/test_handlers.py -v`
Expected: FAIL with `ModuleNotFoundError: ...handlers`.

- [ ] **Step 3: Implement `handlers.py`**

Note on `evaluate`/`serialize_answer`: copy the matching logic from the current `answer_eval.py` per type, but read from `exercise.payload` instead of flat fields. `correct_answer` equivalents now come from the payload: MCQ `payload.correct_option`; true_false `payload.correct_answer`; fill_blank `payload.blank_answers`; multi_correct `payload.correct_options`; ordering `payload.correct_order`; matching the `{left: right}` map built from `payload.pairs`; short_answer `payload.sample_answer` + `payload.rubric`.

```python
"""
handlers.py — One ExerciseTypeHandler per exercise type. Each owns prompt config,
serialization, grading, tutor context, and answer-history for its type.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from api.core.learning.prompts.constants import EXPLANATION_GUIDANCE, NEGATIVE_CONSTRAINTS

from .base import ExerciseTypeHandler
from .models import (
    ExerciseType, FillBlankOutput, MatchingOutput, MCQOutput, MultiCorrectOutput,
    OrderingOutput, ShortAnswerOutput, TrueFalseOutput,
)
from .payloads import (
    FillBlankPayload, MatchingPayload, MCQPayload, MultiCorrectPayload,
    OrderingPayload, ShortAnswerPayload, TrueFalsePayload,
)
from .registry import register
from .selection import join_lines
from .shuffle import deterministic_shuffle

if TYPE_CHECKING:
    from api.core.learning.session import ExerciseRecord


def _normalize(value: str) -> str:
    return " ".join(value.strip().casefold().split())


# ---- MCQ ----------------------------------------------------------------

_MCQ_INSTRUCTION = (
    "Hãy tạo 1 câu hỏi trắc nghiệm khách quan gồm đúng 4 đáp án A, B, C, D.\n"
    "- Có duy nhất 1 đáp án đúng.\n"
    "- Distractor phải hợp lý và đủ gần để học sinh có thể nhầm nếu hiểu chưa chắc.\n"
)


@register
class MCQHandler(ExerciseTypeHandler):
    exercise_type = ExerciseType.MCQ
    output_model = MCQOutput
    payload_model = MCQPayload

    def prompt_instruction(self) -> str:
        return _MCQ_INSTRUCTION

    def negative_constraints(self) -> str:
        return NEGATIVE_CONSTRAINTS[ExerciseType.MCQ]

    def explanation_guidance(self) -> str:
        return EXPLANATION_GUIDANCE[ExerciseType.MCQ]

    def payload_from_output(self, result: MCQOutput) -> MCQPayload:
        return MCQPayload(
            options={
                "A": result.options.A, "B": result.options.B,
                "C": result.options.C, "D": result.options.D,
            },
            correct_option=result.correct_option,
        )

    def to_response_dict(self, exercise: ExerciseRecord) -> dict[str, Any]:
        payload: MCQPayload = exercise.payload
        return {
            "exercise_type": self.exercise_type.value,
            "question": exercise.question,
            "options": dict(payload.options),
            "correct_option": payload.correct_option,
            "correct_answer": payload.correct_option,
            "explanation_correct": exercise.explanation_correct,
            "explanation_incorrect": exercise.explanation_incorrect,
        }

    def evaluate(self, exercise: ExerciseRecord, answer: dict[str, Any]) -> tuple[bool, str]:
        payload: MCQPayload = exercise.payload
        selected = (answer.get("choice") or "").strip().upper()
        return selected == payload.correct_option.strip().upper(), selected

    def tutor_question(self, exercise: ExerciseRecord) -> str:
        return exercise.question

    def tutor_options(self, exercise: ExerciseRecord) -> list[str]:
        payload: MCQPayload = exercise.payload
        return [payload.options[k] for k in sorted(payload.options) if payload.options.get(k)]

    def serialize_answer(self, exercise: ExerciseRecord, answer: dict[str, Any]) -> str | None:
        return (answer.get("choice") or "").strip().upper() or None


# ---- True/False ---------------------------------------------------------

_TRUE_FALSE_INSTRUCTION = (
    "Hãy tạo 1 bài tập dạng Đúng/Sai.\n"
    "- `statement` là một mệnh đề duy nhất để học sinh đánh giá.\n"
    "- `question` là lời dẫn ngắn yêu cầu chọn đúng hoặc sai.\n"
)


@register
class TrueFalseHandler(ExerciseTypeHandler):
    exercise_type = ExerciseType.TRUE_FALSE
    output_model = TrueFalseOutput
    payload_model = TrueFalsePayload

    def prompt_instruction(self) -> str:
        return _TRUE_FALSE_INSTRUCTION

    def negative_constraints(self) -> str:
        return NEGATIVE_CONSTRAINTS[ExerciseType.TRUE_FALSE]

    def explanation_guidance(self) -> str:
        return EXPLANATION_GUIDANCE[ExerciseType.TRUE_FALSE]

    def payload_from_output(self, result: TrueFalseOutput) -> TrueFalsePayload:
        return TrueFalsePayload(statement=result.statement, correct_answer=result.correct_answer)

    def to_response_dict(self, exercise: ExerciseRecord) -> dict[str, Any]:
        payload: TrueFalsePayload = exercise.payload
        return {
            "exercise_type": self.exercise_type.value,
            "question": exercise.question,
            "statement": payload.statement,
            "correct_answer": payload.correct_answer,
            "correct_option": "True" if payload.correct_answer else "False",
            "explanation_correct": exercise.explanation_correct,
            "explanation_incorrect": exercise.explanation_incorrect,
        }

    def evaluate(self, exercise: ExerciseRecord, answer: dict[str, Any]) -> tuple[bool, str]:
        payload: TrueFalsePayload = exercise.payload
        selected = answer.get("boolean")
        return (
            selected is not None and bool(selected) == bool(payload.correct_answer),
            "True" if selected else "False",
        )

    def tutor_question(self, exercise: ExerciseRecord) -> str:
        payload: TrueFalsePayload = exercise.payload
        return f"{exercise.question}\n\nPhát biểu: {payload.statement}".strip()

    def tutor_options(self, exercise: ExerciseRecord) -> list[str]:
        return ["True", "False"]

    def serialize_answer(self, exercise: ExerciseRecord, answer: dict[str, Any]) -> str | None:
        value = answer.get("boolean")
        return None if value is None else ("True" if value else "False")


# ---- Fill blank ---------------------------------------------------------

_FILL_BLANK_INSTRUCTION = (
    "Hãy tạo 1 bài tập điền vào chỗ trống.\n"
    "- `sentence` phải chứa đúng 1 chỗ trống ký hiệu là `_____`.\n"
    "- `blank_answers` gồm 1-3 đáp án tương đương được chấp nhận.\n"
    "- `hint` ngắn gọn nhưng không lộ đáp án.\n"
)


@register
class FillBlankHandler(ExerciseTypeHandler):
    exercise_type = ExerciseType.FILL_BLANK
    output_model = FillBlankOutput
    payload_model = FillBlankPayload

    def prompt_instruction(self) -> str:
        return _FILL_BLANK_INSTRUCTION

    def negative_constraints(self) -> str:
        return NEGATIVE_CONSTRAINTS[ExerciseType.FILL_BLANK]

    def explanation_guidance(self) -> str:
        return EXPLANATION_GUIDANCE[ExerciseType.FILL_BLANK]

    def payload_from_output(self, result: FillBlankOutput) -> FillBlankPayload:
        accepted = [a.strip() for a in result.blank_answers if a.strip()]
        return FillBlankPayload(
            sentence=result.sentence, hint=result.hint, blank_answers=accepted,
        )

    def to_response_dict(self, exercise: ExerciseRecord) -> dict[str, Any]:
        payload: FillBlankPayload = exercise.payload
        canonical = payload.blank_answers[0] if payload.blank_answers else ""
        return {
            "exercise_type": self.exercise_type.value,
            "question": exercise.question,
            "sentence": payload.sentence,
            "hint": payload.hint,
            "blank_answers": list(payload.blank_answers),
            "correct_answer": list(payload.blank_answers),
            "correct_option": canonical,
            "explanation_correct": exercise.explanation_correct,
            "explanation_incorrect": exercise.explanation_incorrect,
        }

    def evaluate(self, exercise: ExerciseRecord, answer: dict[str, Any]) -> tuple[bool, str]:
        payload: FillBlankPayload = exercise.payload
        user = [_normalize(b) for b in (answer.get("blanks") or []) if b and b.strip()]
        accepted = [_normalize(a) for a in payload.blank_answers]
        ok = bool(user and accepted and user[0] in accepted)
        return ok, ", ".join(answer.get("blanks") or [])

    def tutor_question(self, exercise: ExerciseRecord) -> str:
        payload: FillBlankPayload = exercise.payload
        return f"{exercise.question}\n\nCâu cần điền: {payload.sentence}".strip()

    def tutor_options(self, exercise: ExerciseRecord) -> list[str]:
        payload: FillBlankPayload = exercise.payload
        return [f"Gợi ý: {payload.hint}"] if payload.hint else []

    def serialize_answer(self, exercise: ExerciseRecord, answer: dict[str, Any]) -> str | None:
        blanks = [b.strip() for b in (answer.get("blanks") or []) if b and b.strip()]
        return ", ".join(blanks) or None


# ---- Multi-correct ------------------------------------------------------

_MULTI_CORRECT_INSTRUCTION = (
    "Hãy tạo 1 câu hỏi trắc nghiệm nhiều đáp án đúng gồm đúng 5 lựa chọn A, B, C, D, E.\n"
    "- Số đáp án đúng có thể là 2, 3, hoặc 4 — hãy thoải mái chọn số lượng phù hợp nhất với nội dung câu hỏi.\n"
    "- Các lựa chọn sai phải sai vì thiếu điều kiện hoặc sai bản chất, không được vô lý.\n"
    "- Trước khi output, hãy tự kiểm tra TỪNG lựa chọn A-E: tính toán/suy luận cụ thể để xác nhận đúng hay sai.\n"
)


@register
class MultiCorrectHandler(ExerciseTypeHandler):
    exercise_type = ExerciseType.MULTI_CORRECT
    output_model = MultiCorrectOutput
    payload_model = MultiCorrectPayload

    def prompt_instruction(self) -> str:
        return _MULTI_CORRECT_INSTRUCTION

    def negative_constraints(self) -> str:
        return NEGATIVE_CONSTRAINTS[ExerciseType.MULTI_CORRECT]

    def explanation_guidance(self) -> str:
        return EXPLANATION_GUIDANCE[ExerciseType.MULTI_CORRECT]

    def payload_from_output(self, result: MultiCorrectOutput) -> MultiCorrectPayload:
        return MultiCorrectPayload(
            options={
                "A": result.options.A, "B": result.options.B, "C": result.options.C,
                "D": result.options.D, "E": result.options.E,
            },
            correct_options=sorted(set(result.correct_options)),
        )

    def to_response_dict(self, exercise: ExerciseRecord) -> dict[str, Any]:
        payload: MultiCorrectPayload = exercise.payload
        correct = sorted(set(payload.correct_options))
        return {
            "exercise_type": self.exercise_type.value,
            "question": exercise.question,
            "options": dict(payload.options),
            "correct_answer": correct,
            "correct_option": ", ".join(correct),
            "explanation_correct": exercise.explanation_correct,
            "explanation_incorrect": exercise.explanation_incorrect,
        }

    def evaluate(self, exercise: ExerciseRecord, answer: dict[str, Any]) -> tuple[bool, str]:
        payload: MultiCorrectPayload = exercise.payload
        selected = sorted({c.strip().upper() for c in (answer.get("choices") or []) if c and c.strip()})
        expected = sorted({c.strip().upper() for c in payload.correct_options})
        return selected == expected, ", ".join(selected)

    def tutor_question(self, exercise: ExerciseRecord) -> str:
        return exercise.question

    def tutor_options(self, exercise: ExerciseRecord) -> list[str]:
        payload: MultiCorrectPayload = exercise.payload
        return [payload.options[k] for k in sorted(payload.options) if payload.options.get(k)]

    def serialize_answer(self, exercise: ExerciseRecord, answer: dict[str, Any]) -> str | None:
        choices = answer.get("choices") or []
        return ", ".join(sorted(choices)) if choices else None


# ---- Ordering -----------------------------------------------------------

_ORDERING_INSTRUCTION = (
    "Hãy tạo 1 bài tập sắp xếp thứ tự.\n"
    "- `correct_order` là nguồn chân lý, phải đầy đủ và đúng tuyệt đối.\n"
    "- `items` phải chứa đúng các phần tử của `correct_order`, không thêm bớt.\n"
    "- Nội dung phải chấm được bằng một trình tự duy nhất.\n"
)


@register
class OrderingHandler(ExerciseTypeHandler):
    exercise_type = ExerciseType.ORDERING
    output_model = OrderingOutput
    payload_model = OrderingPayload

    def prompt_instruction(self) -> str:
        return _ORDERING_INSTRUCTION

    def negative_constraints(self) -> str:
        return NEGATIVE_CONSTRAINTS[ExerciseType.ORDERING]

    def explanation_guidance(self) -> str:
        return EXPLANATION_GUIDANCE[ExerciseType.ORDERING]

    def payload_from_output(self, result: OrderingOutput) -> OrderingPayload:
        return OrderingPayload(correct_order=[i.strip() for i in result.correct_order if i.strip()])

    def to_response_dict(self, exercise: ExerciseRecord) -> dict[str, Any]:
        payload: OrderingPayload = exercise.payload
        display = deterministic_shuffle(payload.correct_order, exercise.exercise_id)
        return {
            "exercise_type": self.exercise_type.value,
            "question": exercise.question,
            "items": display,
            "correct_answer": list(payload.correct_order),
            "correct_option": join_lines(payload.correct_order),
            "explanation_correct": exercise.explanation_correct,
            "explanation_incorrect": exercise.explanation_incorrect,
        }

    def evaluate(self, exercise: ExerciseRecord, answer: dict[str, Any]) -> tuple[bool, str]:
        payload: OrderingPayload = exercise.payload
        selected = [_normalize(i) for i in (answer.get("ordering") or []) if i and i.strip()]
        expected = [_normalize(i) for i in payload.correct_order]
        return bool(selected) and selected == expected, " → ".join(answer.get("ordering") or [])

    def tutor_question(self, exercise: ExerciseRecord) -> str:
        return exercise.question

    def tutor_options(self, exercise: ExerciseRecord) -> list[str]:
        payload: OrderingPayload = exercise.payload
        return deterministic_shuffle(payload.correct_order, exercise.exercise_id)

    def serialize_answer(self, exercise: ExerciseRecord, answer: dict[str, Any]) -> str | None:
        ordering = [i.strip() for i in (answer.get("ordering") or []) if i and i.strip()]
        return " → ".join(ordering) or None


# ---- Matching -----------------------------------------------------------

_MATCHING_INSTRUCTION = (
    "Hãy tạo 1 bài tập ghép nối.\n"
    "- `pairs` gồm 3-5 cặp ghép đúng.\n"
    "- Mỗi `left` chỉ khớp tốt với đúng 1 `right`.\n"
)


@register
class MatchingHandler(ExerciseTypeHandler):
    exercise_type = ExerciseType.MATCHING
    output_model = MatchingOutput
    payload_model = MatchingPayload

    def prompt_instruction(self) -> str:
        return _MATCHING_INSTRUCTION

    def negative_constraints(self) -> str:
        return NEGATIVE_CONSTRAINTS[ExerciseType.MATCHING]

    def explanation_guidance(self) -> str:
        return EXPLANATION_GUIDANCE[ExerciseType.MATCHING]

    def payload_from_output(self, result: MatchingOutput) -> MatchingPayload:
        return MatchingPayload(
            pairs=[{"left": p.left, "right": p.right} for p in result.pairs],
        )

    def to_response_dict(self, exercise: ExerciseRecord) -> dict[str, Any]:
        payload: MatchingPayload = exercise.payload
        left_items = [p["left"] for p in payload.pairs]
        right_canonical = [p["right"] for p in payload.pairs]
        right_items = deterministic_shuffle(right_canonical, exercise.exercise_id)
        return {
            "exercise_type": self.exercise_type.value,
            "question": exercise.question,
            "pairs": [dict(p) for p in payload.pairs],
            "left_items": left_items,
            "right_items": right_items,
            "correct_answer": {p["left"]: p["right"] for p in payload.pairs},
            "correct_option": join_lines([f"{p['left']} → {p['right']}" for p in payload.pairs]),
            "explanation_correct": exercise.explanation_correct,
            "explanation_incorrect": exercise.explanation_incorrect,
        }

    def evaluate(self, exercise: ExerciseRecord, answer: dict[str, Any]) -> tuple[bool, str]:
        payload: MatchingPayload = exercise.payload
        selected = {
            _normalize(left): _normalize(right)
            for left, right in (answer.get("matching") or {}).items() if left and right
        }
        expected = {_normalize(p["left"]): _normalize(p["right"]) for p in payload.pairs}
        summary = ", ".join(
            f"{left} -> {right}" for left, right in (answer.get("matching") or {}).items()
        )
        return bool(selected) and selected == expected, summary

    def tutor_question(self, exercise: ExerciseRecord) -> str:
        return exercise.question

    def tutor_options(self, exercise: ExerciseRecord) -> list[str]:
        payload: MatchingPayload = exercise.payload
        right_canonical = [p["right"] for p in payload.pairs]
        return deterministic_shuffle(right_canonical, exercise.exercise_id)

    def serialize_answer(self, exercise: ExerciseRecord, answer: dict[str, Any]) -> str | None:
        matching = answer.get("matching") or {}
        if not matching:
            return None
        return ", ".join(f"{left} -> {right}" for left, right in matching.items())


# ---- Short answer -------------------------------------------------------

_SHORT_ANSWER_INSTRUCTION = (
    "Hãy tạo 1 câu hỏi trả lời ngắn để chấm bằng rubric.\n"
    "- `question` phải mở vừa đủ để học sinh diễn đạt, nhưng vẫn chấm được khách quan.\n"
    "- `rubric` gồm 2-4 tiêu chí ngắn, rõ ràng.\n"
    "- `sample_answer` súc tích nhưng bám đủ rubric.\n"
)


@register
class ShortAnswerHandler(ExerciseTypeHandler):
    exercise_type = ExerciseType.SHORT_ANSWER
    output_model = ShortAnswerOutput
    payload_model = ShortAnswerPayload

    def prompt_instruction(self) -> str:
        return _SHORT_ANSWER_INSTRUCTION

    def negative_constraints(self) -> str:
        return NEGATIVE_CONSTRAINTS[ExerciseType.SHORT_ANSWER]

    def explanation_guidance(self) -> str:
        return EXPLANATION_GUIDANCE[ExerciseType.SHORT_ANSWER]

    def payload_from_output(self, result: ShortAnswerOutput) -> ShortAnswerPayload:
        return ShortAnswerPayload(rubric=list(result.rubric), sample_answer=result.sample_answer)

    def to_response_dict(self, exercise: ExerciseRecord) -> dict[str, Any]:
        payload: ShortAnswerPayload = exercise.payload
        return {
            "exercise_type": self.exercise_type.value,
            "question": exercise.question,
            "rubric": list(payload.rubric),
            "sample_answer": payload.sample_answer,
            "correct_answer": payload.sample_answer,
            "correct_option": payload.sample_answer,
            "explanation_correct": exercise.explanation_correct,
            "explanation_incorrect": exercise.explanation_incorrect,
        }

    def evaluate(self, exercise: ExerciseRecord, answer: dict[str, Any]) -> tuple[bool, str]:
        if self._grader is None:
            raise RuntimeError("short_answer_grader is required for short_answer exercises")
        payload: ShortAnswerPayload = exercise.payload
        student = (answer.get("text") or "").strip()
        grading = self._grader(
            concept_name=exercise.concept_name,
            question=exercise.question,
            rubric=payload.rubric,
            sample_answer=payload.sample_answer,
            student_answer=student,
        )
        exercise.explanation_correct = str(grading["explanation"])
        exercise.explanation_incorrect = str(grading["explanation"])
        return bool(grading["is_correct"]), student

    def tutor_question(self, exercise: ExerciseRecord) -> str:
        return exercise.question

    def tutor_options(self, exercise: ExerciseRecord) -> list[str]:
        payload: ShortAnswerPayload = exercise.payload
        return list(payload.rubric) or ["Trả lời ngắn gọn, bám sát câu hỏi."]

    def serialize_answer(self, exercise: ExerciseRecord, answer: dict[str, Any]) -> str | None:
        return (answer.get("text") or "").strip() or None
```

- [ ] **Step 4: Register handlers on package import**

Add to `api/core/learning/exercise_types/__init__.py` (at the END, after the existing re-exports), so importing the package registers all handlers:

```python
from . import handlers as handlers  # noqa: E402,F401  (import for @register side effects)
from .base import ExerciseTypeHandler
from .registry import get_handler, register

__all__ += ["ExerciseTypeHandler", "get_handler", "register"]
```

- [ ] **Step 5: Run the handler tests**

Run: `.venv/bin/python -m pytest tests/core/exercise_types/test_handlers.py -v`
Expected: PASS.

- [ ] **Step 6: Type check + commit**

Run: `.venv/bin/python -m mypy api`
Expected: no new errors.

```bash
git add api/core/learning/exercise_types/handlers.py api/core/learning/exercise_types/__init__.py tests/core/exercise_types/test_handlers.py
git commit -m "feat(exercise): implement 7 concrete exercise type handlers"
```

---

## Task 6: Add `payload` to `ExerciseRecord` (expand phase — keep flat fields)

This uses an **expand-contract** strategy: we ADD `payload` now and keep the flat content
fields populated so every existing reader stays green. Flat fields are deleted only in the
final cleanup task (Task 17), once every consumer reads via `payload`.

**Files:**
- Modify: `api/core/learning/session.py:42-66` (`ExerciseRecord` dataclass)
- Test: `tests/core/exercise_types/test_envelope.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/core/exercise_types/test_envelope.py
from api.core.learning.session import ExerciseRecord
from api.core.learning.exercise_types.payloads import MCQPayload


def test_exercise_record_holds_payload():
    rec = ExerciseRecord(
        exercise_id="ex1",
        concept_idx=0,
        concept_name="C",
        bloom_level=1,
        question="Q",
        payload=MCQPayload(options={"A": "a", "B": "b", "C": "c", "D": "d"}, correct_option="A"),
    )
    assert rec.payload.exercise_type == "mcq"
    assert rec.payload.correct_option == "A"
```

- [ ] **Step 2: Run it to verify it fails**

Run: `.venv/bin/python -m pytest tests/core/exercise_types/test_envelope.py -v`
Expected: FAIL — `ExerciseRecord.__init__() missing/unexpected 'payload'` (payload not yet a field).

- [ ] **Step 3: Add the `payload` field**

In `api/core/learning/session.py`, add the import near the other exercise_types imports
(line ~33 `from .exercise_types import ExerciseType`):

```python
from .exercise_types.payloads import ExercisePayload
```

Then add the field to the `ExerciseRecord` dataclass. **Field ordering matters:** a dataclass
field with a default may NOT precede a field without one. `payload` has a default (`None`
during the transition), so it must go in the defaulted block — insert it immediately BEFORE
`exercise_type: ExerciseType = ExerciseType.MCQ` (the first existing defaulted field):

```python
    payload: ExercisePayload | None = None
```

Additionally, the two remaining non-default flat fields `correct_option: str` and
`explanation: str` must be given defaults during the expand phase, so a record can be
constructed from the envelope + `payload` alone (the Step 1 test does exactly that). Change:

```python
    correct_option: str
    explanation: str
```
to:
```python
    correct_option: str = ""
    explanation: str = ""
```

Keep every existing flat field otherwise exactly as-is for the transition (they are removed in
Task 17). All existing constructors pass these explicitly, so adding defaults is
backward-compatible.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/core/exercise_types/test_envelope.py -v`
Expected: PASS.

- [ ] **Step 5: Type check + commit**

Run: `.venv/bin/python -m mypy api`
Expected: no new errors.

```bash
git add api/core/learning/session.py tests/core/exercise_types/test_envelope.py
git commit -m "feat(exercise): add typed payload field to ExerciseRecord (expand phase)"
```

---

## Task 7: Build payload at generation time in `exercise_gen.generate_exercise`

The LM output model only exists inside `exercise_gen.generate_exercise`; the prefetch cache
and service downstream see only a dict. So the payload must be built here and carried in the
returned dict under a `"payload"` key (as `model_dump()`, JSON-safe for the cache).

**Files:**
- Modify: `api/core/learning/exercise_gen.py:70-104` (`generate_exercise`)
- Test: `tests/core/test_exercise_gen_payload.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/core/test_exercise_gen_payload.py
from unittest.mock import patch

from api.core.learning import exercise_gen
from api.core.learning.exercise_types import ExerciseType, TrueFalseOutput


def test_generate_exercise_includes_nested_payload(monkeypatch):
    monkeypatch.setattr(exercise_gen, "select_exercise_type", lambda *_a, **_k: ExerciseType.TRUE_FALSE)

    fake = TrueFalseOutput(
        question="Đúng hay sai?",
        statement="Số 2 là số nguyên tố.",
        correct_answer=True,
        explanation_correct="Đúng",
        explanation_incorrect="Sai",
    )
    with patch.object(exercise_gen, "_invoke_structured_llm", return_value=fake):
        data = exercise_gen.generate_exercise("Số nguyên tố", "def", 1)

    assert data is not None
    assert data["payload"] == {
        "exercise_type": "true_false",
        "statement": "Số 2 là số nguyên tố.",
        "correct_answer": True,
    }
    # Legacy flat fields still present during the transition.
    assert data["statement"] == "Số 2 là số nguyên tố."
```

- [ ] **Step 2: Run it to verify it fails**

Run: `.venv/bin/python -m pytest tests/core/test_exercise_gen_payload.py -v`
Expected: FAIL with `KeyError: 'payload'`.

- [ ] **Step 3: Build payload from the output model**

In `api/core/learning/exercise_gen.py`, update the import line (currently
`from .exercise_types import ExerciseType, ShortAnswerEvaluationOutput, select_exercise_type`):

```python
from .exercise_types import ExerciseType, ShortAnswerEvaluationOutput, select_exercise_type
from .exercise_types.registry import get_handler
```

Replace the final `return cast(...)` block of `generate_exercise` with:

```python
    serialized = serializer(result)
    payload = get_handler(exercise_type).payload_from_output(result)
    serialized["payload"] = payload.model_dump(mode="json")
    return cast(
        "dict[str, str | bool | list[str] | dict[str, str] | list[dict[str, str]]] | None",
        serialized,
    )
```

(`serializer` is `serialize_exercise_result`, still returning the legacy flat dict during the
transition; we add `payload` alongside it.)

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/core/test_exercise_gen_payload.py -v`
Expected: PASS.

- [ ] **Step 5: Type check + commit**

Run: `.venv/bin/python -m mypy api`
Expected: no new errors.

```bash
git add api/core/learning/exercise_gen.py tests/core/test_exercise_gen_payload.py
git commit -m "feat(exercise): build nested payload at generation time"
```

---

## Task 8: Populate `ExerciseRecord.payload` in the service

`exercise_service.generate_exercise` builds `ExerciseRecord` from the flat `exercise_data`
dict (lines 361-382). Add payload reconstruction from `exercise_data["payload"]` via strict
validate, so records created in this session carry a typed payload.

**Files:**
- Modify: `api/core/learning/exercise_service.py:361-382`
- Test: `tests/core/test_exercise_service_payload.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/core/test_exercise_service_payload.py
import pytest

from api.core.learning.exercise_types.payloads import ExercisePayload
from pydantic import TypeAdapter

_adapter = TypeAdapter(ExercisePayload)


def test_payload_validates_from_generation_dict():
    # The exact dict shape generate_exercise now emits under "payload".
    data = {"exercise_type": "mcq", "options": {"A": "1", "B": "2", "C": "3", "D": "4"}, "correct_option": "A"}
    payload = _adapter.validate_python(data)
    assert payload.exercise_type == "mcq"
    assert payload.correct_option == "A"
```

(A focused validate test; the service wiring is covered by the regression suite in Task 17.)

- [ ] **Step 2: Run it to verify it passes (payload union already exists)**

Run: `.venv/bin/python -m pytest tests/core/test_exercise_service_payload.py -v`
Expected: PASS (this locks the dict shape the service will consume).

- [ ] **Step 3: Wire payload into the record constructor**

In `api/core/learning/exercise_service.py`, add near the top imports:

```python
from pydantic import TypeAdapter
from .exercise_types.payloads import ExercisePayload

_PAYLOAD_ADAPTER = TypeAdapter(ExercisePayload)
```

In `generate_exercise`, immediately before `exercise = ExerciseRecord(`, add:

```python
            payload = _PAYLOAD_ADAPTER.validate_python(exercise_data["payload"])
```

Then add `payload=payload,` as the first content argument inside the `ExerciseRecord(...)`
constructor (right after `question=exercise_data["question"],`). Keep the existing flat
kwargs for the transition.

- [ ] **Step 4: Run the service suite**

Run: `.venv/bin/python -m pytest tests/core/test_exercise_service.py tests/core/test_exercise_service_payload.py -v`
Expected: PASS. (Existing service tests feed dicts without `payload`; see Step 5.)

- [ ] **Step 5: Update existing service-test fixtures to include `payload`**

In `tests/core/test_exercise_service.py`, the `fake_generate_exercise_dedup` (around line 47)
returns a dict without `payload`. Add the key so the new constructor line works:

```python
        return {
            "question": "Q",
            "options": {"A": "1", "B": "2", "C": "3", "D": "4"},
            "correct_option": "A",
            "explanation_correct": "ok",
            "explanation_incorrect": "no",
            "payload": {
                "exercise_type": "mcq",
                "options": {"A": "1", "B": "2", "C": "3", "D": "4"},
                "correct_option": "A",
            },
        }
```

Re-run: `.venv/bin/python -m pytest tests/core/test_exercise_service.py -v`
Expected: PASS.

- [ ] **Step 6: Type check + commit**

Run: `.venv/bin/python -m mypy api`
Expected: no new errors.

```bash
git add api/core/learning/exercise_service.py tests/core/test_exercise_service.py tests/core/test_exercise_service_payload.py
git commit -m "feat(exercise): populate typed payload on records built by the service"
```

---

## Task 9: Delegate grading + answer-history to the handler in `answer_eval`

Replace the two if-chains in `answer_eval.py` with handler delegation. The handler reads
content via `exercise.payload`.

**Files:**
- Modify: `api/core/learning/answer_eval.py` (`evaluate_answer`, `serialize_answer_for_history`)
- Test: `tests/core/test_answer_eval_delegates.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/core/test_answer_eval_delegates.py
from api.core.learning.answer_eval import evaluate_answer, serialize_answer_for_history
from api.core.learning.session import ExerciseRecord
from api.core.learning.exercise_types.payloads import TrueFalsePayload


def _tf_record():
    return ExerciseRecord(
        exercise_id="ex1", concept_idx=0, concept_name="C", bloom_level=1,
        question="Đúng hay sai?",
        payload=TrueFalsePayload(statement="S", correct_answer=True),
    )


def test_evaluate_true_false_via_handler():
    assert evaluate_answer(_tf_record(), {"boolean": True}) == (True, "True")
    assert evaluate_answer(_tf_record(), {"boolean": False}) == (False, "False")


def test_serialize_answer_via_handler():
    assert serialize_answer_for_history(_tf_record(), {"boolean": True}) == "True"
```

- [ ] **Step 2: Run it to verify it fails**

Run: `.venv/bin/python -m pytest tests/core/test_answer_eval_delegates.py -v`
Expected: FAIL — current `evaluate_answer` reads `exercise.correct_answer` (flat) and the
record built here only set the payload, so the assertions diverge / AttributeError on the
flat-only path.

- [ ] **Step 3: Rewrite both functions as thin delegators**

Replace the bodies of `evaluate_answer` and `serialize_answer_for_history` in
`api/core/learning/answer_eval.py` with:

```python
def serialize_answer_for_history(exercise: ExerciseRecord, answer: dict[str, Any]) -> str | None:
    from .exercise_types.registry import get_handler

    return get_handler(exercise.payload.exercise_type).serialize_answer(exercise, answer)


def evaluate_answer(
    exercise: ExerciseRecord,
    answer: dict[str, Any],
    *,
    short_answer_grader: Callable[..., dict[str, bool | str | int]] | None = None,
) -> tuple[bool, str]:
    from .exercise_types.registry import get_handler

    handler = get_handler(exercise.payload.exercise_type, short_answer_grader=short_answer_grader)
    return handler.evaluate(exercise, answer)
```

Remove the now-dead `normalize_text` usages only if nothing else imports them; `normalize_text`
itself stays (handlers import it). Keep the `ExerciseType` import only if still referenced;
otherwise drop it to satisfy ruff.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/core/test_answer_eval_delegates.py -v`
Expected: PASS.

- [ ] **Step 5: Update the legacy `test_exercise_types.py` evaluate tests**

`test_evaluate_answer_handles_true_false_fill_blank_multi_correct_and_ordering` and
`test_evaluate_answer_updates_short_answer_feedback` build `SimpleNamespace` records with flat
fields. Give each a `payload` instead. Example for the true_false case:

```python
        from api.core.learning.exercise_types.payloads import TrueFalsePayload
        true_false = SimpleNamespace(
            payload=TrueFalsePayload(statement="S", correct_answer=True),
            concept_name="Concept", question="Question",
            explanation_correct="", explanation_incorrect="",
        )
```

Apply the analogous payload (`FillBlankPayload`, `MultiCorrectPayload`, `OrderingPayload`,
`MatchingPayload`, `ShortAnswerPayload`) to each namespace in those two tests. Drop the flat
`exercise_type`/`correct_answer`/`correct_option`/`rubric` kwargs.

Run: `.venv/bin/python -m pytest tests/core/test_exercise_types.py -v`
Expected: PASS.

- [ ] **Step 6: Type check + commit**

Run: `.venv/bin/python -m mypy api`
Expected: no new errors.

```bash
git add api/core/learning/answer_eval.py tests/core/test_answer_eval_delegates.py tests/core/test_exercise_types.py
git commit -m "refactor(exercise): delegate grading and answer-history to handlers"
```

---

## Task 10: Route generate endpoint + tutor-context through handlers in `routers/session.py`

Replace `_resolve_exercise_question` / `_resolve_exercise_options` (the Bug #1 hotspot) with
handler calls, and build the generate-endpoint response via `to_response_dict`.

**Files:**
- Modify: `api/routers/session.py` (`_resolve_exercise_question`, `_resolve_exercise_options`,
  generate endpoint ~234-255, chat endpoint ~336-385)
- Test: `tests/test_session_router_chat.py`, `tests/core/test_tutor_chat.py`

- [ ] **Step 1: Write the failing test (true_false tutor content is surfaced)**

```python
# tests/test_session_router_tutor_content.py
from types import SimpleNamespace

from api.routers.session import _resolve_exercise_question, _resolve_exercise_options
from api.core.learning.exercise_types.payloads import TrueFalsePayload


def _tf():
    return SimpleNamespace(
        exercise_id="ex1", question="Đánh giá phát biểu sau là đúng hay sai.",
        concept_name="Tích phân", bloom_level=2,
        payload=TrueFalsePayload(
            statement="Tích phân luôn là diện tích.", correct_answer=False,
        ),
    )


def test_tutor_question_includes_statement():
    q = _resolve_exercise_question(_tf())
    assert "Tích phân luôn là diện tích." in q  # Bug #1: statement must reach the tutor


def test_tutor_options_are_true_false():
    assert _resolve_exercise_options(_tf()) == ["True", "False"]
```

- [ ] **Step 2: Run it to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_session_router_tutor_content.py -v`
Expected: FAIL — current helpers read `exercise.statement`/`exercise.options` (flat), which
the payload-only namespace doesn't have.

- [ ] **Step 3: Replace both helpers with handler delegation**

In `api/routers/session.py`, replace the whole bodies of `_resolve_exercise_question` and
`_resolve_exercise_options` with:

```python
def _resolve_exercise_question(exercise: Any) -> str:
    from api.core.learning.exercise_types.registry import get_handler

    return get_handler(exercise.payload.exercise_type).tutor_question(exercise)


def _resolve_exercise_options(exercise: Any) -> list[str]:
    from api.core.learning.exercise_types.registry import get_handler

    return get_handler(exercise.payload.exercise_type).tutor_options(exercise)
```

- [ ] **Step 4: Route the generate endpoint response through `to_response_dict`**

In the generate endpoint (the `ExerciseResponse(...)` block ~234-255), replace the manual
field-by-field construction with handler output merged with the envelope fields. Replace the
`return ok(ExerciseResponse(...).model_dump())` block with:

```python
    from api.core.learning.exercise_types.registry import get_handler

    content = get_handler(exercise.payload.exercise_type).to_response_dict(exercise)
    return ok(
        ExerciseResponse(
            exercise_id=exercise.exercise_id,
            concept_name=exercise.concept_name,
            concept_idx=exercise.concept_idx,
            bloom_level=exercise.bloom_level,
            bloom_label=BLOOM_LABELS.get(exercise.bloom_level, "Unknown"),
            exercise_type=exercise.payload.exercise_type,
            question=exercise.question,
            sentence=content.get("sentence"),
            options=content.get("options", {}),
            statement=content.get("statement"),
            hint=content.get("hint"),
            items=content.get("items", []),
            pairs=content.get("pairs", []),
            right_items=content.get("right_items", []),
            step=env_stats["step"],
            max_steps=env_stats["max_steps"],
            theory=exercise.theory,
            recommendation_reason=getattr(session, "_current_recommendation_reason", None),
        ).model_dump()
    )
```

(`to_response_dict` emits exactly the flat content keys `ExerciseResponse` expects; for
`ordering`/`matching` it carries the deterministic display order keyed by `exercise_id`.)

- [ ] **Step 5: Run the router + tutor tests**

Run: `.venv/bin/python -m pytest tests/test_session_router_tutor_content.py tests/test_session_router_chat.py tests/core/test_tutor_chat.py -v`
Expected: PASS.

- [ ] **Step 6: Type check + commit**

Run: `.venv/bin/python -m mypy api`
Expected: no new errors.

```bash
git add api/routers/session.py tests/test_session_router_tutor_content.py
git commit -m "fix(exercise): surface full question content to tutor via handlers (fixes true_false bug)"
```

---

## Task 11: History formatter reads through the payload

`history_formatter._normalize_history_item` reads flat fields off the record. Point it at
`to_response_dict` (canonical content) so it works regardless of flat fields.

**Files:**
- Modify: `api/core/learning/history_formatter.py`
- Test: `tests/core/test_history_formatter_payload.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/core/test_history_formatter_payload.py
import json
from types import SimpleNamespace

from api.core.learning.history_formatter import format_exercise_history
from api.core.learning.exercise_types.payloads import TrueFalsePayload


def test_formatter_includes_statement_from_payload():
    rec = SimpleNamespace(
        exercise_id="ex1", question="Đúng hay sai?", concept_idx=0,
        concept_name="C", bloom_level=1,
        payload=TrueFalsePayload(statement="S-FROM-PAYLOAD", correct_answer=True),
        user_answer=None, is_correct=None,
    )
    out = json.loads(format_exercise_history([rec]))
    assert out[0]["statement"] == "S-FROM-PAYLOAD"
    assert out[0]["exercise_type"] == "true_false"
```

- [ ] **Step 2: Run it to verify it fails**

Run: `.venv/bin/python -m pytest tests/core/test_history_formatter_payload.py -v`
Expected: FAIL — current code reads `getattr(item, "statement", None)` (flat), which is absent.

- [ ] **Step 3: Reroute `_normalize_history_item` through the handler**

Replace `_normalize_history_item` in `api/core/learning/history_formatter.py` with:

```python
def _normalize_history_item(item: Any) -> dict[str, Any]:
    if isinstance(item, dict):
        return item

    from api.core.learning.exercise_types.registry import get_handler

    payload = getattr(item, "payload", None)
    base: dict[str, Any] = {
        "question": getattr(item, "question", ""),
        "bloom_level": getattr(item, "bloom_level", None),
    }
    if payload is None:
        return base

    content = get_handler(payload.exercise_type).to_response_dict(item)
    base["exercise_type"] = payload.exercise_type.value
    for key in (
        "statement", "sentence", "hint", "options", "items", "pairs",
        "right_items", "rubric", "correct_option", "correct_answer",
    ):
        value = content.get(key)
        if value not in (None, "", [], {}):
            base[key] = value
    return base
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/core/test_history_formatter_payload.py -v`
Expected: PASS.

- [ ] **Step 5: Type check + commit**

Run: `.venv/bin/python -m mypy api`
Expected: no new errors.

```bash
git add api/core/learning/history_formatter.py tests/core/test_history_formatter_payload.py
git commit -m "refactor(exercise): history formatter reads content through handler"
```

## Task 12: Rebuild `get_prompt_spec` from handlers; delete `PROMPT_REGISTRY` (P1)

`PROMPT_REGISTRY` duplicated per-type config that now lives on the handler. Rebuild `ExercisePromptSpec` on demand from `get_handler(type)`, then delete the registry dict. `PromptBuilder` is untouched — it still reads `spec.instruction` / `spec.negative_constraints` / `spec.explanation_guidance`.

**Files:**
- Modify: `api/core/learning/prompts/registry.py`
- Modify: `api/core/learning/prompts/__init__.py` (drop `PROMPT_REGISTRY` from re-exports)
- Test: `tests/core/test_prompt_spec_from_handler.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/core/test_prompt_spec_from_handler.py
import importlib

from api.core.learning.exercise_types.models import ExerciseType
from api.core.learning.prompts.registry import get_prompt_spec


def test_prompt_spec_is_built_from_handler():
    spec = get_prompt_spec(ExerciseType.TRUE_FALSE)
    assert spec.schema.__name__ == "TrueFalseOutput"
    assert "Đúng/Sai" in spec.instruction
    assert spec.negative_constraints.strip() != ""
    assert spec.explanation_guidance.strip() != ""


def test_prompt_registry_symbol_is_gone():
    registry = importlib.import_module("api.core.learning.prompts.registry")
    assert not hasattr(registry, "PROMPT_REGISTRY")


def test_package_import_is_cycle_free():
    # Smoke: a fresh import of both packages must not deadlock on the
    # handlers -> prompts.constants -> exercise_types import chain.
    import api.core.learning.exercise_types as et
    import api.core.learning.prompts as pr
    assert hasattr(et, "get_handler")
    assert hasattr(pr, "get_prompt_spec")
```

- [ ] **Step 2: Run it to verify it fails**

Run: `.venv/bin/python -m pytest tests/core/test_prompt_spec_from_handler.py -v`
Expected: FAIL — `test_prompt_registry_symbol_is_gone` fails (still present) and/or the spec test asserts old wiring.

- [ ] **Step 3: Rewrite `registry.py` to build from the handler**

Replace the entire body of `api/core/learning/prompts/registry.py` with:

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from api.core.learning.exercise_types import ExerciseBaseOutput, ExerciseType
from api.core.learning.exercise_types.registry import get_handler

if TYPE_CHECKING:
    from collections.abc import Callable


@dataclass(frozen=True)
class ExercisePromptSpec:
    schema: type[ExerciseBaseOutput]
    instruction: str
    negative_constraints: str
    explanation_guidance: str
    serializer: Callable[
        [ExerciseBaseOutput],
        dict[str, str | bool | list[str] | dict[str, str] | list[dict[str, str]]],
    ]


def get_prompt_spec(exercise_type: ExerciseType) -> ExercisePromptSpec:
    handler = get_handler(exercise_type)
    from api.core.learning.exercise_types import serialize_exercise_result

    return ExercisePromptSpec(
        schema=handler.output_model,
        instruction=handler.prompt_instruction(),
        negative_constraints=handler.negative_constraints(),
        explanation_guidance=handler.explanation_guidance(),
        serializer=serialize_exercise_result,
    )
```

- [ ] **Step 4: Drop `PROMPT_REGISTRY` from `prompts/__init__.py`**

In `api/core/learning/prompts/__init__.py`, change the registry import line from:

```python
from .registry import PROMPT_REGISTRY, ExercisePromptSpec, get_prompt_spec
```
to:
```python
from .registry import ExercisePromptSpec, get_prompt_spec
```

and remove `"PROMPT_REGISTRY",` from `__all__`.

- [ ] **Step 5: Run prompt + generation tests**

Run: `.venv/bin/python -m pytest tests/core/test_prompt_spec_from_handler.py tests/core/test_exercise_gen_llm.py tests/core/test_exercise_gen_retry.py -v`
Expected: PASS.

- [ ] **Step 6: Type check + commit**

Run: `.venv/bin/python -m mypy api`
Expected: no new errors.

```bash
git add api/core/learning/prompts/registry.py api/core/learning/prompts/__init__.py tests/core/test_prompt_spec_from_handler.py
git commit -m "refactor(exercise): build prompt spec from handler, delete PROMPT_REGISTRY"
```

---

## Task 13: Restructure `ExerciseEntry` document to carry nested `payload`

The Beanie document stops carrying 11 flat content fields and carries `payload: dict` instead. The write/read mappers in `subject_progress.py` are updated in Tasks 14–15; this task changes only the document model and is gated by its own test.

**Files:**
- Modify: `api/core/shared/persistence/documents.py:59-82` (`ExerciseEntry`)
- Test: `tests/core/persistence/test_exercise_entry_payload.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/core/persistence/test_exercise_entry_payload.py
from datetime import UTC, datetime

from api.core.shared.persistence.documents import ExerciseEntry


def test_exercise_entry_accepts_nested_payload():
    entry = ExerciseEntry(
        exercise_id="ex1", concept_idx=0, concept_name="C", bloom_level=1,
        question="Q", explanation="", explanation_correct="", explanation_incorrect="",
        payload={"exercise_type": "true_false", "statement": "S", "correct_answer": True},
        timestamp=datetime.now(UTC),
    )
    assert entry.payload["exercise_type"] == "true_false"


def test_exercise_entry_has_no_flat_content_fields():
    names = set(ExerciseEntry.model_fields)
    for flat in ("statement", "sentence", "options", "items", "pairs", "right_items", "rubric", "correct_option", "correct_answer", "hint"):
        assert flat not in names
```

- [ ] **Step 2: Run it to verify it fails**

Run: `.venv/bin/python -m pytest tests/core/persistence/test_exercise_entry_payload.py -v`
Expected: FAIL — `payload` field absent; flat fields still present.

- [ ] **Step 3: Rewrite `ExerciseEntry`**

Replace `ExerciseEntry` (currently `documents.py:59-82`) with:

```python
class ExerciseEntry(BaseModel):
    exercise_id: str
    concept_idx: int
    concept_name: str
    bloom_level: int
    question: str
    explanation: str
    payload: dict[str, Any] = Field(default_factory=dict)
    explanation_correct: str = ""
    explanation_incorrect: str = ""
    theory: dict[str, Any] | None = None
    user_answer: str | None = None
    is_correct: bool | None = None
    timestamp: datetime
```

Removed flat fields: `correct_option`, `exercise_type`, `sentence`, `options`, `statement`, `hint`, `items`, `pairs`, `right_items`, `rubric`, `correct_answer` (all now inside `payload`).

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/core/persistence/test_exercise_entry_payload.py -v`
Expected: PASS.

- [ ] **Step 5: Type check + commit**

Run: `.venv/bin/python -m mypy api`
Expected: no new errors (the mappers in `subject_progress.py` are fixed in Task 14; if mypy flags them now, proceed — Task 14 immediately follows and the suite is run there).

```bash
git add api/core/shared/persistence/documents.py tests/core/persistence/test_exercise_entry_payload.py
git commit -m "refactor(persistence): ExerciseEntry carries nested payload instead of flat fields"
```

---

## Task 14: Write path — `_snapshot_to_document_payload` emits nested payload

`build_subject_progress_snapshot` produces a per-exercise dict; the persistence mapper turns each into an `ExerciseEntry`. Both must carry `payload` (nested, canonical) instead of flat fields. Because Task 6 (expand phase) kept the flat fields on `ExerciseRecord` AND added `payload`, the snapshot can read `ex.payload`.

**Files:**
- Modify: `api/core/learning/subject_progress_snapshot.py:15-44` (per-exercise dict)
- Modify: `api/core/shared/persistence/subject_progress.py:54-60` (`_snapshot_to_document_payload` exercise loop)
- Test: `tests/core/persistence/test_snapshot_payload_roundtrip.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/core/persistence/test_snapshot_payload_roundtrip.py
from types import SimpleNamespace

import numpy as np

from api.core.learning.subject_progress_snapshot import build_subject_progress_snapshot
from api.core.learning.exercise_types.payloads import OrderingPayload


def _session_with_one_ordering_exercise():
    rec = SimpleNamespace(
        exercise_id="ex1", concept_idx=0, concept_name="C", bloom_level=4,
        question="Sắp xếp",
        payload=OrderingPayload(correct_order=["a", "b", "c"]),
        explanation="", explanation_correct="ok", explanation_incorrect="no",
        theory=None, user_answer=None, is_correct=None, timestamp=1.0,
    )
    return SimpleNamespace(
        env=SimpleNamespace(
            get_session_stats=lambda: {"step": 1, "max_steps": 10},
            get_concept_mastery=lambda: np.array([0.5]),
            get_mastery_matrix=lambda: np.zeros((1, 6)),
        ),
        job_id="job1", user_id="u1", session_id="s1", status="active",
        total_correct=0, total_answered=1, concept_map={"C": 0},
        concept_names={"C": "C"}, exercise_history=[rec],
        created_at=0.0, accessed_at=1.0,
    )


def test_snapshot_entry_carries_nested_canonical_payload():
    snap = build_subject_progress_snapshot(_session_with_one_ordering_exercise())
    entry = snap["exercise_history"][0]
    assert entry["payload"]["exercise_type"] == "ordering"
    assert entry["payload"]["correct_order"] == ["a", "b", "c"]
    # no flat content fields leak into the persisted entry
    for flat in ("items", "statement", "options", "right_items"):
        assert flat not in entry
```

- [ ] **Step 2: Run it to verify it fails**

Run: `.venv/bin/python -m pytest tests/core/persistence/test_snapshot_payload_roundtrip.py -v`
Expected: FAIL — snapshot still emits flat fields, no `payload` key.

- [ ] **Step 3: Rewrite the per-exercise dict in `build_subject_progress_snapshot`**

Replace the `history = [ ... ]` comprehension (`subject_progress_snapshot.py:15-44`) with:

```python
    history = [
        {
            "exercise_id": ex.exercise_id,
            "concept_idx": ex.concept_idx,
            "concept_name": ex.concept_name,
            "bloom_level": ex.bloom_level,
            "question": ex.question,
            "payload": ex.payload.model_dump(mode="json"),
            "explanation": ex.explanation,
            "explanation_correct": ex.explanation_correct,
            "explanation_incorrect": ex.explanation_incorrect,
            "theory": ex.theory,
            "user_answer": ex.user_answer,
            "is_correct": ex.is_correct,
            "timestamp": ex.timestamp,
        }
        for ex in session.exercise_history
    ]
```

- [ ] **Step 4: Update `_snapshot_to_document_payload` exercise loop**

In `api/core/shared/persistence/subject_progress.py`, replace the exercise loop (`54-60`) with:

```python
    exercise_history = []
    for entry in snapshot.get("exercise_history") or []:
        payload = dict(entry)
        payload["timestamp"] = epoch_to_utc(payload.get("timestamp"))
        payload["payload"] = normalize_for_bson(payload.get("payload") or {})
        payload["theory"] = normalize_for_bson(payload.get("theory"))
        exercise_history.append(ExerciseEntry(**payload))
```

(The old lines normalizing `correct_answer` are removed; canonical content now lives inside `payload`.)

- [ ] **Step 5: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/core/persistence/test_snapshot_payload_roundtrip.py -v`
Expected: PASS.

- [ ] **Step 6: Type check + commit**

Run: `.venv/bin/python -m mypy api`
Expected: no new errors.

```bash
git add api/core/learning/subject_progress_snapshot.py api/core/shared/persistence/subject_progress.py tests/core/persistence/test_snapshot_payload_roundtrip.py
git commit -m "refactor(persistence): write nested canonical payload in subject progress snapshot"
```

---

## Task 15: Read path — strict `model_validate` in `_document_to_legacy_payload` + `_restore_exercise_records`

The two places that rebuild a runtime exercise from stored data must reconstruct `payload` strictly. `_document_to_legacy_payload` turns a document entry into a dict; `_restore_exercise_records` (`session.py:399-427`) turns that dict into an `ExerciseRecord`. Both move to the typed payload.

**Files:**
- Modify: `api/core/shared/persistence/subject_progress.py:104-108` (`_document_to_legacy_payload` exercise loop)
- Modify: `api/core/learning/session.py:399-427` (`_restore_exercise_records`)
- Test: `tests/core/persistence/test_restore_strict_payload.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/core/persistence/test_restore_strict_payload.py
import pytest
from pydantic import ValidationError

from api.core.learning.session import SessionState


def _doc_entry(payload: dict) -> dict:
    return {
        "exercise_id": "ex1", "concept_idx": 0, "concept_name": "C", "bloom_level": 1,
        "question": "Q", "explanation": "", "explanation_correct": "",
        "explanation_incorrect": "", "payload": payload, "theory": None,
        "user_answer": None, "is_correct": None, "timestamp": 1.0,
    }


def test_restore_builds_typed_payload():
    session = SessionState.__new__(SessionState)
    session.exercise_history = []
    SessionState._restore_exercise_records(
        session,
        [_doc_entry({"exercise_type": "true_false", "statement": "S", "correct_answer": True})],
    )
    rec = session.exercise_history[0]
    assert rec.payload.exercise_type.value == "true_false"
    assert rec.payload.statement == "S"


def test_restore_rejects_malformed_payload():
    session = SessionState.__new__(SessionState)
    session.exercise_history = []
    with pytest.raises(ValidationError):
        SessionState._restore_exercise_records(
            session, [_doc_entry({"exercise_type": "true_false"})]  # missing statement
        )
```

- [ ] **Step 2: Run it to verify it fails**

Run: `.venv/bin/python -m pytest tests/core/persistence/test_restore_strict_payload.py -v`
Expected: FAIL — `_restore_exercise_records` still builds the old flat `ExerciseRecord`.

- [ ] **Step 3: Update `_document_to_legacy_payload` exercise loop**

In `api/core/shared/persistence/subject_progress.py`, the exercise loop (`104-108`) becomes:

```python
    exercise_history = []
    for entry in doc.exercise_history:
        payload = entry.model_dump()
        payload["timestamp"] = utc_to_epoch(entry.timestamp)
        exercise_history.append(payload)
```

(No change in shape beyond `entry` already carrying nested `payload`; this keeps `payload` as a dict for the session layer to validate.)

- [ ] **Step 4: Rewrite `_restore_exercise_records` to validate strictly**

Replace `_restore_exercise_records` (`session.py:399-427`) with:

```python
    @staticmethod
    def _restore_exercise_records(session: "SessionState", prev_history: list[dict]) -> None:
        from api.core.learning.exercise_types.payloads import ExercisePayload
        from pydantic import TypeAdapter

        adapter = TypeAdapter(ExercisePayload)
        for ex in prev_history:
            session.exercise_history.append(
                ExerciseRecord(
                    exercise_id=ex.get("exercise_id", ""),
                    concept_idx=ex["concept_idx"],
                    concept_name=ex.get("concept_name", ""),
                    bloom_level=ex["bloom_level"],
                    question=ex.get("question", ""),
                    payload=adapter.validate_python(ex["payload"]),
                    explanation=ex.get("explanation", ""),
                    explanation_correct=ex.get("explanation_correct", ""),
                    explanation_incorrect=ex.get("explanation_incorrect", ""),
                    theory=ex.get("theory"),
                    user_answer=ex.get("user_answer"),
                    is_correct=ex.get("is_correct"),
                    timestamp=ex.get("timestamp", 0),
                )
            )
```

Note: the `ExercisePayload` adapter is created once per call (cheap). The `ExerciseRecord` constructor here uses ONLY the envelope fields — this is the point where the flat-field kwargs are dropped from the restore path. (Task 17 removes the flat fields from the dataclass itself.)

- [ ] **Step 5: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/core/persistence/test_restore_strict_payload.py -v`
Expected: PASS.

- [ ] **Step 6: Type check + commit**

Run: `.venv/bin/python -m mypy api`
Expected: no new errors.

```bash
git add api/core/shared/persistence/subject_progress.py api/core/learning/session.py tests/core/persistence/test_restore_strict_payload.py
git commit -m "refactor(persistence): strict typed-payload reconstruction on read path"
```

---

## Task 16: One-time idempotent migration script

Converts every legacy flat `exercise_history` entry in `al_subject_progress` to the nested payload shape. This is the ONLY code that understands the old flat shape. Mirrors `scripts/reset_persistence_for_beanie_cutover.py` bootstrap.

**Files:**
- Create: `scripts/migrate_exercise_payload.py`
- Test: `tests/scripts/test_migrate_exercise_payload.py`

- [ ] **Step 1: Write the failing test (pure converter, no DB)**

```python
# tests/scripts/test_migrate_exercise_payload.py
from scripts.migrate_exercise_payload import flat_to_payload, migrate_entry, needs_migration


def _flat_true_false():
    return {
        "exercise_id": "ex1", "concept_idx": 0, "concept_name": "C", "bloom_level": 1,
        "question": "Đúng hay sai?", "exercise_type": "true_false",
        "statement": "S", "correct_answer": True, "options": {},
        "explanation": "", "explanation_correct": "ok", "explanation_incorrect": "no",
        "timestamp": 1.0,
    }


def test_flat_to_payload_true_false():
    payload = flat_to_payload(_flat_true_false())
    assert payload == {"exercise_type": "true_false", "statement": "S", "correct_answer": True}


def test_flat_to_payload_ordering_uses_correct_answer_as_canonical():
    flat = {
        "exercise_type": "ordering", "items": ["x", "y", "z"],
        "correct_answer": ["a", "b", "c"],
    }
    assert flat_to_payload(flat) == {"exercise_type": "ordering", "correct_order": ["a", "b", "c"]}


def test_flat_to_payload_matching_rebuilds_pairs_from_correct_answer():
    flat = {
        "exercise_type": "matching",
        "correct_answer": {"L1": "R1", "L2": "R2"},
        "right_items": ["R2", "R1"],
    }
    assert flat_to_payload(flat) == {
        "exercise_type": "matching",
        "pairs": [{"left": "L1", "right": "R1"}, {"left": "L2", "right": "R2"}],
    }


def test_migrate_entry_strips_flat_keys_and_adds_payload():
    out = migrate_entry(_flat_true_false())
    assert out["payload"] == {"exercise_type": "true_false", "statement": "S", "correct_answer": True}
    for flat in ("statement", "correct_answer", "options", "exercise_type"):
        assert flat not in out
    # envelope fields preserved
    assert out["question"] == "Đúng hay sai?"
    assert out["explanation_correct"] == "ok"


def test_needs_migration_is_idempotent():
    assert needs_migration(_flat_true_false()) is True
    assert needs_migration(migrate_entry(_flat_true_false())) is False
```

- [ ] **Step 2: Run it to verify it fails**

Run: `.venv/bin/python -m pytest tests/scripts/test_migrate_exercise_payload.py -v`
Expected: FAIL with `ModuleNotFoundError: scripts.migrate_exercise_payload`.

- [ ] **Step 3: Implement the script**

```python
"""
migrate_exercise_payload.py — One-time, idempotent migration of al_subject_progress
exercise_history entries from the legacy flat shape to the nested `payload` shape.

Run dry first:   .venv/bin/python scripts/migrate_exercise_payload.py
Apply:           .venv/bin/python scripts/migrate_exercise_payload.py --force
"""

from __future__ import annotations

import argparse
import asyncio
from importlib import import_module
from pathlib import Path
import sys
from typing import Any

from loguru import logger
from pymongo import AsyncMongoClient

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

_ENVELOPE_KEYS = (
    "exercise_id", "concept_idx", "concept_name", "bloom_level", "question",
    "explanation", "explanation_correct", "explanation_incorrect", "theory",
    "user_answer", "is_correct", "timestamp",
)


def _get_settings() -> Any:
    return import_module("api.config").get_settings()


def flat_to_payload(entry: dict[str, Any]) -> dict[str, Any]:
    et = entry.get("exercise_type", "mcq")
    if et == "mcq":
        return {"exercise_type": "mcq", "options": entry.get("options") or {},
                "correct_option": entry.get("correct_option", "")}
    if et == "true_false":
        return {"exercise_type": "true_false", "statement": entry.get("statement") or "",
                "correct_answer": bool(entry.get("correct_answer"))}
    if et == "fill_blank":
        ca = entry.get("correct_answer")
        answers = ca if isinstance(ca, list) else [a for a in [entry.get("correct_option")] if a]
        return {"exercise_type": "fill_blank", "sentence": entry.get("sentence") or "",
                "hint": entry.get("hint") or "", "blank_answers": answers}
    if et == "multi_correct":
        ca = entry.get("correct_answer")
        correct = ca if isinstance(ca, list) else []
        return {"exercise_type": "multi_correct", "options": entry.get("options") or {},
                "correct_options": sorted(correct)}
    if et == "ordering":
        ca = entry.get("correct_answer")
        order = ca if isinstance(ca, list) else (entry.get("items") or [])
        return {"exercise_type": "ordering", "correct_order": order}
    if et == "matching":
        ca = entry.get("correct_answer")
        if isinstance(ca, dict):
            pairs = [{"left": left, "right": right} for left, right in ca.items()]
        else:
            pairs = entry.get("pairs") or []
        return {"exercise_type": "matching", "pairs": pairs}
    if et == "short_answer":
        sample = entry.get("correct_answer") or entry.get("correct_option") or ""
        return {"exercise_type": "short_answer", "rubric": entry.get("rubric") or [],
                "sample_answer": sample if isinstance(sample, str) else ""}
    raise ValueError(f"Unknown exercise_type in legacy entry: {et!r}")


def needs_migration(entry: dict[str, Any]) -> bool:
    return "payload" not in entry


def migrate_entry(entry: dict[str, Any]) -> dict[str, Any]:
    if not needs_migration(entry):
        return entry
    out = {key: entry[key] for key in _ENVELOPE_KEYS if key in entry}
    out["payload"] = flat_to_payload(entry)
    return out


def migrate_document(doc: dict[str, Any]) -> tuple[dict[str, Any], int]:
    history = doc.get("exercise_history") or []
    changed = 0
    new_history = []
    for entry in history:
        if needs_migration(entry):
            changed += 1
            new_history.append(migrate_entry(entry))
        else:
            new_history.append(entry)
    return new_history, changed


async def run(*, force: bool) -> None:
    settings = _get_settings()
    if not settings.mongodb_uri:
        raise RuntimeError("MONGODB_URI is not configured")
    client = AsyncMongoClient(settings.mongodb_uri)
    try:
        db = client["adaptive_learning"]
        collection = db["al_subject_progress"]
        total_docs = 0
        total_entries = 0
        async for doc in collection.find({}):
            new_history, changed = migrate_document(doc)
            if changed == 0:
                continue
            total_docs += 1
            total_entries += changed
            if force:
                await collection.update_one(
                    {"_id": doc["_id"]}, {"$set": {"exercise_history": new_history}}
                )
        verb = "migrated" if force else "would migrate"
        logger.info("{} {} entries across {} documents", verb, total_entries, total_docs)
        if not force:
            logger.info("dry-run only — re-run with --force to apply")
    finally:
        await client.close()


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="Apply the migration (default: dry-run).")
    args = parser.parse_args()
    await run(force=args.force)


if __name__ == "__main__":
    asyncio.run(main())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/scripts/test_migrate_exercise_payload.py -v`
Expected: PASS.

- [ ] **Step 5: Dry-run against the configured DB (read-only — safe)**

Run: `.venv/bin/python scripts/migrate_exercise_payload.py`
Expected: logs "would migrate N entries across M documents" and "dry-run only". No writes.

- [ ] **Step 6: Commit (do NOT run --force here; that is a deploy step)**

```bash
git add scripts/migrate_exercise_payload.py tests/scripts/test_migrate_exercise_payload.py
git commit -m "feat(persistence): add idempotent exercise payload migration script"
```

---

## Task 17: Contract phase — drop flat fields from `ExerciseRecord` + final regression

The expand phase (Task 6) kept flat fields on `ExerciseRecord` so each integration task stayed green. Now every reader goes through `payload`, so remove them. Then update the legacy tests that asserted flat fields and run the whole suite.

**Files:**
- Modify: `api/core/learning/session.py` (`ExerciseRecord` dataclass — remove flat fields)
- Modify: `api/core/learning/exercise_service.py` (drop flat kwargs when building the record)
- Modify: `tests/core/test_exercise_types.py` (SimpleNamespace fixtures → use `payload`)
- Test: full suite

- [ ] **Step 1: Remove flat fields from the `ExerciseRecord` dataclass**

In `api/core/learning/session.py`, the dataclass becomes exactly (matching the spec envelope):

```python
@dataclass
class ExerciseRecord:
    exercise_id: str
    concept_idx: int
    concept_name: str
    bloom_level: int
    question: str
    payload: "ExercisePayload"
    explanation: str = ""
    explanation_correct: str = ""
    explanation_incorrect: str = ""
    theory: dict[str, str | list[str]] | None = None
    user_answer: str | None = None
    is_correct: bool | None = None
    timestamp: float = field(default_factory=time.time)
```

Add the import at the top of `session.py` (outside `TYPE_CHECKING`, since it is a runtime annotation used by the dataclass default machinery only via string — keep it importable for `TypeAdapter` users):

```python
from api.core.learning.exercise_types.payloads import ExercisePayload
```

Removed fields: `exercise_type`, `sentence`, `options`, `statement`, `hint`, `items`, `pairs`, `right_items`, `rubric`, `correct_option`, `correct_answer`.

- [ ] **Step 2: Drop the flat kwargs where the service builds the record**

In `api/core/learning/exercise_service.py` (the `ExerciseRecord(...)` construction Task 8 edited), remove every flat kwarg so only the envelope + `payload=` remain:

```python
        exercise = ExerciseRecord(
            exercise_id=str(uuid.uuid4())[:8],
            concept_idx=concept_idx,
            concept_name=concept_name,
            bloom_level=bloom_level,
            question=exercise_data["question"],
            payload=payload,
            explanation="",
            explanation_correct=exercise_data.get("explanation_correct", ""),
            explanation_incorrect=exercise_data.get("explanation_incorrect", ""),
            theory=None,
        )
```

(`payload` was already built in Task 8 from `exercise_data` via `get_handler(type).payload_from_output(...)` / the record-dict path; this step only deletes the now-unused flat kwargs.)

- [ ] **Step 3: Update legacy fixtures in `test_exercise_types.py`**

The `_evaluate_answer` tests build `SimpleNamespace(exercise_type=..., correct_answer=..., ...)`. Replace each with an envelope carrying a typed `payload`. Example for the true_false case:

```python
        from api.core.learning.exercise_types.payloads import TrueFalsePayload
        true_false = SimpleNamespace(
            payload=TrueFalsePayload(statement="S", correct_answer=True),
            concept_name="Concept", question="Question",
            explanation_correct="", explanation_incorrect="",
        )
        assert service._evaluate_answer(true_false, {"boolean": True}) == (True, "True")
```

Apply the same transform to the fill_blank, multi_correct, ordering, matching, and short_answer cases in that file (each gets the matching payload: `FillBlankPayload`, `MultiCorrectPayload`, `OrderingPayload`, `MatchingPayload`, `ShortAnswerPayload`), dropping the old flat kwargs. The `serialize_exercise_result` tests stay as-is (the shim still accepts output models).

- [ ] **Step 4: Run the FULL suite**

Run: `.venv/bin/python -m pytest -q`
Expected: PASS. Pay attention to `test_session_router_chat.py`, `test_tutor_chat.py`, `test_exercise_service.py`, `test_exercise_types.py`.

- [ ] **Step 5: Full type check + lint**

Run: `.venv/bin/python -m mypy api && .venv/bin/python -m ruff check api scripts`
Expected: clean.

- [ ] **Step 6: Commit**

```bash
git add api/core/learning/session.py api/core/learning/exercise_service.py tests/
git commit -m "refactor(exercise): drop flat content fields from ExerciseRecord (contract phase)"
```

---

## Self-review checklist (for the plan author)

- Every `ExerciseType` enum value (7) has a handler in Task 5 — guarded by the registry test in Task 4/Task 5.
- The `true_false` tutor bug fix is locked by `test_true_false_tutor_surfaces_statement` (Task 5) and the regression assertion (Task 17).
- Expand→contract: flat fields added alongside `payload` in Task 6, removed only in Task 17, so the suite stays green at every commit.
- Strict read-path: Task 15; migration that feeds it: Task 16 (must run before new code serves traffic — see cutover note).
- Deterministic shuffle: Task 3 (helper) + Task 5 (used in `to_response_dict`/`tutor_options`) + determinism assertions.
