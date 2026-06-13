# Exercise Type Handler Contract — Design

**Status:** Approved (brainstorming) — pending implementation plan
**Date:** 2026-06-14
**Scope:** `rinkuzu-ai-api` backend only. No frontend changes, no DB migration.

## Problem

The "contract" for an exercise type is currently scattered across six places, each
re-implementing the same `exercise_type` branch:

1. `exercise_types.py` — enum + Pydantic output models + `serialize_exercise_result` (isinstance chain)
2. `answer_eval.py` — `evaluate_answer` + `serialize_answer_for_history` (two if-chains)
3. `routers/session.py` — `_resolve_exercise_question` + `_resolve_exercise_options`
4. `exercise_service.py` — builds `ExerciseRecord` from a flat dict (maps ~15 fields)
5. `history_formatter.py` — a flat list of optional fields
6. `prompts/registry.py` — per-type generation instructions

Adding a new exercise type means touching all six and silently missing one is easy —
exactly how the `true_false` tutor bug arose (the `statement` field was never surfaced
to the chatbot because the tutor-display branch was forgotten).

Two root causes:

- **Optional-bag record.** `ExerciseRecord` carries every type's content fields as
  nullable attributes (`statement`, `sentence`, `items`, `pairs`, `right_items`,
  `rubric`, ...). Each new type adds more optionals.
- **Type-lossy round trip.** Pydantic output model → `serialize_exercise_result` →
  flat dict → rebuilt `ExerciseRecord` (type information lost) → `evaluate` reads flat
  fields. The flatness is what makes per-type logic spread out and fragile.

## Goal

A single contract — `ExerciseTypeHandler` — that owns every per-type behaviour, plus a
typed `payload` replacing the optional-bag. Adding a future exercise type must touch
**exactly three things in one package**: one output model, one payload model, one
handler class. `ExerciseRecord`, the router, the service, persistence, and the frontend
stay untouched.

## Decisions (locked during brainstorming)

| # | Decision | Choice |
|---|----------|--------|
| Approach | Strategy + Registry vs typed payload vs per-type record | **Typed payload + handler codec (Y)** |
| A1 | Which behaviours the handler owns | **All five** (generate, serialize, evaluate, tutor-context, answer-history) |
| B1 | Handler owns the LM output model | **Yes** — `output_model` is a ClassVar on the handler |
| C1 | Runtime input for evaluate/tutor/history | **Keep `ExerciseRecord`** as the shared runtime envelope |
| Grader | How `short_answer` gets its LM grader | **Injected via handler constructor** |
| D1 | Old DB/session records | **Read-compatible codec** (`payload_from_record_dict`), no migration |

## Architecture

### Data model

Discriminated union of typed payloads — each payload holds only the fields its type needs:

```python
# exercise_types/payloads.py
class MCQPayload(BaseModel):
    exercise_type: Literal[ExerciseType.MCQ]
    options: dict[str, str]
    correct_option: str

class TrueFalsePayload(BaseModel):
    exercise_type: Literal[ExerciseType.TRUE_FALSE]
    statement: str
    correct_answer: bool

# ... one payload per type (fill_blank, multi_correct, ordering, matching, short_answer)
#
# IMPORTANT — freeze display order at creation. `ordering`/`matching` currently
# shuffle inside serialize_exercise_result, which runs once at create time. The
# payload MUST store the already-shuffled display fields so they are stable across
# every later read:
#   OrderingPayload:  items (shuffled display, frozen) + correct_order
#   MatchingPayload:  pairs + left_items + right_items (shuffled display, frozen)
# Shuffling happens once in `payload_from_output`; `to_response_dict` only reads.
# Otherwise the displayed order would change between the generate call, the tutor
# call, and submit — desyncing the learner's view from the graded answer.

ExercisePayload = Annotated[
    Union[MCQPayload, TrueFalsePayload, FillBlankPayload, MultiCorrectPayload,
          OrderingPayload, MatchingPayload, ShortAnswerPayload],
    Field(discriminator="exercise_type"),
]
```

### Envelope: `ExerciseRecord`

Keeps only the shared fields that RL env / persistence / history bind to; content moves
into `payload`:

```python
@dataclass
class ExerciseRecord:
    exercise_id: str
    concept_idx: int
    concept_name: str
    bloom_level: int
    question: str
    payload: ExercisePayload          # replaces 8+ optional content fields
    explanation: str = ""
    explanation_correct: str = ""
    explanation_incorrect: str = ""
    theory: dict[str, str | list[str]] | None = None
    user_answer: str | None = None
    is_correct: bool | None = None
    timestamp: float = field(default_factory=time.time)
```

Removed from `ExerciseRecord`: `sentence`, `options`, `statement`, `hint`, `items`,
`pairs`, `right_items`, `rubric`, `correct_option`, `correct_answer`. These are now
reached through `record.payload` (typed) via the handler.

### Contract: `ExerciseTypeHandler`

```python
class ExerciseTypeHandler(ABC):
    exercise_type: ClassVar[ExerciseType]
    output_model: ClassVar[type[ExerciseBaseOutput]]
    payload_model: ClassVar[type[BaseModel]]

    def __init__(self, *, short_answer_grader: Callable[..., dict] | None = None) -> None:
        self._grader = short_answer_grader

    # 1. generation
    @abstractmethod
    def prompt_instruction(self) -> str: ...

    # 2a. LM output model -> payload (new exercise)
    @abstractmethod
    def payload_from_output(self, result: ExerciseBaseOutput) -> BaseModel: ...

    # 2b. D1: legacy/new flat record dict -> payload (read-compatible)
    @abstractmethod
    def payload_from_record_dict(self, data: dict[str, Any]) -> BaseModel: ...

    # 3a. payload (+ explanations) -> API response dict (SAME shape as today -> frontend unchanged)
    @abstractmethod
    def to_response_dict(self, payload: BaseModel) -> dict[str, Any]: ...

    # 3b. payload (+ explanations) -> persistence dict (SAME flat shape as today -> no DB migration)
    @abstractmethod
    def to_persistence_dict(
        self, payload: BaseModel, explanations: dict[str, str]
    ) -> dict[str, Any]: ...

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
    def serialize_answer(self, exercise: ExerciseRecord, answer: dict[str, Any]) -> str | None: ...
```

**Input convention (removes payload-vs-record ambiguity):**

- **Pure-data methods** take a `payload` (+ explanations dict where needed) and nothing
  else: `to_response_dict`, `to_persistence_dict`. They are total functions of the
  payload — easy to unit-test in isolation, no `ExerciseRecord` required.
- **Behaviour methods** take the full `ExerciseRecord` because they read shared envelope
  fields (`question`, `concept_name`, `correct_answer` via payload, mutate
  `explanation_*` for short_answer): `evaluate`, `tutor_question`, `tutor_options`,
  `serialize_answer`. Each reaches the typed content through `exercise.payload`.

### Registry

```python
# exercise_types/registry.py
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

Handlers are created fresh per call (lightweight; avoids cross-request state). A missing
type raises `KeyError` — caught at the single entry points and surfaced as a clear error.

### Package layout

`exercise_types.py` becomes a package (the current file is ~316 lines and growing):

```
api/core/learning/exercise_types/
  __init__.py      # re-exports so `from api.core.learning.exercise_types import X` keeps working
  models.py        # ExerciseType enum + LM output models (MCQOutput, ...) + ExerciseBaseOutput
  payloads.py      # payload discriminated union
  handlers.py      # one handler class per type, each @register-ed
  registry.py      # register / get_handler
  selection.py     # select_exercise_type + EXERCISE_WEIGHTS + BLOOM_VERBS (unchanged logic)
```

`__init__.py` must re-export every name currently imported from `exercise_types` so no
external import breaks: `ExerciseType`, all `*Output` models, `serialize_exercise_result`
(kept as a thin shim delegating to the registry), `select_exercise_type`,
`EXERCISE_WEIGHTS`, `BLOOM_VERBS`, `join_lines`, `shuffle_ordering_items`,
`ShortAnswerEvaluationOutput`.

## Data flow

**Create exercise** (`exercise_service.generate_exercise`):
LM returns output model → `get_handler(type).payload_from_output(result)` → build
`ExerciseRecord(payload=...)`. (Replaces the ~15-field dict mapping.)

**Serve to API** (`routers/session.py` generate endpoint):
`get_handler(type).to_response_dict(exercise.payload)` → same `ExerciseResponse` shape as
today.

**Submit/grade** (`exercise_service.submit_answer` → `answer_eval`):
`get_handler(type, short_answer_grader=evaluate_short_answer).evaluate(exercise, answer)`.

**Tutor chat** (`routers/session.py` chat endpoint):
`h = get_handler(type)`; `h.tutor_question(exercise)`, `h.tutor_options(exercise)`.
This is where the `true_false`/`fill_blank`/`ordering`/`matching` content is now
guaranteed surfaced — the ABC forces every handler to implement it.

**Persist / history**:
`to_persistence_dict` emits the current flat shape; `history_formatter` reads through the
payload (or the flat dict via `payload_from_record_dict` for legacy entries).

**Read legacy record** (D1): any flat dict (old DB row or in-flight session) →
`payload_from_record_dict(data)` → payload. Accepts both the old flat shape and the new
one, so no migration and no broken live sessions.

## Backwards-compatibility invariants

- `ExerciseResponse` (API schema) shape is **unchanged** — frontend untouched.
- Persistence dict shape is **unchanged** — no DB migration; old rows still load via D1.
- `serialize_exercise_result(result)` remains importable as a thin shim that composes
  the new methods: `h = get_handler(result.exercise_type)`,
  `payload = h.payload_from_output(result)`, then
  `h.to_persistence_dict(payload, explanations_from(result))`. Output equals today's flat
  dict exactly, so any external caller keeps working.

## Error handling

- `get_handler` on an unknown type → `KeyError`; entry points wrap it in the existing
  domain error (`ExerciseGenerationError` on the generate path; validation error on the
  chat path) rather than leaking `KeyError`.
- `short_answer` handler with no grader injected and asked to `evaluate` → raises
  `RuntimeError("short_answer_grader is required ...")`, matching today's behaviour.
- `payload_from_record_dict` on a malformed/partial legacy dict → fills sensible defaults
  where the old code did (e.g. empty options) rather than raising, to keep old sessions
  alive.

## Testing strategy

- **Per-handler unit tests** — for each of the 7 types: `payload_from_output`,
  `payload_from_record_dict` (legacy flat dict + new dict), `to_response_dict` /
  `to_persistence_dict` shape equals the current serializer output (golden), `evaluate`
  (correct + incorrect), `tutor_question` / `tutor_options`, `serialize_answer`.
- **Round-trip test** — output model → payload → response dict matches the current
  `serialize_exercise_result` output exactly (lock the no-frontend-change invariant).
- **Legacy decode test** — a hand-written old flat dict (pre-refactor shape) decodes via
  `payload_from_record_dict` for every type.
- **Registry test** — every `ExerciseType` enum value has a registered handler (guards
  against adding an enum value without a handler).
- **Regression** — existing `test_session_router_chat.py`, `test_tutor_chat.py`, and any
  exercise/answer tests stay green; the `true_false` tutor content now appears in the
  prompt (explicit assertion).

## Out of scope

- Frontend changes (none needed).
- DB migration (none — D1 read-compatibility instead).
- Changing the RL environment, mastery, or selection weights.
- Adding new exercise types (the point is to make that easy *later*).
