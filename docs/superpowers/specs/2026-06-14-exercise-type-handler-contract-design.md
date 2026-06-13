# Exercise Type Handler Contract — Design

**Status:** Approved (brainstorming) — pending implementation plan
**Date:** 2026-06-14
**Scope:** `rinkuzu-ai-api` backend only. No frontend changes. **One-time DB migration.**

## Problem

The "contract" for an exercise type is currently scattered across seven places, each
re-implementing the same `exercise_type` branch:

1. `exercise_types.py` — enum + Pydantic output models + `serialize_exercise_result` (isinstance chain)
2. `answer_eval.py` — `evaluate_answer` + `serialize_answer_for_history` (two if-chains)
3. `routers/session.py` — `_resolve_exercise_question` + `_resolve_exercise_options`
4. `exercise_service.py` — builds `ExerciseRecord` from a flat dict (maps ~15 fields)
5. `history_formatter.py` — a flat list of optional fields
6. `prompts/registry.py` — per-type generation config (`instruction`, `negative_constraints`,
   `explanation_guidance`, `serializer`)
7. `subject_progress_snapshot.py` — `build_subject_progress_snapshot` reads 10 flat
   content fields off each `ExerciseRecord` to build the persistence dict.

Adding a new exercise type means touching all seven and silently missing one is easy —
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
typed `payload` replacing the optional-bag, persisted **nested** in the DB. Adding a
future exercise type must touch **exactly three things in one package**: one output
model, one payload model, one handler class. `ExerciseRecord`, the router, the service,
and the frontend stay untouched.

## Decisions (locked during brainstorming)

| # | Decision | Choice |
|---|----------|--------|
| Approach | Strategy + Registry vs typed payload vs per-type record | **Typed payload + handler codec (Y)** |
| A1 | Which behaviours the handler owns | **All** (generate-config, serialize, evaluate, tutor-context, answer-history) |
| B1 | Handler owns the LM output model | **Yes** — `output_model` is a ClassVar on the handler |
| C1 | Runtime input for evaluate/tutor/history | **Keep `ExerciseRecord`** as the shared runtime envelope |
| Grader | How `short_answer` gets its LM grader | **Injected via handler constructor** |
| P1 | How much prompt config the handler absorbs | **All four** — instruction, negative_constraints, explanation_guidance, output_model. `PROMPT_REGISTRY` is removed. |
| D2 | Old DB records | **Migrate** — DB stores nested `payload`; a one-time idempotent script rewrites old flat rows |
| Read-path | How runtime loads persisted exercises | **Strict** `ExercisePayload.model_validate` — no legacy-flat fallback in the hot path |
| Shuffle | Display order for `ordering`/`matching` | **Deterministic, seeded by `exercise_id`** — never stored; DB holds canonical only |

## Architecture

### Data model — typed payloads (canonical only)

Discriminated union of typed payloads. Each payload holds only the fields its type needs,
and **stores canonical order only** — no shuffled display order is ever persisted.

```python
# exercise_types/payloads.py
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
    blank_answers: list[str]            # accepted answers; [0] is canonical

class MultiCorrectPayload(BaseModel):
    exercise_type: Literal[ExerciseType.MULTI_CORRECT] = ExerciseType.MULTI_CORRECT
    options: dict[str, str]             # A..E
    correct_options: list[str]          # sorted

class OrderingPayload(BaseModel):
    exercise_type: Literal[ExerciseType.ORDERING] = ExerciseType.ORDERING
    correct_order: list[str]            # canonical; display order derived at serve time

class MatchingPayload(BaseModel):
    exercise_type: Literal[ExerciseType.MATCHING] = ExerciseType.MATCHING
    pairs: list[dict[str, str]]         # [{"left":..,"right":..}] canonical, in-order

class ShortAnswerPayload(BaseModel):
    exercise_type: Literal[ExerciseType.SHORT_ANSWER] = ExerciseType.SHORT_ANSWER
    rubric: list[str]
    sample_answer: str

ExercisePayload = Annotated[
    Union[MCQPayload, TrueFalsePayload, FillBlankPayload, MultiCorrectPayload,
          OrderingPayload, MatchingPayload, ShortAnswerPayload],
    Field(discriminator="exercise_type"),
]
```

### Deterministic display shuffle

`ordering` and `matching` need a shuffled display order, but the DB stores only the
canonical order. The shuffle is **derived deterministically from `exercise_id`**, so it
is stable across every serve/tutor/refetch without storing any state:

```python
# exercise_types/shuffle.py
import random

def deterministic_shuffle(items: list[str], seed: str) -> list[str]:
    out = list(items)
    random.Random(seed).shuffle(out)   # seeded PRNG -> same seed, same order
    return out
```

This is also a strict improvement over today: the current `serialize_exercise_result`
shuffles with `SystemRandom` (non-deterministic), so a refetch reorders the display. With
a per-`exercise_id` seed, the learner sees the same order on generate, on every tutor
call, and on submit — and the graded canonical order never moves.

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
    payload: ExercisePayload          # replaces 10 optional content fields
    explanation: str = ""
    explanation_correct: str = ""
    explanation_incorrect: str = ""
    theory: dict[str, str | list[str]] | None = None
    user_answer: str | None = None
    is_correct: bool | None = None
    timestamp: float = field(default_factory=time.time)
```

Removed from `ExerciseRecord`: `exercise_type` (now `payload.exercise_type`), `sentence`,
`options`, `statement`, `hint`, `items`, `pairs`, `right_items`, `rubric`,
`correct_option`, `correct_answer`. All reached through `record.payload` via the handler.

### Contract: `ExerciseTypeHandler`

```python
class ExerciseTypeHandler(ABC):
    exercise_type: ClassVar[ExerciseType]
    output_model: ClassVar[type[ExerciseBaseOutput]]
    payload_model: ClassVar[type[BaseModel]]

    def __init__(self, *, short_answer_grader: Callable[..., dict] | None = None) -> None:
        self._grader = short_answer_grader

    # 1. generation config (P1 — replaces PROMPT_REGISTRY entry)
    @abstractmethod
    def prompt_instruction(self) -> str: ...
    @abstractmethod
    def negative_constraints(self) -> str: ...
    @abstractmethod
    def explanation_guidance(self) -> str: ...

    # 2. LM output model -> payload (canonical; no shuffle)
    @abstractmethod
    def payload_from_output(self, result: ExerciseBaseOutput) -> BaseModel: ...

    # 3. ExerciseRecord -> API response dict (SAME shape as today; shuffle from exercise_id)
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
    def serialize_answer(self, exercise: ExerciseRecord, answer: dict[str, Any]) -> str | None: ...
```

**No `payload_from_record_dict` and no `to_persistence_dict` in the hot path.** D2 makes
both unnecessary:

- **Persistence is `payload.model_dump()` nested** — the DB stores the typed payload
  directly, so there is no flat-shape codec to write.
- **Read-path is strict** — `ExercisePayload.model_validate(stored_payload)` reconstructs
  the typed payload in one line. No per-type flat decode at runtime.
- The only place that understands the **old flat shape** is the one-time migration script
  (see below), which converts legacy rows once and is then irrelevant to the running app.

**Input convention:** every method takes the full `ExerciseRecord` and reaches content via
`exercise.payload`. `to_response_dict` and `tutor_options` need `exercise.exercise_id` to
derive the deterministic shuffle, so they require the envelope, not just the payload.

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
  payloads.py      # payload discriminated union (ExercisePayload)
  shuffle.py       # deterministic_shuffle(items, seed)
  handlers.py      # one handler class per type, each @register-ed; owns prompt text via constants
  registry.py      # register / get_handler
  selection.py     # select_exercise_type + EXERCISE_WEIGHTS + BLOOM_VERBS (unchanged logic)
```

`__init__.py` must re-export every name currently imported from `exercise_types` so no
external import breaks: `ExerciseType`, all `*Output` models, `select_exercise_type`,
`EXERCISE_WEIGHTS`, `BLOOM_VERBS`, `join_lines`, `shuffle_ordering_items`,
`ShortAnswerEvaluationOutput`, and a `serialize_exercise_result` shim (see compat below).

## Data flow

**Create exercise** (`exercise_service.generate_exercise` → service builds the record):
LM returns output model → `get_handler(type).payload_from_output(result)` → build
`ExerciseRecord(payload=...)`. (Replaces the ~15-field dict mapping.)

**Serve to API** (`routers/session.py` generate endpoint):
`get_handler(type).to_response_dict(exercise)` → same `ExerciseResponse` shape as today,
with `ordering`/`matching` display order derived from `exercise_id`.

**Submit/grade** (`exercise_service.submit_answer` → `answer_eval`):
`get_handler(type, short_answer_grader=evaluate_short_answer).evaluate(exercise, answer)`.

**Tutor chat** (`routers/session.py` chat endpoint):
`h = get_handler(type)`; `h.tutor_question(exercise)`, `h.tutor_options(exercise)`.
This is where `true_false`/`fill_blank`/`ordering`/`matching` content is now guaranteed
surfaced — the ABC forces every handler to implement it.

**Prompt generation** (`exercise_gen` / `prompts`):
`get_prompt_spec(type)` is rebuilt from the handler: `schema = handler.output_model`,
`instruction = handler.prompt_instruction()`, `negative_constraints =
handler.negative_constraints()`, `explanation_guidance = handler.explanation_guidance()`.
`PROMPT_REGISTRY` is deleted.

**Persist** (`subject_progress_snapshot.build_subject_progress_snapshot` →
`persistence/subject_progress.py`): each history entry serializes the envelope shared
fields plus `payload = exercise.payload.model_dump()` (nested, canonical). `ExerciseEntry`
(the Beanie document) is restructured to carry `payload: dict` instead of the 11 flat
content fields.

**Read persisted** (`persistence/subject_progress.py` load path):
`_document_to_legacy_payload` reconstructs each history entry by validating
`ExercisePayload.model_validate(entry.payload)` (strict) and rebuilding the runtime
`ExerciseRecord`. No flat fallback.

## Migration

A standalone, idempotent script: `scripts/migrate_exercise_payload.py` (mirrors the
existing `scripts/reset_persistence_for_beanie_cutover.py` bootstrap: `sys.path` insert,
`AsyncMongoClient`, `argparse`, `--force` guard, plus a `--dry-run` default).

- Iterates every `al_subject_progress` document.
- For each `exercise_history` entry still in the **old flat shape** (detected by the
  ABSENCE of a `payload` key), converts flat fields → nested payload using a one-shot
  `flat_to_payload(entry)` converter (the only code that understands the legacy shape),
  then rewrites the entry as `{<shared fields>, "payload": <nested>}` and drops the old
  flat content keys.
- **Idempotent:** an entry that already has a `payload` key is left untouched, so re-runs
  are safe.
- `--dry-run` (default) reports how many documents/entries would change without writing;
  `--force` performs the update.
- Documented run order in the plan: deploy code that can *read* both shapes is NOT
  required because the read-path is strict — therefore the migration must run **before**
  the new code serves traffic (or during a short maintenance window). The plan calls this
  out explicitly.

## Backwards-compatibility invariants

- `ExerciseResponse` (API schema) shape is **unchanged** — frontend untouched.
- `serialize_exercise_result(result)` remains importable as a thin shim that composes the
  new methods so any external caller keeps working:
  `h = get_handler(result.exercise_type)`, `payload = h.payload_from_output(result)`,
  then return the flat dict the old serializer produced (the shim is the one remaining
  flat emitter, kept only for callers that still expect the old return shape; internal
  callers move to `payload_from_output` + `to_response_dict`).

## Error handling

- `get_handler` on an unknown type → `KeyError`; entry points wrap it in the existing
  domain error (`ExerciseGenerationError` on the generate path; validation error on the
  chat path) rather than leaking `KeyError`.
- `short_answer` handler with no grader injected and asked to `evaluate` → raises
  `RuntimeError("short_answer_grader is required ...")`, matching today's behaviour.
- **Strict read-path:** `ExercisePayload.model_validate` on a malformed stored payload
  raises `ValidationError`. Because the migration runs first and is idempotent, persisted
  payloads are well-formed; a validation error therefore signals real corruption and
  should surface (logged) rather than be silently defaulted.

## Testing strategy

- **Per-handler unit tests** — for each of the 7 types: `payload_from_output`,
  `to_response_dict` shape equals the current serializer output (golden), `evaluate`
  (correct + incorrect), `tutor_question` / `tutor_options`, `serialize_answer`,
  `prompt_instruction` / `negative_constraints` / `explanation_guidance` return the
  expected per-type constant text.
- **Deterministic shuffle test** — `to_response_dict` / `tutor_options` for the same
  `exercise_id` produce the SAME display order across repeated calls; different
  `exercise_id`s generally differ; the canonical order in the payload is never mutated.
- **Round-trip test** — output model → payload → response dict matches the current
  `serialize_exercise_result` output for non-shuffled types exactly (lock the
  no-frontend-change invariant); for `ordering`/`matching` the response `items`/`pairs`
  are a permutation of canonical and the `correct_answer` field is canonical.
- **Persistence round-trip test** — `ExerciseRecord` → snapshot entry (nested payload) →
  `ExercisePayload.model_validate` → equal payload. Strict validate succeeds on a
  freshly-written entry.
- **Migration test** — a hand-written OLD flat document converts to the nested shape;
  running the migration twice is a no-op the second time (idempotency); a document already
  in the new shape is untouched.
- **Registry test** — every `ExerciseType` enum value has a registered handler (guards
  against adding an enum value without a handler).
- **Regression** — existing `test_session_router_chat.py`, `test_tutor_chat.py`,
  `test_exercise_types.py`, `test_exercise_service.py` stay green (updated where they
  asserted the old flat `ExerciseRecord` fields); the `true_false` tutor content now
  appears in the prompt (explicit assertion).

## Out of scope

- Frontend changes (none needed).
- Changing the RL environment, mastery, or selection weights.
- Adding new exercise types (the point is to make that easy *later*).
