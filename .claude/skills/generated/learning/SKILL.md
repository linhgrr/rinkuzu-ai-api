---
name: learning
description: "Skill for the Learning area of rinkuzu-ai-api. 114 symbols across 13 files."
---

# Learning

114 symbols | 13 files | Cohesion: 85%

## When to Use

- Working with code in `api/`
- Understanding how get_prereq_ok_mask, reset, inject_history work
- Modifying learning-related functionality

## Key Files

| File | Symbols |
|------|---------|
| `api/core/learning/session.py` | _build_id_to_concept_map, _build_concept_info_from_data, _get_text_encoder, _encode_concepts, _build_external_embeddings (+20) |
| `api/core/learning/environment.py` | _decode_action, _compute_prereq_ok_mask, get_prereq_ok_mask, _build_history_tensors, _compute_hidden_state (+16) |
| `api/core/learning/exercise_service.py` | _normalize_text, _evaluate_answer, close, _recent_examples_fingerprint, _generate_exercise_dedup (+16) |
| `api/core/learning/exercise_types.py` | select_exercise_type, ExerciseBaseOutput, MCQOutput, TrueFalseOutput, FillBlankOutput (+7) |
| `api/core/learning/models.py` | load_saint_model, load_dqn_model, _make_causal_mask, _responses_to_idx, forward (+3) |
| `tests/core/test_exercise_types.py` | test_select_exercise_type_covers_new_bloom_mapping, test_select_exercise_type_uses_correct_weights_for_mastery, test_evaluate_answer_handles_true_false_fill_blank_multi_correct_and_ordering, test_evaluate_answer_updates_short_answer_feedback, test_serialize_exercise_result_normalizes_fill_blank_and_matching_payloads (+2) |
| `api/core/learning/exercise_gen.py` | _build_generation_spec, _invoke_structured_llm, generate_exercise, evaluate_short_answer, generate_theory |
| `api/core/shared/llm.py` | sleep_before_retry, with_llm_retry, _resolve_shared_llm_model |
| `tests/core/test_exercise_service.py` | test_exercise_service_uses_separate_request_and_prefetch_timeouts, test_eager_prefetch_uses_prefetch_timeout, test_get_recent_same_concept_exercises_respects_setting_and_order |
| `api/core/learning/answer_eval.py` | normalize_text, evaluate_answer, serialize_answer_for_history |

## Entry Points

Start here when exploring this area:

- **`get_prereq_ok_mask`** (Function) — `api/core/learning/environment.py:186`
- **`reset`** (Function) — `api/core/learning/environment.py:335`
- **`inject_history`** (Function) — `api/core/learning/environment.py:360`
- **`step`** (Function) — `api/core/learning/environment.py:398`
- **`action_masks`** (Function) — `api/core/learning/environment.py:494`

## Key Symbols

| Symbol | Type | File | Line |
|--------|------|------|------|
| `ExerciseBaseOutput` | Class | `api/core/learning/exercise_types.py` | 38 |
| `MCQOutput` | Class | `api/core/learning/exercise_types.py` | 54 |
| `TrueFalseOutput` | Class | `api/core/learning/exercise_types.py` | 60 |
| `FillBlankOutput` | Class | `api/core/learning/exercise_types.py` | 68 |
| `MultiCorrectOutput` | Class | `api/core/learning/exercise_types.py` | 87 |
| `OrderingOutput` | Class | `api/core/learning/exercise_types.py` | 97 |
| `MatchingOutput` | Class | `api/core/learning/exercise_types.py` | 112 |
| `ShortAnswerOutput` | Class | `api/core/learning/exercise_types.py` | 117 |
| `get_prereq_ok_mask` | Function | `api/core/learning/environment.py` | 186 |
| `reset` | Function | `api/core/learning/environment.py` | 335 |
| `inject_history` | Function | `api/core/learning/environment.py` | 360 |
| `step` | Function | `api/core/learning/environment.py` | 398 |
| `action_masks` | Function | `api/core/learning/environment.py` | 494 |
| `get_mastery_matrix` | Function | `api/core/learning/environment.py` | 512 |
| `get_concept_mastery` | Function | `api/core/learning/environment.py` | 518 |
| `test_select_exercise_type_covers_new_bloom_mapping` | Function | `tests/core/test_exercise_types.py` | 18 |
| `test_select_exercise_type_uses_correct_weights_for_mastery` | Function | `tests/core/test_exercise_types.py` | 28 |
| `test_resolve_exercise_llm_model_prefers_exercise_specific_override` | Function | `tests/core/test_exercise_gen_retry.py` | 23 |
| `test_resolve_exercise_llm_model_falls_back_to_openai_model` | Function | `tests/core/test_exercise_gen_retry.py` | 38 |
| `sleep_before_retry` | Function | `api/core/shared/llm.py` | 54 |

## Execution Flows

| Flow | Type | Steps |
|------|------|-------|
| `Create_session → _build_history_tensors` | cross_community | 7 |
| `Generate_exercise → _normalize_history_item` | cross_community | 6 |
| `Generate_exercise → Get_settings` | cross_community | 6 |
| `Eager_generate_first_exercise → _normalize_history_item` | cross_community | 6 |
| `Evaluate_short_answer → Get_settings` | cross_community | 6 |
| `Generate_theory → Get_settings` | cross_community | 6 |
| `_prefetch_branch → _normalize_history_item` | cross_community | 6 |
| `Get_or_recover_session → _get_text_encoder` | cross_community | 5 |
| `Generate_exercise → _recent_examples_fingerprint` | cross_community | 5 |
| `Generate_exercise → _normalize_openai_base_url` | cross_community | 5 |

## Connected Areas

| Area | Connections |
|------|-------------|
| Quiz | 2 calls |
| Api | 1 calls |

## How to Explore

1. `gitnexus_context({name: "get_prereq_ok_mask"})` — see callers and callees
2. `gitnexus_query({query: "learning"})` — find related execution flows
3. Read key files listed above for implementation details
