---
name: prompts
description: "Skill for the Prompts area of rinkuzu-ai-api. 16 symbols across 4 files."
---

# Prompts

16 symbols | 4 files | Cohesion: 90%

## When to Use

- Working with code in `tests/`
- Understanding how test_build_system_message_contains_role_and_math_rules, test_few_shot_included_for_each_type, test_negative_constraints_included work
- Modifying prompts-related functionality

## Key Files

| File | Symbols |
|------|---------|
| `tests/core/test_prompt_builder.py` | test_build_system_message_contains_role_and_math_rules, test_few_shot_included_for_each_type, test_negative_constraints_included, test_meta_validation_checklist_present, test_recent_exercises_dedup_block (+2) |
| `api/core/learning/prompts/builder.py` | build_system_message, build_messages, build_exercise_messages, __init__, _select_few_shots (+1) |
| `tests/core/test_prompt_registry.py` | test_get_prompt_spec_returns_correct_schema, test_spec_has_negative_constraints |
| `api/core/learning/prompts/registry.py` | get_prompt_spec |

## Entry Points

Start here when exploring this area:

- **`test_build_system_message_contains_role_and_math_rules`** (Function) — `tests/core/test_prompt_builder.py:6`
- **`test_few_shot_included_for_each_type`** (Function) — `tests/core/test_prompt_builder.py:55`
- **`test_negative_constraints_included`** (Function) — `tests/core/test_prompt_builder.py:68`
- **`test_meta_validation_checklist_present`** (Function) — `tests/core/test_prompt_builder.py:81`
- **`test_recent_exercises_dedup_block`** (Function) — `tests/core/test_prompt_builder.py:95`

## Key Symbols

| Symbol | Type | File | Line |
|--------|------|------|------|
| `test_build_system_message_contains_role_and_math_rules` | Function | `tests/core/test_prompt_builder.py` | 6 |
| `test_few_shot_included_for_each_type` | Function | `tests/core/test_prompt_builder.py` | 55 |
| `test_negative_constraints_included` | Function | `tests/core/test_prompt_builder.py` | 68 |
| `test_meta_validation_checklist_present` | Function | `tests/core/test_prompt_builder.py` | 81 |
| `test_recent_exercises_dedup_block` | Function | `tests/core/test_prompt_builder.py` | 95 |
| `build_system_message` | Function | `api/core/learning/prompts/builder.py` | 47 |
| `build_messages` | Function | `api/core/learning/prompts/builder.py` | 175 |
| `build_exercise_messages` | Function | `api/core/learning/prompts/builder.py` | 191 |
| `test_get_prompt_spec_returns_correct_schema` | Function | `tests/core/test_prompt_registry.py` | 17 |
| `test_spec_has_negative_constraints` | Function | `tests/core/test_prompt_registry.py` | 27 |
| `get_prompt_spec` | Function | `api/core/learning/prompts/registry.py` | 118 |
| `test_build_user_message_contains_concept_and_bloom` | Function | `tests/core/test_prompt_builder.py` | 20 |
| `test_empty_definition_guard_added_when_short` | Function | `tests/core/test_prompt_builder.py` | 38 |
| `build_user_message` | Function | `api/core/learning/prompts/builder.py` | 103 |
| `__init__` | Function | `api/core/learning/prompts/builder.py` | 32 |
| `_select_few_shots` | Function | `api/core/learning/prompts/builder.py` | 66 |

## Execution Flows

| Flow | Type | Steps |
|------|------|-------|
| `Build_exercise_messages → _normalize_history_item` | cross_community | 5 |
| `Build_exercise_messages → _select_few_shots` | cross_community | 4 |
| `Build_exercise_messages → Build_system_message` | intra_community | 3 |

## Connected Areas

| Area | Connections |
|------|-------------|
| Learning | 1 calls |

## How to Explore

1. `gitnexus_context({name: "test_build_system_message_contains_role_and_math_rules"})` — see callers and callees
2. `gitnexus_query({query: "prompts"})` — find related execution flows
3. Read key files listed above for implementation details
