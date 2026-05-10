---
name: api
description: "Skill for the Api area of rinkuzu-ai-api. 62 symbols across 16 files."
---

# Api

62 symbols | 16 files | Cohesion: 88%

## When to Use

- Working with code in `api/`
- Understanding how test_count_mastered_concepts_uses_backend_mastery_threshold, reset_mongo, reset_chroma work
- Modifying api-related functionality

## Key Files

| File | Symbols |
|------|---------|
| `api/exceptions.py` | AppError, SessionNotFoundError, SessionCompletedError, ExerciseGenerationError, ServiceUnavailableError (+14) |
| `api/main.py` | _configure_logging, _configure_llm_tracing, _load_models, _init_rag_stores, _init_pipeline (+9) |
| `api/dependencies.py` | get_current_user, get_app_settings, _resolve_state, get_session_manager, get_session_service (+1) |
| `api/config.py` | get_settings, _normalize_endpoint, object_storage_client_endpoint, object_storage_public_base_url |
| `scripts/reset_persistence_for_beanie_cutover.py` | reset_mongo, reset_chroma, main |
| `api/rate_limit.py` | is_admin_request, set_current_rate_limit_request, reset_current_rate_limit_request |
| `api/core/quiz/draft_service.py` | create_draft, _normalize_and_validate_s3_key, _delete_pdf_best_effort |
| `api/core/content_pipeline/application/stages/model_worker.py` | shutdown, shutdown_sentence_transformer_worker |
| `tests/test_history_thresholds.py` | test_count_mastered_concepts_uses_backend_mastery_threshold |
| `tests/core/test_quiz_draft_service.py` | test_quiz_draft_s3_key_must_belong_to_user |

## Entry Points

Start here when exploring this area:

- **`test_count_mastered_concepts_uses_backend_mastery_threshold`** (Function) — `tests/test_history_thresholds.py:4`
- **`reset_mongo`** (Function) — `scripts/reset_persistence_for_beanie_cutover.py:21`
- **`reset_chroma`** (Function) — `scripts/reset_persistence_for_beanie_cutover.py:36`
- **`main`** (Function) — `scripts/reset_persistence_for_beanie_cutover.py:43`
- **`is_admin_request`** (Function) — `api/rate_limit.py:30`

## Key Symbols

| Symbol | Type | File | Line |
|--------|------|------|------|
| `AppError` | Class | `api/exceptions.py` | 12 |
| `SessionNotFoundError` | Class | `api/exceptions.py` | 21 |
| `SessionCompletedError` | Class | `api/exceptions.py` | 26 |
| `ExerciseGenerationError` | Class | `api/exceptions.py` | 31 |
| `ServiceUnavailableError` | Class | `api/exceptions.py` | 36 |
| `PipelineNotFoundError` | Class | `api/exceptions.py` | 41 |
| `PipelineNotCompletedError` | Class | `api/exceptions.py` | 46 |
| `test_count_mastered_concepts_uses_backend_mastery_threshold` | Function | `tests/test_history_thresholds.py` | 4 |
| `reset_mongo` | Function | `scripts/reset_persistence_for_beanie_cutover.py` | 21 |
| `reset_chroma` | Function | `scripts/reset_persistence_for_beanie_cutover.py` | 36 |
| `main` | Function | `scripts/reset_persistence_for_beanie_cutover.py` | 43 |
| `is_admin_request` | Function | `api/rate_limit.py` | 30 |
| `lifespan` | Function | `api/main.py` | 174 |
| `readiness` | Function | `api/main.py` | 361 |
| `health` | Function | `api/main.py` | 370 |
| `info` | Function | `api/main.py` | 379 |
| `get_current_user` | Function | `api/dependencies.py` | 10 |
| `get_app_settings` | Function | `api/dependencies.py` | 28 |
| `get_settings` | Function | `api/config.py` | 210 |
| `test_quiz_draft_s3_key_must_belong_to_user` | Function | `tests/core/test_quiz_draft_service.py` | 11 |

## Execution Flows

| Flow | Type | Steps |
|------|------|-------|
| `Generate_quiz_tutor_response → Get_settings` | cross_community | 9 |
| `Create_quiz_tutor_stream → Get_settings` | cross_community | 9 |
| `Ask_ai_about_quiz → Get_settings` | cross_community | 9 |
| `Create_tutor_chat_stream → Get_settings` | cross_community | 7 |
| `Generate_tutor_chat_response → Get_settings` | cross_community | 7 |
| `Run_content_pipeline → Get_settings` | cross_community | 7 |
| `Generate_exercise → Get_settings` | cross_community | 6 |
| `Evaluate_short_answer → Get_settings` | cross_community | 6 |
| `Generate_theory → Get_settings` | cross_community | 6 |
| `Create_quiz_draft → Get_settings` | cross_community | 5 |

## Connected Areas

| Area | Connections |
|------|-------------|
| Content_pipeline | 1 calls |

## How to Explore

1. `gitnexus_context({name: "test_count_mastered_concepts_uses_backend_mastery_threshold"})` — see callers and callees
2. `gitnexus_query({query: "api"})` — find related execution flows
3. Read key files listed above for implementation details
