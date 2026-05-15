---
name: quiz
description: "Skill for the Quiz area of rinkuzu-ai-api. 46 symbols across 9 files."
---

# Quiz

46 symbols | 9 files | Cohesion: 73%

## When to Use

- Working with code in `api/`
- Understanding how test_get_structured_llm_uses_provider_native_json_schema, get_llm, get_structured_llm work
- Modifying quiz-related functionality

## Key Files

| File | Symbols |
|------|---------|
| `api/core/quiz/tutor_chat.py` | _request_text_response, _try, sanitize_chat_input, validate_chat_input, normalize_chat_history (+9) |
| `api/core/shared/llm.py` | _normalize_openai_base_url, _ngrok_headers, get_llm, get_structured_llm, extract_llm_text (+3) |
| `api/core/quiz/quiz_tutor.py` | _request_quiz_tutor_text, _try, _open_quiz_tutor_stream, _resolve_quiz_tutor_model, _build_input_message (+3) |
| `api/core/quiz/draft_service.py` | _require_processing_settings, process_draft, _mark_failed, QuizDraftServiceError, QuizDraftNotFoundError (+2) |
| `api/core/quiz/extraction.py` | _invoke_pdf_extract_llm_sync, build_extraction_prompt, invoke_pdf_extract_llm, to_public_dict |
| `tests/core/test_exercise_gen_retry.py` | test_get_structured_llm_uses_provider_native_json_schema, test_resolve_retry_policy_uses_backend_settings |
| `api/core/content_pipeline/infrastructure/graph/cycle_removal.py` | __init__ |
| `api/routers/quiz_tutor.py` | ask_ai_about_quiz |
| `tests/core/test_quiz_extraction.py` | test_extracted_quiz_question_requires_single_correct_index |

## Entry Points

Start here when exploring this area:

- **`test_get_structured_llm_uses_provider_native_json_schema`** (Function) — `tests/core/test_exercise_gen_retry.py:53`
- **`get_llm`** (Function) — `api/core/shared/llm.py:145`
- **`get_structured_llm`** (Function) — `api/core/shared/llm.py:177`
- **`extract_llm_text`** (Function) — `api/core/shared/llm.py:217`
- **`sanitize_chat_input`** (Function) — `api/core/quiz/tutor_chat.py:54`

## Key Symbols

| Symbol | Type | File | Line |
|--------|------|------|------|
| `QuizDraftServiceError` | Class | `api/core/quiz/draft_service.py` | 36 |
| `QuizDraftNotFoundError` | Class | `api/core/quiz/draft_service.py` | 40 |
| `QuizDraftValidationError` | Class | `api/core/quiz/draft_service.py` | 44 |
| `QuizDraftDependencyError` | Class | `api/core/quiz/draft_service.py` | 48 |
| `test_get_structured_llm_uses_provider_native_json_schema` | Function | `tests/core/test_exercise_gen_retry.py` | 53 |
| `get_llm` | Function | `api/core/shared/llm.py` | 145 |
| `get_structured_llm` | Function | `api/core/shared/llm.py` | 177 |
| `extract_llm_text` | Function | `api/core/shared/llm.py` | 217 |
| `sanitize_chat_input` | Function | `api/core/quiz/tutor_chat.py` | 54 |
| `validate_chat_input` | Function | `api/core/quiz/tutor_chat.py` | 58 |
| `normalize_chat_history` | Function | `api/core/quiz/tutor_chat.py` | 85 |
| `summarize_chat_history` | Function | `api/core/quiz/tutor_chat.py` | 133 |
| `build_chat_context` | Function | `api/core/quiz/tutor_chat.py` | 161 |
| `build_tutor_prompt` | Function | `api/core/quiz/tutor_chat.py` | 182 |
| `create_tutor_chat_stream` | Function | `api/core/quiz/tutor_chat.py` | 259 |
| `generate_tutor_chat_response` | Function | `api/core/quiz/tutor_chat.py` | 323 |
| `test_resolve_retry_policy_uses_backend_settings` | Function | `tests/core/test_exercise_gen_retry.py` | 8 |
| `resolve_retry_policy` | Function | `api/core/shared/llm.py` | 46 |
| `awith_llm_retry` | Function | `api/core/shared/llm.py` | 102 |
| `ask_ai_about_quiz` | Function | `api/routers/quiz_tutor.py` | 24 |

## Execution Flows

| Flow | Type | Steps |
|------|------|-------|
| `Generate_quiz_tutor_response → Get_settings` | cross_community | 9 |
| `Create_quiz_tutor_stream → Get_settings` | cross_community | 9 |
| `Ask_ai_about_quiz → Get_settings` | cross_community | 9 |
| `Ask_ai_about_quiz → _normalize_openai_base_url` | cross_community | 9 |
| `Ask_ai_about_quiz → _ngrok_headers` | cross_community | 9 |
| `Generate_quiz_tutor_response → _normalize_openai_base_url` | cross_community | 8 |
| `Generate_quiz_tutor_response → _ngrok_headers` | cross_community | 8 |
| `Ask_ai_about_quiz → Sanitize_chat_input` | cross_community | 8 |
| `Ask_ai_about_quiz → Extract_llm_text` | cross_community | 8 |
| `Generate_quiz_tutor_response → Sanitize_chat_input` | cross_community | 7 |

## Connected Areas

| Area | Connections |
|------|-------------|
| Api | 13 calls |
| Learning | 2 calls |
| Routers | 1 calls |

## How to Explore

1. `gitnexus_context({name: "test_get_structured_llm_uses_provider_native_json_schema"})` — see callers and callees
2. `gitnexus_query({query: "quiz"})` — find related execution flows
3. Read key files listed above for implementation details
