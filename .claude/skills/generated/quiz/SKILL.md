---
name: quiz
description: "Skill for the Quiz area of rinkuzu-ai-api. 44 symbols across 9 files."
---

# Quiz

44 symbols | 9 files | Cohesion: 79%

## When to Use

- Working with code in `api/`
- Understanding how ask_ai_about_quiz, awith_llm_retry, sanitize_chat_input work
- Modifying quiz-related functionality

## Key Files

| File | Symbols |
|------|---------|
| `api/core/quiz/tutor_chat.py` | sanitize_chat_input, validate_chat_input, normalize_chat_history, summarize_chat_history, build_chat_context (+9) |
| `api/core/quiz/quiz_tutor.py` | _resolve_quiz_tutor_model, _build_input_message, _open_quiz_tutor_stream, generate_quiz_tutor_response, create_quiz_tutor_stream (+3) |
| `api/core/shared/llm.py` | awith_llm_retry, _normalize_openai_base_url, _ngrok_headers, get_llm, get_structured_llm (+2) |
| `api/core/quiz/draft_service.py` | _require_processing_settings, process_draft, _mark_failed, QuizDraftServiceError, QuizDraftNotFoundError (+2) |
| `api/core/quiz/extraction.py` | _invoke_pdf_extract_llm_sync, build_extraction_prompt, invoke_pdf_extract_llm, to_public_dict |
| `api/routers/quiz_tutor.py` | ask_ai_about_quiz |
| `tests/core/test_exercise_gen_retry.py` | test_get_structured_llm_uses_provider_native_json_schema |
| `api/core/content_pipeline/infrastructure/graph/cycle_removal.py` | __init__ |
| `tests/core/test_quiz_extraction.py` | test_extracted_quiz_question_requires_single_correct_index |

## Entry Points

Start here when exploring this area:

- **`ask_ai_about_quiz`** (Function) — `api/routers/quiz_tutor.py:24`
- **`awith_llm_retry`** (Function) — `api/core/shared/llm.py:98`
- **`sanitize_chat_input`** (Function) — `api/core/quiz/tutor_chat.py:56`
- **`validate_chat_input`** (Function) — `api/core/quiz/tutor_chat.py:60`
- **`normalize_chat_history`** (Function) — `api/core/quiz/tutor_chat.py:87`

## Key Symbols

| Symbol | Type | File | Line |
|--------|------|------|------|
| `QuizDraftServiceError` | Class | `api/core/quiz/draft_service.py` | 36 |
| `QuizDraftNotFoundError` | Class | `api/core/quiz/draft_service.py` | 40 |
| `QuizDraftValidationError` | Class | `api/core/quiz/draft_service.py` | 44 |
| `QuizDraftDependencyError` | Class | `api/core/quiz/draft_service.py` | 48 |
| `ask_ai_about_quiz` | Function | `api/routers/quiz_tutor.py` | 24 |
| `awith_llm_retry` | Function | `api/core/shared/llm.py` | 98 |
| `sanitize_chat_input` | Function | `api/core/quiz/tutor_chat.py` | 56 |
| `validate_chat_input` | Function | `api/core/quiz/tutor_chat.py` | 60 |
| `normalize_chat_history` | Function | `api/core/quiz/tutor_chat.py` | 87 |
| `summarize_chat_history` | Function | `api/core/quiz/tutor_chat.py` | 135 |
| `build_chat_context` | Function | `api/core/quiz/tutor_chat.py` | 163 |
| `build_tutor_prompt` | Function | `api/core/quiz/tutor_chat.py` | 184 |
| `create_tutor_chat_stream` | Function | `api/core/quiz/tutor_chat.py` | 261 |
| `generate_tutor_chat_response` | Function | `api/core/quiz/tutor_chat.py` | 325 |
| `generate_quiz_tutor_response` | Function | `api/core/quiz/quiz_tutor.py` | 114 |
| `create_quiz_tutor_stream` | Function | `api/core/quiz/quiz_tutor.py` | 162 |
| `test_get_structured_llm_uses_provider_native_json_schema` | Function | `tests/core/test_exercise_gen_retry.py` | 53 |
| `get_llm` | Function | `api/core/shared/llm.py` | 137 |
| `get_structured_llm` | Function | `api/core/shared/llm.py` | 169 |
| `extract_llm_text` | Function | `api/core/shared/llm.py` | 209 |

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
| `Ask_ai_about_quiz → Sanitize_chat_input` | intra_community | 8 |
| `Ask_ai_about_quiz → Extract_llm_text` | cross_community | 8 |
| `Generate_quiz_tutor_response → Sanitize_chat_input` | intra_community | 7 |

## Connected Areas

| Area | Connections |
|------|-------------|
| Api | 12 calls |
| Learning | 3 calls |
| Routers | 1 calls |

## How to Explore

1. `gitnexus_context({name: "ask_ai_about_quiz"})` — see callers and callees
2. `gitnexus_query({query: "quiz"})` — find related execution flows
3. Read key files listed above for implementation details
