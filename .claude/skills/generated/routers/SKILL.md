---
name: routers
description: "Skill for the Routers area of rinkuzu-ai-api. 47 symbols across 10 files."
---

# Routers

47 symbols | 10 files | Cohesion: 73%

## When to Use

- Working with code in `api/`
- Understanding how ok, start_session, pipeline_status work
- Modifying routers-related functionality

## Key Files

| File | Symbols |
|------|---------|
| `api/routers/session.py` | start_session, _get_tutor_chat_history, _append_tutor_chat_turn, _build_rag_context, _resolve_exercise_options (+7) |
| `api/routers/history.py` | _to_progress_percent, _count_mastered_concepts, _build_subject_progress_detail, _build_subject_progress_summary, list_subjects (+5) |
| `api/routers/quiz_drafts.py` | _service_error_to_http, create_quiz_draft, list_quiz_drafts, get_quiz_draft, patch_quiz_draft (+2) |
| `api/core/quiz/draft_service.py` | public_draft, get_draft, list_drafts, patch_draft, delete_draft (+1) |
| `api/routers/pipeline.py` | pipeline_status, get_job_status, create_session_from_pipeline, process_document |
| `api/core/shared/url_fetch.py` | _resolve_ips, _is_private_host, validate_download_url, stream_download |
| `api/schemas/common.py` | ok |
| `tests/core/test_quiz_draft_service.py` | test_public_draft_uses_safe_defaults |
| `tests/test_session_router_chat.py` | test_tutor_chat_history_is_scoped_to_current_exercise |
| `api/dependencies.py` | resolve_user_session |

## Entry Points

Start here when exploring this area:

- **`ok`** (Function) — `api/schemas/common.py:9`
- **`start_session`** (Function) — `api/routers/session.py:131`
- **`pipeline_status`** (Function) — `api/routers/pipeline.py:48`
- **`get_job_status`** (Function) — `api/routers/pipeline.py:177`
- **`create_session_from_pipeline`** (Function) — `api/routers/pipeline.py:255`

## Key Symbols

| Symbol | Type | File | Line |
|--------|------|------|------|
| `ok` | Function | `api/schemas/common.py` | 9 |
| `start_session` | Function | `api/routers/session.py` | 131 |
| `pipeline_status` | Function | `api/routers/pipeline.py` | 48 |
| `get_job_status` | Function | `api/routers/pipeline.py` | 177 |
| `create_session_from_pipeline` | Function | `api/routers/pipeline.py` | 255 |
| `list_subjects` | Function | `api/routers/history.py` | 125 |
| `list_subject_progress` | Function | `api/routers/history.py` | 165 |
| `get_subject_history` | Function | `api/routers/history.py` | 186 |
| `list_pipeline_jobs` | Function | `api/routers/history.py` | 203 |
| `get_pipeline_job` | Function | `api/routers/history.py` | 216 |
| `delete_subject` | Function | `api/routers/history.py` | 231 |
| `test_public_draft_uses_safe_defaults` | Function | `tests/core/test_quiz_draft_service.py` | 19 |
| `create_quiz_draft` | Function | `api/routers/quiz_drafts.py` | 39 |
| `list_quiz_drafts` | Function | `api/routers/quiz_drafts.py` | 59 |
| `get_quiz_draft` | Function | `api/routers/quiz_drafts.py` | 73 |
| `patch_quiz_draft` | Function | `api/routers/quiz_drafts.py` | 90 |
| `delete_quiz_draft` | Function | `api/routers/quiz_drafts.py` | 108 |
| `submit_quiz_draft` | Function | `api/routers/quiz_drafts.py` | 125 |
| `public_draft` | Function | `api/core/quiz/draft_service.py` | 52 |
| `get_draft` | Function | `api/core/quiz/draft_service.py` | 149 |

## Execution Flows

| Flow | Type | Steps |
|------|------|-------|
| `Create_quiz_draft → Get_settings` | cross_community | 5 |
| `Process_document → _resolve_ips` | intra_community | 5 |
| `Patch_quiz_draft → Get_draft` | intra_community | 3 |
| `Create_quiz_draft → _normalize_and_validate_s3_key` | cross_community | 3 |
| `Submit_quiz_draft → Get_draft` | intra_community | 3 |
| `Get_knowledge_graph → Resolve_user_session` | cross_community | 3 |
| `Get_knowledge_graph → Ok` | cross_community | 3 |
| `Get_mastery_matrix → Resolve_user_session` | cross_community | 3 |
| `Get_mastery_matrix → Ok` | cross_community | 3 |
| `Get_concept_detail → Resolve_user_session` | cross_community | 3 |

## Connected Areas

| Area | Connections |
|------|-------------|
| Api | 3 calls |
| Quiz | 2 calls |

## How to Explore

1. `gitnexus_context({name: "ok"})` — see callers and callees
2. `gitnexus_query({query: "routers"})` — find related execution flows
3. Read key files listed above for implementation details
