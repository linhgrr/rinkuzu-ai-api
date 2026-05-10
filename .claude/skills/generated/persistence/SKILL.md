---
name: persistence
description: "Skill for the Persistence area of rinkuzu-ai-api. 36 symbols across 8 files."
---

# Persistence

36 symbols | 8 files | Cohesion: 82%

## When to Use

- Working with code in `api/`
- Understanding how create_quiz_draft, load_quiz_draft_for_user, update_quiz_draft_for_user work
- Modifying persistence-related functionality

## Key Files

| File | Symbols |
|------|---------|
| `api/core/shared/persistence/subject_progress.py` | _resolve_concept_indices, _snapshot_to_document_payload, save_subject_progress_snapshot, _document_to_legacy_payload, load_subject_progress_for_user (+4) |
| `api/core/shared/persistence/pipeline_jobs.py` | list_recent_pipeline_jobs, pipeline_job_to_document, save_pipeline_job, _document_to_runtime_payload, load_pipeline_job (+3) |
| `api/core/shared/persistence/quiz_drafts.py` | _normalize_questions, _document_to_public_dict, create_quiz_draft, load_quiz_draft_for_user, update_quiz_draft_for_user (+2) |
| `api/core/shared/persistence/common.py` | _is_numpy_value, normalize_for_bson, ensure_utc, utc_to_epoch, epoch_to_utc (+2) |
| `api/core/shared/persistence/documents.py` | touch_updated_at, touch_updated_at |
| `api/core/content_pipeline/infrastructure/serializers.py` | _to_bson_safe |
| `api/core/shared/persistence/openai_file_cache.py` | save_cached_openai_file |
| `api/core/shared/persistence/document_chunks.py` | delete_chunks_for_job |

## Entry Points

Start here when exploring this area:

- **`create_quiz_draft`** (Function) — `api/core/shared/persistence/quiz_drafts.py:58`
- **`load_quiz_draft_for_user`** (Function) — `api/core/shared/persistence/quiz_drafts.py:84`
- **`update_quiz_draft_for_user`** (Function) — `api/core/shared/persistence/quiz_drafts.py:98`
- **`list_recent_quiz_drafts_for_user`** (Function) — `api/core/shared/persistence/quiz_drafts.py:130`
- **`delete_quiz_draft_for_user`** (Function) — `api/core/shared/persistence/quiz_drafts.py:153`

## Key Symbols

| Symbol | Type | File | Line |
|--------|------|------|------|
| `create_quiz_draft` | Function | `api/core/shared/persistence/quiz_drafts.py` | 58 |
| `load_quiz_draft_for_user` | Function | `api/core/shared/persistence/quiz_drafts.py` | 84 |
| `update_quiz_draft_for_user` | Function | `api/core/shared/persistence/quiz_drafts.py` | 98 |
| `list_recent_quiz_drafts_for_user` | Function | `api/core/shared/persistence/quiz_drafts.py` | 130 |
| `delete_quiz_draft_for_user` | Function | `api/core/shared/persistence/quiz_drafts.py` | 153 |
| `save_subject_progress_snapshot` | Function | `api/core/shared/persistence/subject_progress.py` | 128 |
| `normalize_for_bson` | Function | `api/core/shared/persistence/common.py` | 47 |
| `load_subject_progress_for_user` | Function | `api/core/shared/persistence/subject_progress.py` | 153 |
| `load_subject_progress_by_session_for_user` | Function | `api/core/shared/persistence/subject_progress.py` | 167 |
| `load_many_subject_progress_for_user` | Function | `api/core/shared/persistence/subject_progress.py` | 186 |
| `list_recent_subject_progress` | Function | `api/core/shared/persistence/subject_progress.py` | 202 |
| `list_recent_pipeline_jobs` | Function | `api/core/shared/persistence/pipeline_jobs.py` | 157 |
| `ensure_utc` | Function | `api/core/shared/persistence/common.py` | 15 |
| `utc_to_epoch` | Function | `api/core/shared/persistence/common.py` | 35 |
| `pipeline_job_to_document` | Function | `api/core/shared/persistence/pipeline_jobs.py` | 19 |
| `save_pipeline_job` | Function | `api/core/shared/persistence/pipeline_jobs.py` | 87 |
| `epoch_to_utc` | Function | `api/core/shared/persistence/common.py` | 21 |
| `optional_epoch_to_utc` | Function | `api/core/shared/persistence/common.py` | 29 |
| `load_pipeline_job` | Function | `api/core/shared/persistence/pipeline_jobs.py` | 103 |
| `load_pipeline_job_for_user` | Function | `api/core/shared/persistence/pipeline_jobs.py` | 112 |

## Execution Flows

| Flow | Type | Steps |
|------|------|-------|
| `Save_pipeline_job → Ensure_utc` | cross_community | 5 |
| `Save_pipeline_job → Utc_now` | cross_community | 5 |
| `Save_subject_progress_snapshot → Ensure_utc` | cross_community | 4 |
| `Save_subject_progress_snapshot → Utc_now` | cross_community | 4 |
| `Save_subject_progress_snapshot → _is_numpy_value` | intra_community | 4 |
| `Load_subject_progress_for_user → Ensure_utc` | cross_community | 4 |
| `Load_subject_progress_by_session_for_user → Ensure_utc` | cross_community | 4 |
| `Load_many_subject_progress_for_user → Ensure_utc` | cross_community | 4 |
| `Save_pipeline_job → _is_numpy_value` | cross_community | 4 |
| `Load_pipeline_job → _is_numpy_value` | cross_community | 4 |

## Connected Areas

| Area | Connections |
|------|-------------|
| Api | 1 calls |

## How to Explore

1. `gitnexus_context({name: "create_quiz_draft"})` — see callers and callees
2. `gitnexus_query({query: "persistence"})` — find related execution flows
3. Read key files listed above for implementation details
