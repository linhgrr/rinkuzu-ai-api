---
name: persistence
description: "Skill for the Persistence area of rinkuzu-ai-api. 33 symbols across 7 files."
---

# Persistence

33 symbols | 7 files | Cohesion: 83%

## When to Use

- Working with code in `api/`
- Understanding how save_subject_progress_snapshot, pipeline_job_to_document, save_pipeline_job work
- Modifying persistence-related functionality

## Key Files

| File | Symbols |
|------|---------|
| `api/core/shared/persistence/subject_progress.py` | _resolve_concept_indices, _snapshot_to_document_payload, save_subject_progress_snapshot, _document_to_legacy_payload, load_subject_progress_for_user (+4) |
| `api/core/shared/persistence/pipeline_jobs.py` | pipeline_job_to_document, save_pipeline_job, list_recent_pipeline_jobs, _document_to_runtime_payload, load_pipeline_job (+3) |
| `api/core/shared/persistence/quiz_drafts.py` | _normalize_questions, _document_to_public_dict, create_quiz_draft, load_quiz_draft_for_user, update_quiz_draft_for_user (+2) |
| `api/core/shared/persistence/common.py` | epoch_to_utc, optional_epoch_to_utc, ensure_utc, utc_to_epoch, utc_now |
| `api/core/shared/persistence/documents.py` | touch_updated_at, touch_updated_at |
| `api/core/shared/persistence/openai_file_cache.py` | save_cached_openai_file |
| `api/core/shared/persistence/document_chunks.py` | delete_chunks_for_job |

## Entry Points

Start here when exploring this area:

- **`save_subject_progress_snapshot`** (Function) — `api/core/shared/persistence/subject_progress.py:129`
- **`pipeline_job_to_document`** (Function) — `api/core/shared/persistence/pipeline_jobs.py:20`
- **`save_pipeline_job`** (Function) — `api/core/shared/persistence/pipeline_jobs.py:88`
- **`epoch_to_utc`** (Function) — `api/core/shared/persistence/common.py:21`
- **`optional_epoch_to_utc`** (Function) — `api/core/shared/persistence/common.py:29`

## Key Symbols

| Symbol | Type | File | Line |
|--------|------|------|------|
| `save_subject_progress_snapshot` | Function | `api/core/shared/persistence/subject_progress.py` | 129 |
| `pipeline_job_to_document` | Function | `api/core/shared/persistence/pipeline_jobs.py` | 20 |
| `save_pipeline_job` | Function | `api/core/shared/persistence/pipeline_jobs.py` | 88 |
| `epoch_to_utc` | Function | `api/core/shared/persistence/common.py` | 21 |
| `optional_epoch_to_utc` | Function | `api/core/shared/persistence/common.py` | 29 |
| `create_quiz_draft` | Function | `api/core/shared/persistence/quiz_drafts.py` | 59 |
| `load_quiz_draft_for_user` | Function | `api/core/shared/persistence/quiz_drafts.py` | 85 |
| `update_quiz_draft_for_user` | Function | `api/core/shared/persistence/quiz_drafts.py` | 99 |
| `list_recent_quiz_drafts_for_user` | Function | `api/core/shared/persistence/quiz_drafts.py` | 131 |
| `delete_quiz_draft_for_user` | Function | `api/core/shared/persistence/quiz_drafts.py` | 157 |
| `load_subject_progress_for_user` | Function | `api/core/shared/persistence/subject_progress.py` | 154 |
| `load_subject_progress_by_session_for_user` | Function | `api/core/shared/persistence/subject_progress.py` | 168 |
| `load_many_subject_progress_for_user` | Function | `api/core/shared/persistence/subject_progress.py` | 187 |
| `list_recent_subject_progress` | Function | `api/core/shared/persistence/subject_progress.py` | 203 |
| `list_recent_pipeline_jobs` | Function | `api/core/shared/persistence/pipeline_jobs.py` | 158 |
| `ensure_utc` | Function | `api/core/shared/persistence/common.py` | 15 |
| `utc_to_epoch` | Function | `api/core/shared/persistence/common.py` | 35 |
| `load_pipeline_job` | Function | `api/core/shared/persistence/pipeline_jobs.py` | 104 |
| `load_pipeline_job_for_user` | Function | `api/core/shared/persistence/pipeline_jobs.py` | 113 |
| `load_many_pipeline_jobs_for_user` | Function | `api/core/shared/persistence/pipeline_jobs.py` | 127 |

## Execution Flows

| Flow | Type | Steps |
|------|------|-------|
| `Save_pipeline_job → Ensure_utc` | cross_community | 5 |
| `Save_pipeline_job → Utc_now` | cross_community | 5 |
| `Save_subject_progress_snapshot → Ensure_utc` | cross_community | 4 |
| `Save_subject_progress_snapshot → Utc_now` | cross_community | 4 |
| `Save_subject_progress_snapshot → _is_numpy_value` | cross_community | 4 |
| `Load_subject_progress_for_user → Ensure_utc` | cross_community | 4 |
| `Load_subject_progress_by_session_for_user → Ensure_utc` | cross_community | 4 |
| `Load_many_subject_progress_for_user → Ensure_utc` | cross_community | 4 |
| `Save_pipeline_job → _is_numpy_value` | cross_community | 4 |
| `Load_pipeline_job → _is_numpy_value` | cross_community | 4 |

## Connected Areas

| Area | Connections |
|------|-------------|
| Stages | 3 calls |
| Api | 1 calls |

## How to Explore

1. `gitnexus_context({name: "save_subject_progress_snapshot"})` — see callers and callees
2. `gitnexus_query({query: "persistence"})` — find related execution flows
3. Read key files listed above for implementation details
