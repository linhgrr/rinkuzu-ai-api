---
name: application
description: "Skill for the Application area of rinkuzu-ai-api. 12 symbols across 5 files."
---

# Application

12 symbols | 5 files | Cohesion: 100%

## When to Use

- Working with code in `api/`
- Understanding how persist_pipeline_job_state, test_start_job_persists_pending_then_queued_and_schedules_background_task, test_start_job_returns_failed_job_when_content_processor_is_unavailable work
- Modifying application-related functionality

## Key Files

| File | Symbols |
|------|---------|
| `api/core/content_pipeline/application/pipeline_service.py` | persist_job_state, start_job, _build_job, _schedule_background_run, _gated_run (+1) |
| `tests/core/content_pipeline/test_pipeline_service.py` | test_start_job_persists_pending_then_queued_and_schedules_background_task, test_start_job_returns_failed_job_when_content_processor_is_unavailable, test_shutdown_cancels_inflight_background_tasks |
| `api/main.py` | persist_pipeline_job_state |
| `tests/core/content_pipeline/test_jobs.py` | test_pipeline_job_mark_failed_sets_status_and_message |
| `api/core/content_pipeline/domain/jobs.py` | mark_failed |

## Entry Points

Start here when exploring this area:

- **`persist_pipeline_job_state`** (Function) — `api/main.py:95`
- **`test_start_job_persists_pending_then_queued_and_schedules_background_task`** (Function) — `tests/core/content_pipeline/test_pipeline_service.py:9`
- **`test_start_job_returns_failed_job_when_content_processor_is_unavailable`** (Function) — `tests/core/content_pipeline/test_pipeline_service.py:51`
- **`test_shutdown_cancels_inflight_background_tasks`** (Function) — `tests/core/content_pipeline/test_pipeline_service.py:87`
- **`test_pipeline_job_mark_failed_sets_status_and_message`** (Function) — `tests/core/content_pipeline/test_jobs.py:29`

## Key Symbols

| Symbol | Type | File | Line |
|--------|------|------|------|
| `persist_pipeline_job_state` | Function | `api/main.py` | 95 |
| `test_start_job_persists_pending_then_queued_and_schedules_background_task` | Function | `tests/core/content_pipeline/test_pipeline_service.py` | 9 |
| `test_start_job_returns_failed_job_when_content_processor_is_unavailable` | Function | `tests/core/content_pipeline/test_pipeline_service.py` | 51 |
| `test_shutdown_cancels_inflight_background_tasks` | Function | `tests/core/content_pipeline/test_pipeline_service.py` | 87 |
| `test_pipeline_job_mark_failed_sets_status_and_message` | Function | `tests/core/content_pipeline/test_jobs.py` | 29 |
| `mark_failed` | Function | `api/core/content_pipeline/domain/jobs.py` | 104 |
| `persist_job_state` | Function | `api/core/content_pipeline/application/pipeline_service.py` | 51 |
| `start_job` | Function | `api/core/content_pipeline/application/pipeline_service.py` | 72 |
| `shutdown` | Function | `api/core/content_pipeline/application/pipeline_service.py` | 170 |
| `_build_job` | Function | `api/core/content_pipeline/application/pipeline_service.py` | 126 |
| `_schedule_background_run` | Function | `api/core/content_pipeline/application/pipeline_service.py` | 143 |
| `_gated_run` | Function | `api/core/content_pipeline/application/pipeline_service.py` | 155 |

## Execution Flows

| Flow | Type | Steps |
|------|------|-------|
| `Start_job → _gated_run` | intra_community | 3 |

## How to Explore

1. `gitnexus_context({name: "persist_pipeline_job_state"})` — see callers and callees
2. `gitnexus_query({query: "application"})` — find related execution flows
3. Read key files listed above for implementation details
