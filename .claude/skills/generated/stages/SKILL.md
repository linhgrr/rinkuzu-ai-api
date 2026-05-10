---
name: stages
description: "Skill for the Stages area of rinkuzu-ai-api. 33 symbols across 13 files."
---

# Stages

33 symbols | 13 files | Cohesion: 78%

## When to Use

- Working with code in `api/`
- Understanding how test_load_document_chunks_updates_progress_and_total_chunks, test_try_restore_completed_job_from_s3_hashes_and_reads_via_blocking_stage, get_pipeline_executor work
- Modifying stages-related functionality

## Key Files

| File | Symbols |
|------|---------|
| `api/core/content_pipeline/application/stages/model_worker.py` | get_sentence_transformer_worker, encode_texts_with_sentence_transformer_worker, _start_locked, _ensure_started_locked, _close_locked (+3) |
| `api/core/content_pipeline/application/stages/enrichment.py` | _generate_theories, generate_one, generate_saint_concept_embeddings, generate_concept_theories, build_ordered_embedding_texts (+1) |
| `api/core/content_pipeline/application/stages/execution.py` | get_pipeline_executor, run_blocking_stage, safe_run, _resolve_target, _process_stage_entrypoint |
| `tests/core/content_pipeline/test_enrichment_stage.py` | test_generate_saint_concept_embeddings_updates_progress_and_returns_vectors, test_generate_concept_theories_fills_missing_theories_only, test_build_ordered_embedding_texts_uses_concept_map_order |
| `api/core/content_pipeline/application/stages/finalization.py` | _upload, upload_result_cache |
| `api/core/content_pipeline/application/stages/chunk_persistence.py` | _persist_chroma, persist_document_chunks |
| `tests/core/content_pipeline/test_document_loading_stage.py` | test_load_document_chunks_updates_progress_and_total_chunks |
| `tests/core/content_pipeline/test_cache_restore_stage.py` | test_try_restore_completed_job_from_s3_hashes_and_reads_via_blocking_stage |
| `api/core/content_pipeline/application/stages/document_loading.py` | load_document_chunks |
| `api/core/content_pipeline/application/stages/cache_restore.py` | try_restore_completed_job_from_s3 |

## Entry Points

Start here when exploring this area:

- **`test_load_document_chunks_updates_progress_and_total_chunks`** (Function) — `tests/core/content_pipeline/test_document_loading_stage.py:6`
- **`test_try_restore_completed_job_from_s3_hashes_and_reads_via_blocking_stage`** (Function) — `tests/core/content_pipeline/test_cache_restore_stage.py:69`
- **`get_pipeline_executor`** (Function) — `api/core/content_pipeline/application/stages/execution.py:32`
- **`run_blocking_stage`** (Function) — `api/core/content_pipeline/application/stages/execution.py:42`
- **`generate_one`** (Function) — `api/core/content_pipeline/application/stages/enrichment.py:89`

## Key Symbols

| Symbol | Type | File | Line |
|--------|------|------|------|
| `test_load_document_chunks_updates_progress_and_total_chunks` | Function | `tests/core/content_pipeline/test_document_loading_stage.py` | 6 |
| `test_try_restore_completed_job_from_s3_hashes_and_reads_via_blocking_stage` | Function | `tests/core/content_pipeline/test_cache_restore_stage.py` | 69 |
| `get_pipeline_executor` | Function | `api/core/content_pipeline/application/stages/execution.py` | 32 |
| `run_blocking_stage` | Function | `api/core/content_pipeline/application/stages/execution.py` | 42 |
| `generate_one` | Function | `api/core/content_pipeline/application/stages/enrichment.py` | 89 |
| `load_document_chunks` | Function | `api/core/content_pipeline/application/stages/document_loading.py` | 15 |
| `try_restore_completed_job_from_s3` | Function | `api/core/content_pipeline/application/stages/cache_restore.py` | 54 |
| `test_upload_result_cache_writes_json_payload_when_s3_is_configured` | Function | `tests/core/content_pipeline/test_finalization_stage.py` | 43 |
| `test_generate_saint_concept_embeddings_updates_progress_and_returns_vectors` | Function | `tests/core/content_pipeline/test_enrichment_stage.py` | 24 |
| `test_generate_concept_theories_fills_missing_theories_only` | Function | `tests/core/content_pipeline/test_enrichment_stage.py` | 86 |
| `upload_result_cache` | Function | `api/core/content_pipeline/application/stages/finalization.py` | 44 |
| `safe_run` | Function | `api/core/content_pipeline/application/stages/execution.py` | 193 |
| `generate_saint_concept_embeddings` | Function | `api/core/content_pipeline/application/stages/enrichment.py` | 34 |
| `generate_concept_theories` | Function | `api/core/content_pipeline/application/stages/enrichment.py` | 71 |
| `persist_document_chunks` | Function | `api/core/content_pipeline/application/stages/chunk_persistence.py` | 29 |
| `test_build_ordered_embedding_texts_uses_concept_map_order` | Function | `tests/core/content_pipeline/test_enrichment_stage.py` | 13 |
| `test_compute_concept_embeddings_updates_progress_and_uses_worker` | Function | `tests/core/content_pipeline/test_embedding_stage.py` | 21 |
| `get_sentence_transformer_worker` | Function | `api/core/content_pipeline/application/stages/model_worker.py` | 329 |
| `encode_texts_with_sentence_transformer_worker` | Function | `api/core/content_pipeline/application/stages/model_worker.py` | 336 |
| `build_ordered_embedding_texts` | Function | `api/core/content_pipeline/application/stages/enrichment.py` | 19 |

## Execution Flows

| Flow | Type | Steps |
|------|------|-------|
| `Discover_relations → Get_settings` | cross_community | 5 |
| `Discover_relations → _normalize_timeout` | cross_community | 5 |
| `Compute_concept_embeddings → _start_locked` | cross_community | 5 |
| `Compute_concept_embeddings → _close_locked` | cross_community | 5 |
| `_generate → _start_locked` | cross_community | 5 |
| `_generate → _close_locked` | cross_community | 5 |
| `Build_knowledge_graph → Get_settings` | cross_community | 4 |
| `Build_knowledge_graph → _normalize_timeout` | cross_community | 4 |
| `Discover_relations → Get_pipeline_executor` | cross_community | 4 |
| `Optimize_graph → Get_settings` | cross_community | 4 |

## Connected Areas

| Area | Connections |
|------|-------------|
| Content_pipeline | 3 calls |
| Api | 1 calls |

## How to Explore

1. `gitnexus_context({name: "test_load_document_chunks_updates_progress_and_total_chunks"})` — see callers and callees
2. `gitnexus_query({query: "stages"})` — find related execution flows
3. Read key files listed above for implementation details
