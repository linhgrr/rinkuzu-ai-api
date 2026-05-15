---
name: stages
description: "Skill for the Stages area of rinkuzu-ai-api. 28 symbols across 14 files."
---

# Stages

28 symbols | 14 files | Cohesion: 73%

## When to Use

- Working with code in `api/`
- Understanding how test_build_ordered_embedding_texts_uses_concept_map_order, test_compute_concept_embeddings_updates_progress_and_uses_worker, get_sentence_transformer_worker work
- Modifying stages-related functionality

## Key Files

| File | Symbols |
|------|---------|
| `api/core/content_pipeline/application/stages/model_worker.py` | get_sentence_transformer_worker, encode_texts_with_sentence_transformer_worker, _start_locked, _ensure_started_locked, _close_locked (+3) |
| `api/core/content_pipeline/application/stages/enrichment.py` | build_ordered_embedding_texts, _generate, _generate_theories, generate_one |
| `api/core/content_pipeline/application/stages/execution.py` | get_pipeline_executor, run_blocking_stage, _resolve_target, _process_stage_entrypoint |
| `api/core/shared/persistence/common.py` | _is_numpy_value, normalize_for_bson |
| `tests/core/content_pipeline/test_enrichment_stage.py` | test_build_ordered_embedding_texts_uses_concept_map_order |
| `tests/core/content_pipeline/test_embedding_stage.py` | test_compute_concept_embeddings_updates_progress_and_uses_worker |
| `api/core/content_pipeline/application/stages/embedding.py` | compute_concept_embeddings |
| `tests/core/content_pipeline/test_document_loading_stage.py` | test_load_document_chunks_updates_progress_and_total_chunks |
| `tests/core/content_pipeline/test_cache_restore_stage.py` | test_try_restore_completed_job_from_s3_hashes_and_reads_via_blocking_stage |
| `api/core/content_pipeline/application/stages/document_loading.py` | load_document_chunks |

## Entry Points

Start here when exploring this area:

- **`test_build_ordered_embedding_texts_uses_concept_map_order`** (Function) — `tests/core/content_pipeline/test_enrichment_stage.py:13`
- **`test_compute_concept_embeddings_updates_progress_and_uses_worker`** (Function) — `tests/core/content_pipeline/test_embedding_stage.py:25`
- **`get_sentence_transformer_worker`** (Function) — `api/core/content_pipeline/application/stages/model_worker.py:328`
- **`encode_texts_with_sentence_transformer_worker`** (Function) — `api/core/content_pipeline/application/stages/model_worker.py:335`
- **`build_ordered_embedding_texts`** (Function) — `api/core/content_pipeline/application/stages/enrichment.py:20`

## Key Symbols

| Symbol | Type | File | Line |
|--------|------|------|------|
| `test_build_ordered_embedding_texts_uses_concept_map_order` | Function | `tests/core/content_pipeline/test_enrichment_stage.py` | 13 |
| `test_compute_concept_embeddings_updates_progress_and_uses_worker` | Function | `tests/core/content_pipeline/test_embedding_stage.py` | 25 |
| `get_sentence_transformer_worker` | Function | `api/core/content_pipeline/application/stages/model_worker.py` | 328 |
| `encode_texts_with_sentence_transformer_worker` | Function | `api/core/content_pipeline/application/stages/model_worker.py` | 335 |
| `build_ordered_embedding_texts` | Function | `api/core/content_pipeline/application/stages/enrichment.py` | 20 |
| `compute_concept_embeddings` | Function | `api/core/content_pipeline/application/stages/embedding.py` | 22 |
| `test_load_document_chunks_updates_progress_and_total_chunks` | Function | `tests/core/content_pipeline/test_document_loading_stage.py` | 6 |
| `test_try_restore_completed_job_from_s3_hashes_and_reads_via_blocking_stage` | Function | `tests/core/content_pipeline/test_cache_restore_stage.py` | 69 |
| `get_pipeline_executor` | Function | `api/core/content_pipeline/application/stages/execution.py` | 34 |
| `run_blocking_stage` | Function | `api/core/content_pipeline/application/stages/execution.py` | 44 |
| `load_document_chunks` | Function | `api/core/content_pipeline/application/stages/document_loading.py` | 15 |
| `try_restore_completed_job_from_s3` | Function | `api/core/content_pipeline/application/stages/cache_restore.py` | 54 |
| `normalize_for_bson` | Function | `api/core/shared/persistence/common.py` | 47 |
| `generate_one` | Function | `api/core/content_pipeline/application/stages/enrichment.py` | 92 |
| `encode` | Function | `api/core/content_pipeline/application/stages/model_worker.py` | 166 |
| `_generate` | Function | `api/core/content_pipeline/application/stages/enrichment.py` | 50 |
| `_persist_chroma` | Function | `api/core/content_pipeline/application/stages/chunk_persistence.py` | 66 |
| `_is_numpy_value` | Function | `api/core/shared/persistence/common.py` | 43 |
| `_to_bson_safe` | Function | `api/core/content_pipeline/infrastructure/serializers.py` | 12 |
| `_upload` | Function | `api/core/content_pipeline/application/stages/finalization.py` | 59 |

## Execution Flows

| Flow | Type | Steps |
|------|------|-------|
| `Discover_relations → Get_settings` | cross_community | 5 |
| `Discover_relations → _normalize_timeout` | cross_community | 5 |
| `Compute_concept_embeddings → _start_locked` | cross_community | 5 |
| `Compute_concept_embeddings → _close_locked` | cross_community | 5 |
| `_generate → _start_locked` | cross_community | 5 |
| `_generate → _close_locked` | cross_community | 5 |
| `_generate_theories → Get_settings` | cross_community | 5 |
| `_generate_theories → _normalize_timeout` | cross_community | 5 |
| `Build_knowledge_graph → Get_settings` | cross_community | 4 |
| `Build_knowledge_graph → _normalize_timeout` | cross_community | 4 |

## Connected Areas

| Area | Connections |
|------|-------------|
| Content_pipeline | 3 calls |
| Api | 1 calls |

## How to Explore

1. `gitnexus_context({name: "test_build_ordered_embedding_texts_uses_concept_map_order"})` — see callers and callees
2. `gitnexus_query({query: "stages"})` — find related execution flows
3. Read key files listed above for implementation details
