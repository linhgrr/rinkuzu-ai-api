---
name: content-pipeline
description: "Skill for the Content_pipeline area of rinkuzu-ai-api. 86 symbols across 35 files."
---

# Content_pipeline

86 symbols | 35 files | Cohesion: 79%

## When to Use

- Working with code in `tests/`
- Understanding how run_content_pipeline, test_serialize_concepts_returns_serializable_payloads_and_index_map, test_serialize_prerequisite_edges_keeps_only_known_prerequisite_edges work
- Modifying content_pipeline-related functionality

## Key Files

| File | Symbols |
|------|---------|
| `tests/core/content_pipeline/test_finalization_stage.py` | test_complete_pipeline_job_persists_completed_status, test_timeout_policy_defaults_are_positive, test_persist_terminal_failure_updates_job_and_saves_once, test_classify_terminal_failure_marks_timeouts_as_retryable, test_classify_terminal_failure_maps_cancelled_jobs_to_terminal_cancelled |
| `api/core/content_pipeline/infrastructure/runtime.py` | get_s3_client, _build_content_processor_bindings, get_content_processor_bindings, try_import_content_processor, calculate_file_hash |
| `tests/core/content_pipeline/test_extract_chain_portable.py` | test_extraction_response_retries_on_invalid_structured_output, _run, test_invoke_extraction_response_uses_pydantic_structured_output, test_extract_from_document_splits_again_when_provider_rejects_upload_size, test_materialize_concept_extraction_sets_optional_embeddings_to_none |
| `api/core/content_pipeline/application/pipeline_runner.py` | populate_job_metrics_from_result, _cleanup_upload, run, _resolve_effective_job_timeout |
| `api/core/content_pipeline/application/stages/result_assembly.py` | serialize_concepts, serialize_prerequisite_edges, build_graph_nodes, assemble_pipeline_result |
| `tests/core/content_pipeline/test_graph_building_stage.py` | test_build_partial_graph_serializes_nodes_and_edges, test_sanitize_concept_relations_drops_invalid_and_duplicate_relations, test_remove_invalid_graph_members_keeps_only_valid_prerequisite_edges, test_build_knowledge_graph_updates_partial_graph_and_stats |
| `api/core/content_pipeline/application/stages/graph_building.py` | build_partial_graph, sanitize_concept_relations, remove_invalid_graph_members, build_knowledge_graph |
| `tests/core/content_pipeline/test_result_assembly_stage.py` | test_serialize_concepts_returns_serializable_payloads_and_index_map, test_serialize_prerequisite_edges_keeps_only_known_prerequisite_edges, test_assemble_pipeline_result_builds_final_payload_shape |
| `api/core/content_pipeline/application/stages/finalization.py` | complete_pipeline_job, persist_terminal_failure, classify_terminal_failure |
| `tests/core/content_pipeline/test_concept_extraction_stage.py` | test_resolve_extraction_timeout_uses_retry_aware_llm_budget, test_build_partial_concept_graph_serializes_basic_node_data, test_extract_concepts_from_chunks_updates_job_metrics_and_progress |

## Entry Points

Start here when exploring this area:

- **`run_content_pipeline`** (Function) — `api/main.py:107`
- **`test_serialize_concepts_returns_serializable_payloads_and_index_map`** (Function) — `tests/core/content_pipeline/test_result_assembly_stage.py:12`
- **`test_serialize_prerequisite_edges_keeps_only_known_prerequisite_edges`** (Function) — `tests/core/content_pipeline/test_result_assembly_stage.py:62`
- **`test_assemble_pipeline_result_builds_final_payload_shape`** (Function) — `tests/core/content_pipeline/test_result_assembly_stage.py:73`
- **`test_populate_job_metrics_from_result_derives_summary_fields`** (Function) — `tests/core/content_pipeline/test_pipeline_runner.py:11`

## Key Symbols

| Symbol | Type | File | Line |
|--------|------|------|------|
| `run_content_pipeline` | Function | `api/main.py` | 107 |
| `test_serialize_concepts_returns_serializable_payloads_and_index_map` | Function | `tests/core/content_pipeline/test_result_assembly_stage.py` | 12 |
| `test_serialize_prerequisite_edges_keeps_only_known_prerequisite_edges` | Function | `tests/core/content_pipeline/test_result_assembly_stage.py` | 62 |
| `test_assemble_pipeline_result_builds_final_payload_shape` | Function | `tests/core/content_pipeline/test_result_assembly_stage.py` | 73 |
| `test_populate_job_metrics_from_result_derives_summary_fields` | Function | `tests/core/content_pipeline/test_pipeline_runner.py` | 11 |
| `test_complete_pipeline_job_persists_completed_status` | Function | `tests/core/content_pipeline/test_finalization_stage.py` | 13 |
| `test_resolve_embedding_settings_reads_unified_backend_config` | Function | `tests/core/content_pipeline/test_embedding_stage.py` | 8 |
| `test_try_restore_completed_job_from_mongo_populates_job_state` | Function | `tests/core/content_pipeline/test_cache_restore_stage.py` | 30 |
| `test_try_restore_completed_job_from_mongo_returns_false_for_miss` | Function | `tests/core/content_pipeline/test_cache_restore_stage.py` | 54 |
| `get_s3_client` | Function | `api/core/content_pipeline/infrastructure/runtime.py` | 91 |
| `get_content_processor_bindings` | Function | `api/core/content_pipeline/infrastructure/runtime.py` | 192 |
| `try_import_content_processor` | Function | `api/core/content_pipeline/infrastructure/runtime.py` | 197 |
| `populate_job_metrics_from_result` | Function | `api/core/content_pipeline/application/pipeline_runner.py` | 47 |
| `run` | Function | `api/core/content_pipeline/application/pipeline_runner.py` | 128 |
| `serialize_concepts` | Function | `api/core/content_pipeline/application/stages/result_assembly.py` | 7 |
| `serialize_prerequisite_edges` | Function | `api/core/content_pipeline/application/stages/result_assembly.py` | 37 |
| `build_graph_nodes` | Function | `api/core/content_pipeline/application/stages/result_assembly.py` | 51 |
| `assemble_pipeline_result` | Function | `api/core/content_pipeline/application/stages/result_assembly.py` | 67 |
| `complete_pipeline_job` | Function | `api/core/content_pipeline/application/stages/finalization.py` | 30 |
| `resolve_embedding_settings` | Function | `api/core/content_pipeline/application/stages/embedding.py` | 16 |

## Execution Flows

| Flow | Type | Steps |
|------|------|-------|
| `Run_content_pipeline → Get_settings` | cross_community | 7 |
| `Run_content_pipeline → _normalize_timeout` | cross_community | 7 |
| `Discover_relations → Get_settings` | cross_community | 5 |
| `Discover_relations → _normalize_timeout` | cross_community | 5 |
| `Extract_concepts_from_chunks → Get_settings` | cross_community | 5 |
| `Extract_concepts_from_chunks → _normalize_timeout` | cross_community | 5 |
| `Build_knowledge_graph → Get_settings` | cross_community | 4 |
| `Build_knowledge_graph → _normalize_timeout` | cross_community | 4 |
| `Run_content_pipeline → _build_content_processor_bindings` | intra_community | 4 |
| `Discover_relations → Get_pipeline_executor` | cross_community | 4 |

## Connected Areas

| Area | Connections |
|------|-------------|
| Stages | 12 calls |
| Api | 7 calls |

## How to Explore

1. `gitnexus_context({name: "run_content_pipeline"})` — see callers and callees
2. `gitnexus_query({query: "content_pipeline"})` — find related execution flows
3. Read key files listed above for implementation details
