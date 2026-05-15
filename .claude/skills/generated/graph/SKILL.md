---
name: graph
description: "Skill for the Graph area of rinkuzu-ai-api. 20 symbols across 4 files."
---

# Graph

20 symbols | 4 files | Cohesion: 100%

## When to Use

- Working with code in `api/`
- Understanding how add_concepts, add_concept, add_relation work
- Modifying graph-related functionality

## Key Files

| File | Symbols |
|------|---------|
| `api/core/content_pipeline/infrastructure/graph/builder.py` | _norm_rel, _title_from_id, _ensure_node, _set_non_placeholder, _normalize_evidence (+6) |
| `api/core/content_pipeline/infrastructure/graph/cycle_removal.py` | remove_cycles, _remove_cycle_with_llm, _format_cycle_info, make_dag_with_llm |
| `api/core/content_pipeline/infrastructure/graph/reduction.py` | apply_transitive_reduction, _extract_prerequisite_subgraph, _find_removed_edges, _rebuild_graph |
| `tests/core/content_pipeline/test_cycle_removal.py` | test_cycle_remover_uses_langchain_structured_output |

## Entry Points

Start here when exploring this area:

- **`add_concepts`** (Function) — `api/core/content_pipeline/infrastructure/graph/builder.py:92`
- **`add_concept`** (Function) — `api/core/content_pipeline/infrastructure/graph/builder.py:97`
- **`add_relation`** (Function) — `api/core/content_pipeline/infrastructure/graph/builder.py:117`
- **`get_prerequisites`** (Function) — `api/core/content_pipeline/infrastructure/graph/builder.py:185`
- **`get_dependents`** (Function) — `api/core/content_pipeline/infrastructure/graph/builder.py:196`

## Key Symbols

| Symbol | Type | File | Line |
|--------|------|------|------|
| `add_concepts` | Function | `api/core/content_pipeline/infrastructure/graph/builder.py` | 92 |
| `add_concept` | Function | `api/core/content_pipeline/infrastructure/graph/builder.py` | 97 |
| `add_relation` | Function | `api/core/content_pipeline/infrastructure/graph/builder.py` | 117 |
| `get_prerequisites` | Function | `api/core/content_pipeline/infrastructure/graph/builder.py` | 185 |
| `get_dependents` | Function | `api/core/content_pipeline/infrastructure/graph/builder.py` | 196 |
| `test_cycle_remover_uses_langchain_structured_output` | Function | `tests/core/content_pipeline/test_cycle_removal.py` | 7 |
| `remove_cycles` | Function | `api/core/content_pipeline/infrastructure/graph/cycle_removal.py` | 42 |
| `make_dag_with_llm` | Function | `api/core/content_pipeline/infrastructure/graph/cycle_removal.py` | 248 |
| `apply_transitive_reduction` | Function | `api/core/content_pipeline/infrastructure/graph/reduction.py` | 6 |
| `_norm_rel` | Function | `api/core/content_pipeline/infrastructure/graph/builder.py` | 33 |
| `_title_from_id` | Function | `api/core/content_pipeline/infrastructure/graph/builder.py` | 40 |
| `_ensure_node` | Function | `api/core/content_pipeline/infrastructure/graph/builder.py` | 43 |
| `_set_non_placeholder` | Function | `api/core/content_pipeline/infrastructure/graph/builder.py` | 53 |
| `_normalize_evidence` | Function | `api/core/content_pipeline/infrastructure/graph/builder.py` | 59 |
| `_merge_evidence` | Function | `api/core/content_pipeline/infrastructure/graph/builder.py` | 76 |
| `_remove_cycle_with_llm` | Function | `api/core/content_pipeline/infrastructure/graph/cycle_removal.py` | 128 |
| `_format_cycle_info` | Function | `api/core/content_pipeline/infrastructure/graph/cycle_removal.py` | 190 |
| `_extract_prerequisite_subgraph` | Function | `api/core/content_pipeline/infrastructure/graph/reduction.py` | 40 |
| `_find_removed_edges` | Function | `api/core/content_pipeline/infrastructure/graph/reduction.py` | 53 |
| `_rebuild_graph` | Function | `api/core/content_pipeline/infrastructure/graph/reduction.py` | 64 |

## Execution Flows

| Flow | Type | Steps |
|------|------|-------|
| `Add_concepts → _title_from_id` | intra_community | 5 |
| `Make_dag_with_llm → _format_cycle_info` | intra_community | 4 |
| `Add_concepts → _norm_rel` | intra_community | 4 |

## How to Explore

1. `gitnexus_context({name: "add_concepts"})` — see callers and callees
2. `gitnexus_query({query: "graph"})` — find related execution flows
3. Read key files listed above for implementation details
