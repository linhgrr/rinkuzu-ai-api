---
name: merge
description: "Skill for the Merge area of rinkuzu-ai-api. 8 symbols across 2 files."
---

# Merge

8 symbols | 2 files | Cohesion: 100%

## When to Use

- Working with code in `api/`
- Understanding how merge_by_name, normalize_concept_name work
- Modifying merge-related functionality

## Key Files

| File | Symbols |
|------|---------|
| `api/core/content_pipeline/infrastructure/merge/name_merge.py` | merge_by_name, _group_and_merge, _build_global_id_map, _select_canonical, _merge_embeddings (+2) |
| `api/core/content_pipeline/infrastructure/llm/postprocess.py` | normalize_concept_name |

## Entry Points

Start here when exploring this area:

- **`merge_by_name`** (Function) — `api/core/content_pipeline/infrastructure/merge/name_merge.py:13`
- **`normalize_concept_name`** (Function) — `api/core/content_pipeline/infrastructure/llm/postprocess.py:106`

## Key Symbols

| Symbol | Type | File | Line |
|--------|------|------|------|
| `merge_by_name` | Function | `api/core/content_pipeline/infrastructure/merge/name_merge.py` | 13 |
| `normalize_concept_name` | Function | `api/core/content_pipeline/infrastructure/llm/postprocess.py` | 106 |
| `_group_and_merge` | Function | `api/core/content_pipeline/infrastructure/merge/name_merge.py` | 57 |
| `_build_global_id_map` | Function | `api/core/content_pipeline/infrastructure/merge/name_merge.py` | 91 |
| `_select_canonical` | Function | `api/core/content_pipeline/infrastructure/merge/name_merge.py` | 109 |
| `_merge_embeddings` | Function | `api/core/content_pipeline/infrastructure/merge/name_merge.py` | 124 |
| `_merge_concepts` | Function | `api/core/content_pipeline/infrastructure/merge/name_merge.py` | 152 |
| `_remap_relations` | Function | `api/core/content_pipeline/infrastructure/merge/name_merge.py` | 218 |

## Execution Flows

| Flow | Type | Steps |
|------|------|-------|
| `Merge_by_name → _select_canonical` | intra_community | 4 |
| `Merge_by_name → _merge_embeddings` | intra_community | 4 |
| `Merge_by_name → Normalize_concept_name` | intra_community | 3 |

## How to Explore

1. `gitnexus_context({name: "merge_by_name"})` — see callers and callees
2. `gitnexus_query({query: "merge"})` — find related execution flows
3. Read key files listed above for implementation details
