---
name: embed
description: "Skill for the Embed area of rinkuzu-ai-api. 10 symbols across 4 files."
---

# Embed

10 symbols | 4 files | Cohesion: 100%

## When to Use

- Working with code in `api/`
- Understanding how compute_embedding_for_concepts, embed_query, embed_documents work
- Modifying embed-related functionality

## Key Files

| File | Symbols |
|------|---------|
| `api/core/content_pipeline/infrastructure/embed/embedding_client.py` | _maybe_tokenize, embed_query, embed_documents, encode, _load_model_handle (+1) |
| `api/core/content_pipeline/infrastructure/embed/prereq_ranking.py` | rank_prerequisites, _compute_prerequisite_scores |
| `api/core/content_pipeline/infrastructure/embed/embeddings.py` | compute_embedding_for_concepts |
| `api/core/content_pipeline/infrastructure/embed/__init__.py` | compute_embeddings_batch |

## Entry Points

Start here when exploring this area:

- **`compute_embedding_for_concepts`** (Function) — `api/core/content_pipeline/infrastructure/embed/embeddings.py:9`
- **`embed_query`** (Function) — `api/core/content_pipeline/infrastructure/embed/embedding_client.py:91`
- **`embed_documents`** (Function) — `api/core/content_pipeline/infrastructure/embed/embedding_client.py:119`
- **`encode`** (Function) — `api/core/content_pipeline/infrastructure/embed/embedding_client.py:147`
- **`compute_embeddings_batch`** (Function) — `api/core/content_pipeline/infrastructure/embed/__init__.py:17`

## Key Symbols

| Symbol | Type | File | Line |
|--------|------|------|------|
| `compute_embedding_for_concepts` | Function | `api/core/content_pipeline/infrastructure/embed/embeddings.py` | 9 |
| `embed_query` | Function | `api/core/content_pipeline/infrastructure/embed/embedding_client.py` | 91 |
| `embed_documents` | Function | `api/core/content_pipeline/infrastructure/embed/embedding_client.py` | 119 |
| `encode` | Function | `api/core/content_pipeline/infrastructure/embed/embedding_client.py` | 147 |
| `compute_embeddings_batch` | Function | `api/core/content_pipeline/infrastructure/embed/__init__.py` | 17 |
| `rank_prerequisites` | Function | `api/core/content_pipeline/infrastructure/embed/prereq_ranking.py` | 16 |
| `_maybe_tokenize` | Function | `api/core/content_pipeline/infrastructure/embed/embedding_client.py` | 85 |
| `_compute_prerequisite_scores` | Function | `api/core/content_pipeline/infrastructure/embed/prereq_ranking.py` | 68 |
| `_load_model_handle` | Function | `api/core/content_pipeline/infrastructure/embed/embedding_client.py` | 31 |
| `__init__` | Function | `api/core/content_pipeline/infrastructure/embed/embedding_client.py` | 44 |

## Execution Flows

| Flow | Type | Steps |
|------|------|-------|
| `Compute_embedding_for_concepts → _maybe_tokenize` | intra_community | 4 |
| `Compute_embeddings_batch → _maybe_tokenize` | intra_community | 4 |
| `Embed_query → _maybe_tokenize` | intra_community | 3 |

## How to Explore

1. `gitnexus_context({name: "compute_embedding_for_concepts"})` — see callers and callees
2. `gitnexus_query({query: "embed"})` — find related execution flows
3. Read key files listed above for implementation details
