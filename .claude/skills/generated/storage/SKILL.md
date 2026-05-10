---
name: storage
description: "Skill for the Storage area of rinkuzu-ai-api. 7 symbols across 3 files."
---

# Storage

7 symbols | 3 files | Cohesion: 100%

## When to Use

- Working with code in `api/`
- Understanding how reset_collection, init_chroma_store, add_chunks work
- Modifying storage-related functionality

## Key Files

| File | Symbols |
|------|---------|
| `api/core/content_pipeline/infrastructure/storage/chunk_chroma_store.py` | __init__, reset_collection, add_chunks, delete_by_job, replace_chunks |
| `api/core/content_pipeline/infrastructure/storage/chroma_store.py` | __init__ |
| `api/core/content_pipeline/infrastructure/storage/_base.py` | init_chroma_store |

## Entry Points

Start here when exploring this area:

- **`reset_collection`** (Function) — `api/core/content_pipeline/infrastructure/storage/chunk_chroma_store.py:109`
- **`init_chroma_store`** (Function) — `api/core/content_pipeline/infrastructure/storage/_base.py:15`
- **`add_chunks`** (Function) — `api/core/content_pipeline/infrastructure/storage/chunk_chroma_store.py:35`
- **`delete_by_job`** (Function) — `api/core/content_pipeline/infrastructure/storage/chunk_chroma_store.py:66`
- **`replace_chunks`** (Function) — `api/core/content_pipeline/infrastructure/storage/chunk_chroma_store.py:80`

## Key Symbols

| Symbol | Type | File | Line |
|--------|------|------|------|
| `reset_collection` | Function | `api/core/content_pipeline/infrastructure/storage/chunk_chroma_store.py` | 109 |
| `init_chroma_store` | Function | `api/core/content_pipeline/infrastructure/storage/_base.py` | 15 |
| `add_chunks` | Function | `api/core/content_pipeline/infrastructure/storage/chunk_chroma_store.py` | 35 |
| `delete_by_job` | Function | `api/core/content_pipeline/infrastructure/storage/chunk_chroma_store.py` | 66 |
| `replace_chunks` | Function | `api/core/content_pipeline/infrastructure/storage/chunk_chroma_store.py` | 80 |
| `__init__` | Function | `api/core/content_pipeline/infrastructure/storage/chunk_chroma_store.py` | 19 |
| `__init__` | Function | `api/core/content_pipeline/infrastructure/storage/chroma_store.py` | 18 |

## How to Explore

1. `gitnexus_context({name: "reset_collection"})` — see callers and callees
2. `gitnexus_query({query: "storage"})` — find related execution flows
3. Read key files listed above for implementation details
