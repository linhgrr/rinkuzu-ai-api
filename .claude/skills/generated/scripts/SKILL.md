---
name: scripts
description: "Skill for the Scripts area of rinkuzu-ai-api. 5 symbols across 1 files."
---

# Scripts

5 symbols | 1 files | Cohesion: 100%

## When to Use

- Working with code in `scripts/`
- Understanding how reset_mongo, reset_chroma, main work
- Modifying scripts-related functionality

## Key Files

| File | Symbols |
|------|---------|
| `scripts/reset_persistence_for_beanie_cutover.py` | _get_settings, _get_chroma_store_types, reset_mongo, reset_chroma, main |

## Entry Points

Start here when exploring this area:

- **`reset_mongo`** (Function) — `scripts/reset_persistence_for_beanie_cutover.py:38`
- **`reset_chroma`** (Function) — `scripts/reset_persistence_for_beanie_cutover.py:53`
- **`main`** (Function) — `scripts/reset_persistence_for_beanie_cutover.py:61`

## Key Symbols

| Symbol | Type | File | Line |
|--------|------|------|------|
| `reset_mongo` | Function | `scripts/reset_persistence_for_beanie_cutover.py` | 38 |
| `reset_chroma` | Function | `scripts/reset_persistence_for_beanie_cutover.py` | 53 |
| `main` | Function | `scripts/reset_persistence_for_beanie_cutover.py` | 61 |
| `_get_settings` | Function | `scripts/reset_persistence_for_beanie_cutover.py` | 24 |
| `_get_chroma_store_types` | Function | `scripts/reset_persistence_for_beanie_cutover.py` | 28 |

## Execution Flows

| Flow | Type | Steps |
|------|------|-------|
| `Main → _get_settings` | intra_community | 3 |
| `Main → _get_chroma_store_types` | intra_community | 3 |

## How to Explore

1. `gitnexus_context({name: "reset_mongo"})` — see callers and callees
2. `gitnexus_query({query: "scripts"})` — find related execution flows
3. Read key files listed above for implementation details
