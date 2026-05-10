---
name: schemas
description: "Skill for the Schemas area of rinkuzu-ai-api. 4 symbols across 1 files."
---

# Schemas

4 symbols | 1 files | Cohesion: 100%

## When to Use

- Working with code in `api/`
- Understanding how BaseStandardModel, StandardResponse, ErrorDetail work
- Modifying schemas-related functionality

## Key Files

| File | Symbols |
|------|---------|
| `api/schemas/common.py` | BaseStandardModel, StandardResponse, ErrorDetail, StandardErrorResponse |

## Entry Points

Start here when exploring this area:

- **`BaseStandardModel`** (Class) — `api/schemas/common.py:14`
- **`StandardResponse`** (Class) — `api/schemas/common.py:18`
- **`ErrorDetail`** (Class) — `api/schemas/common.py:24`
- **`StandardErrorResponse`** (Class) — `api/schemas/common.py:31`

## Key Symbols

| Symbol | Type | File | Line |
|--------|------|------|------|
| `BaseStandardModel` | Class | `api/schemas/common.py` | 14 |
| `StandardResponse` | Class | `api/schemas/common.py` | 18 |
| `ErrorDetail` | Class | `api/schemas/common.py` | 24 |
| `StandardErrorResponse` | Class | `api/schemas/common.py` | 31 |

## How to Explore

1. `gitnexus_context({name: "BaseStandardModel"})` — see callers and callees
2. `gitnexus_query({query: "schemas"})` — find related execution flows
3. Read key files listed above for implementation details
