---
name: tests
description: "Skill for the Tests area of rinkuzu-ai-api. 14 symbols across 4 files."
---

# Tests

14 symbols | 4 files | Cohesion: 93%

## When to Use

- Working with code in `tests/`
- Understanding how test_get_session_resource_supports_sync_knowledge_graph_fetcher, test_get_session_resource_supports_sync_mastery_fetcher, test_get_session_resource_supports_sync_concept_detail_fetcher work
- Modifying tests-related functionality

## Key Files

| File | Symbols |
|------|---------|
| `tests/test_exceptions.py` | _build_app, test_http_exception_is_normalized, test_app_error_is_normalized, test_validation_error_is_normalized, test_unexpected_error_is_sanitized |
| `tests/test_knowledge_router.py` | test_get_session_resource_supports_sync_knowledge_graph_fetcher, test_get_session_resource_supports_sync_mastery_fetcher, test_get_session_resource_supports_sync_concept_detail_fetcher, test_get_session_resource_raises_session_not_found_when_fetcher_returns_none |
| `api/routers/knowledge.py` | _get_session_resource, get_knowledge_graph, get_mastery_matrix, get_concept_detail |
| `api/exceptions.py` | register_exception_handlers |

## Entry Points

Start here when exploring this area:

- **`test_get_session_resource_supports_sync_knowledge_graph_fetcher`** (Function) — `tests/test_knowledge_router.py:9`
- **`test_get_session_resource_supports_sync_mastery_fetcher`** (Function) — `tests/test_knowledge_router.py:58`
- **`test_get_session_resource_supports_sync_concept_detail_fetcher`** (Function) — `tests/test_knowledge_router.py:86`
- **`test_get_session_resource_raises_session_not_found_when_fetcher_returns_none`** (Function) — `tests/test_knowledge_router.py:116`
- **`get_knowledge_graph`** (Function) — `api/routers/knowledge.py:38`

## Key Symbols

| Symbol | Type | File | Line |
|--------|------|------|------|
| `test_get_session_resource_supports_sync_knowledge_graph_fetcher` | Function | `tests/test_knowledge_router.py` | 9 |
| `test_get_session_resource_supports_sync_mastery_fetcher` | Function | `tests/test_knowledge_router.py` | 58 |
| `test_get_session_resource_supports_sync_concept_detail_fetcher` | Function | `tests/test_knowledge_router.py` | 86 |
| `test_get_session_resource_raises_session_not_found_when_fetcher_returns_none` | Function | `tests/test_knowledge_router.py` | 116 |
| `get_knowledge_graph` | Function | `api/routers/knowledge.py` | 38 |
| `get_mastery_matrix` | Function | `api/routers/knowledge.py` | 57 |
| `get_concept_detail` | Function | `api/routers/knowledge.py` | 78 |
| `test_http_exception_is_normalized` | Function | `tests/test_exceptions.py` | 29 |
| `test_app_error_is_normalized` | Function | `tests/test_exceptions.py` | 42 |
| `test_validation_error_is_normalized` | Function | `tests/test_exceptions.py` | 55 |
| `test_unexpected_error_is_sanitized` | Function | `tests/test_exceptions.py` | 69 |
| `register_exception_handlers` | Function | `api/exceptions.py` | 131 |
| `_get_session_resource` | Function | `api/routers/knowledge.py` | 21 |
| `_build_app` | Function | `tests/test_exceptions.py` | 6 |

## Execution Flows

| Flow | Type | Steps |
|------|------|-------|
| `Get_knowledge_graph → Resolve_user_session` | cross_community | 3 |
| `Get_knowledge_graph → Ok` | cross_community | 3 |
| `Get_mastery_matrix → Resolve_user_session` | cross_community | 3 |
| `Get_mastery_matrix → Ok` | cross_community | 3 |
| `Get_concept_detail → Resolve_user_session` | cross_community | 3 |
| `Get_concept_detail → Ok` | cross_community | 3 |

## Connected Areas

| Area | Connections |
|------|-------------|
| Routers | 2 calls |

## How to Explore

1. `gitnexus_context({name: "test_get_session_resource_supports_sync_knowledge_graph_fetcher"})` — see callers and callees
2. `gitnexus_query({query: "tests"})` — find related execution flows
3. Read key files listed above for implementation details
