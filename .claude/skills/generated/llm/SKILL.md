---
name: llm
description: "Skill for the Llm area of rinkuzu-ai-api. 20 symbols across 3 files."
---

# Llm

20 symbols | 3 files | Cohesion: 100%

## When to Use

- Working with code in `api/`
- Understanding how upload_pdf_bytes, parse_response, response_usage_summary work
- Modifying llm-related functionality

## Key Files

| File | Symbols |
|------|---------|
| `api/core/content_pipeline/infrastructure/llm/schemas.py` | StrictSchemaModel, Relation, Formula, Concept, ConceptExtraction (+5) |
| `api/core/content_pipeline/infrastructure/llm/openai_responses.py` | upload_pdf_bytes, parse_response, response_usage_summary, _api_error_message, _looks_like_missing_file (+1) |
| `api/core/content_pipeline/infrastructure/llm/postprocess.py` | postprocess_concepts, _postprocess_relation, _is_valid_relation, normalize_concept_id |

## Entry Points

Start here when exploring this area:

- **`upload_pdf_bytes`** (Function) — `api/core/content_pipeline/infrastructure/llm/openai_responses.py:125`
- **`parse_response`** (Function) — `api/core/content_pipeline/infrastructure/llm/openai_responses.py:195`
- **`response_usage_summary`** (Function) — `api/core/content_pipeline/infrastructure/llm/openai_responses.py:244`
- **`postprocess_concepts`** (Function) — `api/core/content_pipeline/infrastructure/llm/postprocess.py:12`
- **`normalize_concept_id`** (Function) — `api/core/content_pipeline/infrastructure/llm/postprocess.py:131`

## Key Symbols

| Symbol | Type | File | Line |
|--------|------|------|------|
| `StrictSchemaModel` | Class | `api/core/content_pipeline/infrastructure/llm/schemas.py` | 7 |
| `Relation` | Class | `api/core/content_pipeline/infrastructure/llm/schemas.py` | 13 |
| `Formula` | Class | `api/core/content_pipeline/infrastructure/llm/schemas.py` | 31 |
| `Concept` | Class | `api/core/content_pipeline/infrastructure/llm/schemas.py` | 43 |
| `ConceptExtraction` | Class | `api/core/content_pipeline/infrastructure/llm/schemas.py` | 72 |
| `ExtractionConceptPayload` | Class | `api/core/content_pipeline/infrastructure/llm/schemas.py` | 84 |
| `ConceptExtractionPayload` | Class | `api/core/content_pipeline/infrastructure/llm/schemas.py` | 105 |
| `EvidenceVerification` | Class | `api/core/content_pipeline/infrastructure/llm/schemas.py` | 140 |
| `EdgeDecision` | Class | `api/core/content_pipeline/infrastructure/llm/schemas.py` | 165 |
| `CycleRemovalDecision` | Class | `api/core/content_pipeline/infrastructure/llm/schemas.py` | 181 |
| `upload_pdf_bytes` | Function | `api/core/content_pipeline/infrastructure/llm/openai_responses.py` | 125 |
| `parse_response` | Function | `api/core/content_pipeline/infrastructure/llm/openai_responses.py` | 195 |
| `response_usage_summary` | Function | `api/core/content_pipeline/infrastructure/llm/openai_responses.py` | 244 |
| `postprocess_concepts` | Function | `api/core/content_pipeline/infrastructure/llm/postprocess.py` | 12 |
| `normalize_concept_id` | Function | `api/core/content_pipeline/infrastructure/llm/postprocess.py` | 131 |
| `_api_error_message` | Function | `api/core/content_pipeline/infrastructure/llm/openai_responses.py` | 268 |
| `_looks_like_missing_file` | Function | `api/core/content_pipeline/infrastructure/llm/openai_responses.py` | 275 |
| `_looks_like_payload_too_large` | Function | `api/core/content_pipeline/infrastructure/llm/openai_responses.py` | 280 |
| `_postprocess_relation` | Function | `api/core/content_pipeline/infrastructure/llm/postprocess.py` | 59 |
| `_is_valid_relation` | Function | `api/core/content_pipeline/infrastructure/llm/postprocess.py` | 86 |

## Execution Flows

| Flow | Type | Steps |
|------|------|-------|
| `Postprocess_concepts → Normalize_concept_id` | intra_community | 3 |

## How to Explore

1. `gitnexus_context({name: "upload_pdf_bytes"})` — see callers and callees
2. `gitnexus_query({query: "llm"})` — find related execution flows
3. Read key files listed above for implementation details
