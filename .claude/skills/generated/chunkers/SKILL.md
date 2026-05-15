---
name: chunkers
description: "Skill for the Chunkers area of rinkuzu-ai-api. 7 symbols across 4 files."
---

# Chunkers

7 symbols | 4 files | Cohesion: 100%

## When to Use

- Working with code in `api/`
- Understanding how test_local_pdf_text_loader_extracts_page_text, load_and_chunk_pdf, load_pdf work
- Modifying chunkers-related functionality

## Key Files

| File | Symbols |
|------|---------|
| `api/core/content_pipeline/infrastructure/processors/chunkers/text_chunker.py` | chunk, _build_text_splitter, _looks_like_markdown_or_headings |
| `api/core/content_pipeline/infrastructure/processors/loaders/local_pdf_text_loader.py` | _validate_file, load_pdf |
| `tests/core/content_pipeline/test_extract_chain_portable.py` | test_local_pdf_text_loader_extracts_page_text |
| `api/core/content_pipeline/infrastructure/processors/factory.py` | load_and_chunk_pdf |

## Entry Points

Start here when exploring this area:

- **`test_local_pdf_text_loader_extracts_page_text`** (Function) — `tests/core/content_pipeline/test_extract_chain_portable.py:241`
- **`load_and_chunk_pdf`** (Function) — `api/core/content_pipeline/infrastructure/processors/factory.py:15`
- **`load_pdf`** (Function) — `api/core/content_pipeline/infrastructure/processors/loaders/local_pdf_text_loader.py:22`
- **`chunk`** (Function) — `api/core/content_pipeline/infrastructure/processors/chunkers/text_chunker.py:71`

## Key Symbols

| Symbol | Type | File | Line |
|--------|------|------|------|
| `test_local_pdf_text_loader_extracts_page_text` | Function | `tests/core/content_pipeline/test_extract_chain_portable.py` | 241 |
| `load_and_chunk_pdf` | Function | `api/core/content_pipeline/infrastructure/processors/factory.py` | 15 |
| `load_pdf` | Function | `api/core/content_pipeline/infrastructure/processors/loaders/local_pdf_text_loader.py` | 22 |
| `chunk` | Function | `api/core/content_pipeline/infrastructure/processors/chunkers/text_chunker.py` | 71 |
| `_validate_file` | Function | `api/core/content_pipeline/infrastructure/processors/loaders/local_pdf_text_loader.py` | 11 |
| `_build_text_splitter` | Function | `api/core/content_pipeline/infrastructure/processors/chunkers/text_chunker.py` | 129 |
| `_looks_like_markdown_or_headings` | Function | `api/core/content_pipeline/infrastructure/processors/chunkers/text_chunker.py` | 163 |

## Execution Flows

| Flow | Type | Steps |
|------|------|-------|
| `Load_and_chunk_pdf → _validate_file` | intra_community | 3 |
| `Load_and_chunk_pdf → _looks_like_markdown_or_headings` | intra_community | 3 |
| `Load_and_chunk_pdf → _build_text_splitter` | intra_community | 3 |

## How to Explore

1. `gitnexus_context({name: "test_local_pdf_text_loader_extracts_page_text"})` — see callers and callees
2. `gitnexus_query({query: "chunkers"})` — find related execution flows
3. Read key files listed above for implementation details
