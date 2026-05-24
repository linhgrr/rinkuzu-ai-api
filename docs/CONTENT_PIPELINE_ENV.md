# Content Pipeline Environment

The backend reads runtime configuration from `api.config.Settings`.

For the LLM provider, the project now standardizes on the shared LiteLLM-backed chat/completions abstraction in `api.core.shared.llm`. Runtime config is strict: only `LLM_*` env vars are read, but the model string itself is passed through as-is.

## Core LLM

| Logical setting | Supported env vars | Default |
| --- | --- | --- |
| `llm_base_url` | `LLM_BASE_URL` | Required |
| `llm_model` | `LLM_MODEL` | Required |
| `llm_api_key` | `LLM_API_KEY` | Required |
| `exercise_llm_model` | `EXERCISE_LLM_MODEL`, `ADAPTIVE_EXERCISE_LLM_MODEL` | `None` |
| `llm_embedding_model` | `LLM_EMBEDDING_MODEL` | `text-embedding-3-small` |
| `llm_timeout_sec` | `LLM_TIMEOUT_SEC` | `150` |
| `llm_max_retries` | `LLM_MAX_RETRIES` | `2` |
| `llm_max_workers` | `LLM_MAX_WORKERS`, `ADAPTIVE_LLM_MAX_WORKERS` | `8` |
| `llm_max_concurrency` | `LLM_MAX_CONCURRENCY`, `ADAPTIVE_LLM_MAX_CONCURRENCY` | `None` |
| `llm_request_timeout_sec` | `LLM_REQUEST_TIMEOUT_SEC`, `ADAPTIVE_LLM_TIMEOUT_SEC` | `120` |
| `llm_prefetch_timeout_sec` | `LLM_PREFETCH_TIMEOUT_SEC`, `ADAPTIVE_PREFETCH_LLM_TIMEOUT_SEC` | `None` |
| `llm_retry_attempts` | `LLM_RETRY_ATTEMPTS`, `ADAPTIVE_LLM_RETRY_ATTEMPTS` | `3` |
| `llm_retry_backoff_sec` | `LLM_RETRY_BACKOFF_SEC`, `ADAPTIVE_LLM_RETRY_BACKOFF_SEC` | `1.0` |

Notes:
- `exercise_llm_model` lets tutor/exercise flows override the shared `llm_model`.
- `exercise_llm_model` is an optional override for tutor/exercise flows and is also passed through as-is.
- The runtime no longer depends on provider-specific Files/Responses APIs.
- Structured output is generated through chat/completions with JSON-object response formatting plus schema guidance in prompts.

## Embedding And Chunking

| Setting field | Env var | Default |
| --- | --- | --- |
| `embedding_model` | `EMBEDDING_MODEL` | `keepitreal/vietnamese-sbert` |
| `embedding_batch_size` | `EMBEDDING_BATCH_SIZE` | `32` |
| `use_vi_tokenizer` | `USE_VI_TOKENIZER` | `false` |
| `max_seq_length` | `MAX_SEQ_LENGTH` | `None` |
| `chunk_size` | `CHUNK_SIZE` | `1000` |
| `chunk_overlap` | `CHUNK_OVERLAP` | `200` |
| `prs_threshold` | `PRS_THRESHOLD` | `0.75` |
| `similarity_threshold` | `SIMILARITY_THRESHOLD` | `0.9` |

## Pipeline Timeouts

| Setting field | Env var | Default |
| --- | --- | --- |
| `content_pipeline_job_timeout_sec` | `CONTENT_PIPELINE_JOB_TIMEOUT_SEC` | `1800` |
| `content_pipeline_stage_timeout_sec` | `CONTENT_PIPELINE_STAGE_TIMEOUT_SEC` | `300` |
| `content_pipeline_graph_cycle_timeout_sec` | `CONTENT_PIPELINE_GRAPH_CYCLE_TIMEOUT_SEC` | `900` |
| `content_pipeline_llm_request_timeout_sec` | `CONTENT_PIPELINE_LLM_REQUEST_TIMEOUT_SEC`, `CONTENT_PIPELINE_RESPONSES_TIMEOUT_SEC` | `180` |
| `content_pipeline_llm_retry_attempts` | `CONTENT_PIPELINE_LLM_RETRY_ATTEMPTS`, `LLM_RETRY_ATTEMPTS`, `ADAPTIVE_LLM_RETRY_ATTEMPTS` | `3` |
| `content_pipeline_llm_retry_backoff_sec` | `CONTENT_PIPELINE_LLM_RETRY_BACKOFF_SEC`, `LLM_RETRY_BACKOFF_SEC`, `ADAPTIVE_LLM_RETRY_BACKOFF_SEC` | `1.0` |

## PDF Text Extraction

| Setting field | Env var | Default |
| --- | --- | --- |
| `content_pipeline_pdf_page_batch_size` | `CONTENT_PIPELINE_PDF_PAGE_BATCH_SIZE` | `10` |
| `content_pipeline_pdf_batch_max_bytes` | `CONTENT_PIPELINE_PDF_BATCH_MAX_BYTES` | `4194304` |
| `content_pipeline_batch_failure_ratio_threshold` | `CONTENT_PIPELINE_BATCH_FAILURE_RATIO_THRESHOLD` | `0.5` |
| `ocr_base_url` | `OCR_BASE_URL` | `https://api.va.landing.ai/v1/ade/parse` |
| `ocr_model` | `OCR_MODEL` | `dpt-2-mini` |
| `ocr_api_key` | `OCR_API_KEY` | Required |
| `ocr_timeout_sec` | `OCR_TIMEOUT_SEC` | `120` |

Notes:
- PDF text extraction now uses a single OCR API provider only; there is no local-text fallback path.
- The current implementation is wired to LandingAI ADE Parse through the generic `OCR_*` settings above.
- The extractor sends the PDF directly to the parse API with `split=page`, then converts each split into page-level text for downstream batching.
- `OCR_API_KEY` is the only required OCR secret for local setup.

## Object Storage

| Setting field | Env var | Default |
| --- | --- | --- |
| `object_storage_endpoint_external` | `OBJECT_STORAGE_ENDPOINT_EXTERNAL` | `None` |
| `object_storage_access_key` | `OBJECT_STORAGE_ACCESS_KEY` | `None` |
| `object_storage_secret_key` | `OBJECT_STORAGE_SECRET_KEY` | `None` |
| `object_storage_bucket` | `OBJECT_STORAGE_BUCKET` | `None` |

## Compatibility Status

- New LLM code should import from `api.core.shared.llm`.
- New content-pipeline structured generation should go through `api.core.content_pipeline.infrastructure.llm.structured_generation`.
- Legacy provider-specific file-cache and Responses-specific code paths have been removed from the backend repo.
