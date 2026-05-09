# Content Pipeline Environment

The unified content pipeline now reads runtime configuration from [api/config.py](/home/linh/Downloads/datn_1/new/data/rinkuzu-ai-api/api/config.py) via `api.config.Settings`.

`Settings` loads values from `.env` at the backend repo root. Field names map directly to uppercase environment variables.

## Core LLM

| Setting field | Env var | Default |
| --- | --- | --- |
| `openai_base_url` | `OPENAI_BASE_URL` | `None` |
| `openai_model` | `OPENAI_MODEL` | `None` |
| `exercise_llm_model` | `EXERCISE_LLM_MODEL` or `ADAPTIVE_EXERCISE_LLM_MODEL` | `None` |
| `openai_api_key` | `OPENAI_API_KEY` | `None` |
| `llm_embedding_model` | `LLM_EMBEDDING_MODEL` | `text-embedding-3-small` |
| `llm_timeout_sec` | `LLM_TIMEOUT_SEC` | `150` |
| `llm_max_retries` | `LLM_MAX_RETRIES` | `2` |
| `llm_max_workers` | `LLM_MAX_WORKERS` or `ADAPTIVE_LLM_MAX_WORKERS` | `8` |
| `llm_max_concurrency` | `LLM_MAX_CONCURRENCY` or `ADAPTIVE_LLM_MAX_CONCURRENCY` | `None` |
| `llm_request_timeout_sec` | `LLM_REQUEST_TIMEOUT_SEC` or `ADAPTIVE_LLM_TIMEOUT_SEC` | `120` |
| `llm_prefetch_timeout_sec` | `LLM_PREFETCH_TIMEOUT_SEC` or `ADAPTIVE_PREFETCH_LLM_TIMEOUT_SEC` | `None` |
| `llm_retry_attempts` | `LLM_RETRY_ATTEMPTS` or `ADAPTIVE_LLM_RETRY_ATTEMPTS` | `3` |
| `llm_retry_backoff_sec` | `LLM_RETRY_BACKOFF_SEC` or `ADAPTIVE_LLM_RETRY_BACKOFF_SEC` | `1.0` |
| `google_api_key` | `GOOGLE_API_KEY` | `None` |
| `gemini_api_key` | `GEMINI_API_KEY` | `None` |

Notes:
- `get_llm()` falls back from `openai_api_key` to `gemini_api_key` to `google_api_key`.
- If no API key is configured, the local compatibility default key is still used for OpenAI-compatible local gateways.
- `exercise_llm_model` lets the exercise/theory flow use a different model than the shared `openai_model`.
- `llm_prefetch_timeout_sec` lets exercise prefetch run with a different wall-clock timeout than the foreground exercise request path.

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
| `content_pipeline_responses_timeout_sec` | `CONTENT_PIPELINE_RESPONSES_TIMEOUT_SEC` | `180` |

## PDF batching and provider-file cache

| Setting field | Env var | Default |
| --- | --- | --- |
| `content_pipeline_pdf_page_batch_size` | `CONTENT_PIPELINE_PDF_PAGE_BATCH_SIZE` | `10` |
| `content_pipeline_pdf_batch_max_bytes` | `CONTENT_PIPELINE_PDF_BATCH_MAX_BYTES` | `4194304` |
| `content_pipeline_file_cache_ttl_hours` | `CONTENT_PIPELINE_FILE_CACHE_TTL_HOURS` | `168` |
| `content_pipeline_batch_failure_ratio_threshold` | `CONTENT_PIPELINE_BATCH_FAILURE_RATIO_THRESHOLD` | `0.5` |

`content_pipeline_pdf_batch_max_bytes` should stay below the upstream provider's real HTTP upload limit. The default is conservative enough for Vercel-hosted OpenAI-compatible APIs.

## Object storage

| Setting field | Env var | Default |
| --- | --- | --- |
| `object_storage_endpoint_external` | `OBJECT_STORAGE_ENDPOINT_EXTERNAL` | `None` |
| `object_storage_access_key` | `OBJECT_STORAGE_ACCESS_KEY` | `None` |
| `object_storage_secret_key` | `OBJECT_STORAGE_SECRET_KEY` | `None` |
| `object_storage_bucket` | `OBJECT_STORAGE_BUCKET` | `None` |

## Compatibility Status

- New implementation modules under `api/core/content_pipeline/infrastructure/` import `api.config` directly.
- Legacy root packages from the old content-processor layout have been removed from the backend repo.
- PDF concept extraction now uses local `PyMuPDF` page slicing plus provider-portable OpenAI-compatible `Files` + `Responses` APIs.
