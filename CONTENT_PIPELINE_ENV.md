# Content Pipeline Environment

The unified content pipeline now reads runtime configuration from [api/config.py](/home/linh/Downloads/datn_1/new/data/rinkuzu-ai-api/api/config.py) via `api.config.Settings`.

`Settings` loads values from `.env` at the backend repo root. Field names map directly to uppercase environment variables.

## Core LLM

| Setting field | Env var | Default |
| --- | --- | --- |
| `llm_base_url` | `LLM_BASE_URL` | `None` |
| `llm_model` | `LLM_MODEL` | `None` |
| `adaptive_exercise_llm_model` | `ADAPTIVE_EXERCISE_LLM_MODEL` | `None` |
| `llm_api_key` | `LLM_API_KEY` | `None` |
| `llm_embedding_model` | `LLM_EMBEDDING_MODEL` | `text-embedding-3-small` |
| `llm_timeout_sec` | `LLM_TIMEOUT_SEC` | `150` |
| `llm_max_retries` | `LLM_MAX_RETRIES` | `2` |
| `adaptive_llm_max_workers` | `ADAPTIVE_LLM_MAX_WORKERS` | `8` |
| `adaptive_llm_max_concurrency` | `ADAPTIVE_LLM_MAX_CONCURRENCY` | `None` |
| `adaptive_llm_timeout_sec` | `ADAPTIVE_LLM_TIMEOUT_SEC` | `120` |
| `adaptive_prefetch_llm_timeout_sec` | `ADAPTIVE_PREFETCH_LLM_TIMEOUT_SEC` | `None` |
| `adaptive_llm_retry_attempts` | `ADAPTIVE_LLM_RETRY_ATTEMPTS` | `3` |
| `adaptive_llm_retry_backoff_sec` | `ADAPTIVE_LLM_RETRY_BACKOFF_SEC` | `1.0` |
| `google_api_key` | `GOOGLE_API_KEY` | `None` |
| `gemini_api_key` | `GEMINI_API_KEY` | `None` |

Notes:
- `get_llm()` falls back from `llm_api_key` to `gemini_api_key` to `google_api_key`.
- If no API key is configured, the local compatibility default key is still used for OpenAI-compatible local gateways.
- `adaptive_exercise_llm_model` lets the adaptive exercise/theory flow use a different model than the shared `llm_model`.
- `adaptive_prefetch_llm_timeout_sec` lets exercise prefetch run with a different wall-clock timeout than the foreground exercise request path.

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
| `vision_pdf_request_timeout_sec` | `VISION_PDF_REQUEST_TIMEOUT_SEC` | `120` |

## OCR And S3

| Setting field | Env var | Default |
| --- | --- | --- |
| `pdf_ocr_concurrency` | `PDF_OCR_CONCURRENCY` | `5` |
| `vision_agent_api_key` | `VISION_AGENT_API_KEY` | `None` |
| `s3_endpoint_url` | `S3_ENDPOINT_URL` | `None` |
| `s3_access_key_id` | `S3_ACCESS_KEY_ID` | `None` |
| `s3_secret_access_key` | `S3_SECRET_ACCESS_KEY` | `None` |
| `s3_bucket_name` | `S3_BUCKET_NAME` | `None` |

## Compatibility Status

- New implementation modules under `api/core/content_pipeline/infrastructure/` import `api.config` directly.
- Legacy root packages from the old content-processor layout have been removed from the backend repo.
- `PDFLoader` still exports `VISION_AGENT_API_KEY` into `os.environ` because the Landing AI client expects that process-level variable.
