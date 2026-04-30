# Backend Production Audit — Remaining Work

> **Scope:** All items from the production-readiness audit except secrets rotation, auth (`x-user-id`/`INTERNAL_SERVICE_TOKEN`), and CORS — those are handled separately.
>
> **Phase 0 — DONE** (2025-04-30): `.gitignore` expanded, dead `api/services/` removed, stale `api/requirements.txt` deleted, `Optional[str]` → `str | None`, HTTP status codes corrected (502/500), `str(exc)` leaks removed from session router, validation error meta sanitized, ruff clean.

---

## Phase 1 — Security Blockers

> Estimated effort: **1.5 days**

### 1.1 Path traversal in upload filename
**Priority:** BLOCKER
**File:** `api/routers/pipeline.py:84-88`

Current code does `UPLOAD_DIR / f"{file_id}_{request.filename}"` with no sanitization. A filename like `../../etc/cron.d/x` escapes the upload directory.

**Fix:**
```python
from pathlib import Path

raw_name = Path(request.filename or "").name   # strips directory components
if not raw_name or not raw_name.lower().endswith(".pdf"):
    raise HTTPException(status_code=400, detail="Only PDF files are supported.")
if len(raw_name) > 200 or any(c in raw_name for c in ("\x00", "/", "\\")):
    raise HTTPException(status_code=400, detail="Invalid filename.")

file_id = uuid.uuid4().hex[:8]
save_path = (UPLOAD_DIR / f"{file_id}_{raw_name}").resolve()
if not save_path.is_relative_to(UPLOAD_DIR.resolve()):
    raise HTTPException(status_code=400, detail="Invalid filename.")
```

**Tests to add** (`tests/routers/test_pipeline.py`):
- `../../etc/passwd` → 400
- `\\..\\..\\foo` → 400
- filename with embedded NUL → 400
- filename longer than 200 chars → 400
- normal `lecture.pdf` → passes

---

### 1.2 SSRF in pipeline URL fetcher
**Priority:** BLOCKER
**File:** `api/routers/pipeline.py:87-96`

`file_url` is fetched via `httpx` with no scheme validation, no allowlist, no timeout, no size cap. Cloud metadata endpoints (`169.254.169.254`), `file://`, and `gopher://` are all reachable.

**Fix — new file** `api/core/shared/url_fetch.py`:
```python
"""Safe URL fetching: allowlist, private-IP block, size cap, timeout."""
import ipaddress
import socket
from urllib.parse import urlparse

import aiofiles
import httpx

from ...config import get_settings

ALLOWED_SCHEMES = {"https"}
DOWNLOAD_TIMEOUT = httpx.Timeout(connect=10.0, read=60.0, write=10.0, pool=10.0)


class UnsafeURLError(ValueError):
    pass


def _is_private_ip(host: str) -> bool:
    try:
        infos = socket.getaddrinfo(host, None)
    except socket.gaierror:
        raise UnsafeURLError(f"DNS resolution failed for {host}")
    return any(
        ipaddress.ip_address(info[4][0]).is_private
        or ipaddress.ip_address(info[4][0]).is_loopback
        or ipaddress.ip_address(info[4][0]).is_link_local
        or ipaddress.ip_address(info[4][0]).is_reserved
        for info in infos
    )


def validate_download_url(url: str) -> None:
    parsed = urlparse(url)
    if parsed.scheme not in ALLOWED_SCHEMES:
        raise UnsafeURLError(f"Scheme '{parsed.scheme}' not allowed")
    if not parsed.hostname:
        raise UnsafeURLError("Missing hostname")
    settings = get_settings()
    if settings.download_host_allowlist and parsed.hostname not in settings.download_host_allowlist:
        raise UnsafeURLError(f"Host not in allowlist")
    if _is_private_ip(parsed.hostname):
        raise UnsafeURLError("Private/reserved addresses not allowed")


async def stream_download(url: str, dest_path, max_bytes: int) -> int:
    validate_download_url(url)
    bytes_written = 0
    async with httpx.AsyncClient(timeout=DOWNLOAD_TIMEOUT, follow_redirects=False) as client:
        async with client.stream("GET", url) as resp:
            resp.raise_for_status()
            cl = resp.headers.get("content-length")
            if cl and int(cl) > max_bytes:
                raise UnsafeURLError(f"Content-Length {cl} exceeds limit")
            async with aiofiles.open(dest_path, "wb") as f:
                async for chunk in resp.aiter_bytes(chunk_size=65536):
                    bytes_written += len(chunk)
                    if bytes_written > max_bytes:
                        dest_path.unlink(missing_ok=True)
                        raise UnsafeURLError("Download exceeded size limit")
                    await f.write(chunk)
    return bytes_written
```

**Add to `api/config.py`:**
```python
download_host_allowlist: list[str] = []   # empty = any non-private host
download_max_bytes: int = 100 * 1024 * 1024
```

**Replace download block in `api/routers/pipeline.py`:**
```python
from ..core.shared.url_fetch import stream_download, UnsafeURLError

try:
    await stream_download(request.file_url, save_path, max_bytes=settings.download_max_bytes)
except UnsafeURLError as exc:
    logger.warning("[PipelineRouter] Rejected unsafe URL: {}", exc)
    raise HTTPException(status_code=400, detail="URL not allowed.") from None
except httpx.HTTPError:
    logger.exception("[PipelineRouter] Download failed for {}", request.file_url)
    raise HTTPException(status_code=502, detail="Failed to download file.") from None
```

Also use `stream_download` in `api/routers/quiz_extract.py` instead of its own ad-hoc download — single shared helper.

**Tests to add** (`tests/core/shared/test_url_fetch.py`):
- `file://` scheme → `UnsafeURLError`
- `http://169.254.169.254` → `UnsafeURLError`
- `http://10.0.0.1` → `UnsafeURLError`
- `Content-Length` over limit → `UnsafeURLError`
- streamed body over limit → `UnsafeURLError`
- redirect response → blocked (no `follow_redirects`)
- allowlist configured, hostname not in it → `UnsafeURLError`

---

### 1.3 Clean up `uploads/` after successful pipeline run
**Priority:** HIGH
**File:** `api/routers/pipeline.py`, `api/core/content_pipeline/application/pipeline_runner.py`

Files downloaded to `uploads/` are only deleted on `start_job` failure. Successful runs leave the PDF on disk indefinitely.

**Fix (two-part):**
1. In `PipelineRunner.run` final block, unlink `file_path` if it lives under `UPLOAD_DIR`:
```python
finally:
    file = Path(file_path)
    if file.exists() and str(file).startswith(str(UPLOAD_DIR)):
        file.unlink(missing_ok=True)
```
2. Add a startup janitor in `api/main.py` lifespan to remove files older than 24 h:
```python
import time
for f in UPLOAD_DIR.iterdir():
    if f.is_file() and (time.time() - f.stat().st_mtime) > 86400:
        f.unlink(missing_ok=True)
```

---

### 1.4 Hide Swagger / Redoc in non-dev environments
**Priority:** BLOCKER
**File:** `api/main.py:142-147`, `api/config.py`

`/docs` and `/redoc` are publicly accessible in all environments.

**Add to `api/config.py`:**
```python
from typing import Literal
environment: Literal["dev", "staging", "prod"] = "dev"
```

**Update `api/main.py`:**
```python
settings = get_settings()
_docs = {} if settings.environment == "dev" else {
    "docs_url": None,
    "redoc_url": None,
    "openapi_url": None,
}
app = FastAPI(
    title="ALSS-LEPC Adaptive Learning API",
    description="Adaptive Learning System with SAINT KT + D3QN RL",
    version="1.0.0",
    lifespan=lifespan,
    **_docs,
)
```

Set `ENVIRONMENT=prod` in production `.env`.

---

## Phase 2 — Reliability & Observability

> Estimated effort: **2 days**

### 2.1 Pin all dependencies
**Priority:** BLOCKER
**File:** `requirements.txt`

All packages use `>=` lower bounds with no upper bound — a build next week could use different versions than today.

**Fix:**
```bash
pip install pip-tools
# rename current requirements.txt to requirements.in, keeping >= bounds
pip-compile requirements.in --output-file requirements.txt --resolver=backtracking
```
Commit both `requirements.in` (human-edited) and `requirements.txt` (lockfile). Run `pip-compile --upgrade` periodically for dependency updates.

---

### 2.2 Harden Dockerfile + add `.dockerignore`
**Priority:** BLOCKER
**Files:** `Dockerfile`, new `.dockerignore`

**Issues in current Dockerfile:**
- No multi-stage build — dev tools (`build-essential`, `git`) land in the runtime image.
- HuggingFace model cache at `/app/model_cache` is root-owned; app runs as `user` and cannot write to it at runtime.
- No `.dockerignore` — build context includes `.git/`, `.venv/`, `uploads/`, `api/core/chroma_db/` (vector DB data).
- Port hardcoded to `7860` (HF Spaces); internal port should be `8000`.
- `PYTHONPATH` includes both `/home/user/app` and `/home/user/app/api` — double-import risk.

**New `.dockerignore`:**
```
.git
.gitignore
.venv
.env
.env.*
.pytest_cache
.ruff_cache
.worktrees
__pycache__
**/__pycache__
*.pyc
uploads/*
!uploads/.gitkeep
api/core/chroma_db
docs
tests
*.md
.vscode
.idea
```

**Rewritten `Dockerfile` (multi-stage):**
```dockerfile
FROM python:3.11-slim AS builder
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
RUN apt-get update && apt-get install -y --no-install-recommends build-essential curl \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /build
COPY requirements.txt .
RUN pip install --upgrade pip && pip install --prefix=/install --no-warn-script-location -r requirements.txt

FROM python:3.11-slim AS runtime
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
RUN apt-get update && apt-get install -y --no-install-recommends curl libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd -m -u 1000 user
COPY --from=builder /install /usr/local
USER user
ENV HF_HOME=/home/user/.cache/huggingface \
    TRANSFORMERS_CACHE=/home/user/.cache/huggingface \
    PYTHONPATH=/home/user/app
WORKDIR /home/user/app
COPY --chown=user:user . .
RUN python3 -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')" \
 && python3 -c "import underthesea; underthesea.pos_tag('Chào thế giới')"
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
  CMD curl -fsS http://localhost:8000/api/live || exit 1
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
```

---

### 2.3 Request ID middleware + access logging
**Priority:** HIGH
**New file:** `api/middleware/request_context.py`

Only `quiz_extract.py` generates per-request UUIDs locally. No middleware attaches an ID to all requests or logs per-request timing.

```python
import time
import uuid
from fastapi import Request
from loguru import logger
from starlette.middleware.base import BaseHTTPMiddleware

class RequestContextMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        request_id = request.headers.get("x-request-id") or uuid.uuid4().hex
        request.state.request_id = request_id
        start = time.perf_counter()
        with logger.contextualize(request_id=request_id):
            response = await call_next(request)
            logger.info(
                "request method={} path={} status={} duration_ms={:.1f}",
                request.method, request.url.path,
                response.status_code,
                (time.perf_counter() - start) * 1000,
            )
            response.headers["x-request-id"] = request_id
            return response
```

Wire in `api/main.py` after CORS middleware:
```python
from .middleware.request_context import RequestContextMiddleware
app.add_middleware(RequestContextMiddleware)
```

Refactor `quiz_extract.py` to use `request.state.request_id` instead of generating its own.

---

### 2.4 Configurable log level and format
**Priority:** MEDIUM
**Files:** `api/config.py`, `api/main.py`

Log level and format are not configurable from the environment.

**Add to `api/config.py`:**
```python
from typing import Literal
log_level: str = "INFO"
log_format: Literal["text", "json"] = "text"
```

**Add to `api/main.py` lifespan (top):**
```python
import sys
from loguru import logger
logger.remove()
if settings.log_format == "json":
    logger.add(sys.stdout, serialize=True, level=settings.log_level)
else:
    logger.add(sys.stdout, level=settings.log_level)
```

---

### 2.5 Split `/api/health` into `/api/live` and `/api/ready`
**Priority:** MEDIUM
**File:** `api/main.py:171-189`

Current single `/api/health` mixes liveness and readiness concerns. Kubernetes needs them separate.

**Replace with:**
```python
@app.get("/api/live", include_in_schema=False)
async def liveness():
    return {"status": "ok"}

@app.get("/api/ready")
async def readiness():
    # existing logic from health()
    ...
    if not ready:
        return JSONResponse(status_code=503, content=payload)
    return payload

@app.get("/api/health")
async def health():
    return await readiness()   # backwards-compat alias
```

---

### 2.6 Rate limiting on LLM-backed endpoints
**Priority:** HIGH
**Files:** `requirements.in`, `api/main.py`, LLM-backed routers

No rate limiting exists. Any user can flood `/api/quiz/extract`, `/api/session/{id}/chat`, `/api/quiz/ask-ai`, `/api/pipeline/process`.

**Add to `requirements.in`:** `slowapi>=0.1.9`

**Wire in `api/main.py`:**
```python
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

def _rate_key(request: Request) -> str:
    return request.headers.get("x-user-id") or get_remote_address(request)

limiter = Limiter(key_func=_rate_key)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
```

**Apply on expensive routes:**
```python
# e.g. api/routers/quiz_extract.py
@router.post("/extract")
@limiter.limit("10/minute")
async def extract_quiz(request: Request, ...):
    ...
```

Add per-endpoint limit settings to `api/config.py`:
```python
rate_limit_quiz_extract: str = "10/minute"
rate_limit_tutor_chat: str = "30/minute"
rate_limit_pipeline: str = "5/minute"
```

---

### 2.7 Verify no `time.sleep()` in async paths
**Priority:** MEDIUM
**File:** `api/core/shared/llm.py:47` (`sleep_before_retry`)

`time.sleep()` blocks the event loop if called from an async context. Verify it is only ever called from a thread executor. If not, replace with `await asyncio.sleep(...)`.

**Check:**
```bash
grep -rn "sleep_before_retry\|time\.sleep" api/
```

If `sleep_before_retry` is called directly inside any `async def`, change to `asyncio.sleep`. Add a comment asserting the sync-only contract.

---

## Phase 3 — API Quality & Maintainability

> Estimated effort: **2 days**

### 3.1 API versioning prefix `/v1`
**Priority:** HIGH — much harder to add retroactively once clients exist.
**Files:** `api/main.py`, all `api/routers/*.py`

No versioning prefix exists. Mount a versioned sub-router in `main.py`:
```python
from fastapi import APIRouter
v1 = APIRouter(prefix="/v1")
v1.include_router(session_router.router)
v1.include_router(knowledge_router.router)
v1.include_router(pipeline_router.router)
v1.include_router(history_router.router)
v1.include_router(quiz_extract_router.router)
v1.include_router(quiz_tutor_router.router)
app.include_router(v1)
```

Keep the unprefixed routers mounted in parallel during transition. Add a middleware that sets a `Deprecation: true` response header on legacy paths. Plan removal after one release cycle.

---

### 3.2 Cursor-based pagination on history list endpoints
**Priority:** MEDIUM
**Files:** `api/routers/history.py`, `api/repositories/pipeline_repo.py`, `api/repositories/subject_progress_repo.py`, `api/schemas/history.py`

`GET /api/history/subjects` and `/api/history/pipeline-jobs` use a simple `limit` with no cursor or offset. Users with > `limit` records have no way to see the rest.

**Change:**
- Add `cursor: str | None = None` query param.
- Cursor = base64-url encoded `(completed_at_timestamp, _id_hex)` of last returned item.
- Repository filter: `{"$or": [{"completed_at": {"$lt": ts}}, {"completed_at": ts, "_id": {"$lt": last_id}}]}`, sorted `completed_at desc, _id desc`.
- Response gains `next_cursor: str | None` and `has_more: bool`.

---

### 3.3 Add `response_model` to raw-dict endpoints
**Priority:** LOW
**Files:** `api/routers/pipeline.py`, `api/routers/history.py`, `api/main.py`

Endpoints returning raw dicts have no OpenAPI response schema. Define Pydantic models in `api/schemas/` and annotate:
- `POST /api/pipeline/process` → `ProcessDocumentResponse`
- `GET /api/pipeline/jobs/{job_id}` → `PipelineJobStatusResponse`
- `POST /api/pipeline/jobs/{job_id}/create-session` → `CreateSessionResponse`
- `GET /api/health` / `/api/ready` → `HealthResponse`
- `GET /api/info` → `InfoResponse`

---

### 3.4 Add mypy + tighten ruff rules
**Priority:** MEDIUM
**File:** `pyproject.toml`

No static type checking is enforced. Add:

```toml
[tool.ruff.lint]
select = ["E", "F", "I", "B", "UP", "S", "ASYNC", "RUF", "PT"]
ignore = [
    "E501",
    "B008",
    "S101",   # asserts OK in tests
]

[tool.ruff.lint.per-file-ignores]
"tests/**/*" = ["S101", "S105", "S106"]

[tool.mypy]
python_version = "3.11"
strict_optional = true
warn_unused_ignores = true
disallow_untyped_defs = false
files = ["api"]
plugins = ["pydantic.mypy"]

[[tool.mypy.overrides]]
module = ["motor.*", "chromadb.*", "underthesea.*", "pyvi.*", "agentic_doc.*"]
ignore_missing_imports = true
```

Add `mypy` and `types-*` stubs to `requirements-dev.in`. Document baseline errors in `docs/mypy-baseline.txt` and ratchet down incrementally.

---

### 3.5 CI pipeline
**Priority:** HIGH
**New file:** `.github/workflows/ci.yml`

No CI exists — no lint, type, or test gate on PRs.

```yaml
name: ci
on:
  push:
    branches: [main]
  pull_request:

jobs:
  lint-and-test:
    runs-on: ubuntu-latest
    services:
      mongo:
        image: mongo:7
        ports: ["27017:27017"]
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
          cache: pip
      - run: pip install -r requirements.txt -r requirements-dev.txt
      - run: ruff check api tests
      - run: ruff format --check api tests
      - run: mypy api
      - run: pytest -q --cov=api --cov-fail-under=70
        env:
          MONGO_URL: mongodb://localhost:27017
          LOAD_MODELS: "false"
          ENVIRONMENT: dev

  docker-build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - run: docker build -t rinkuzu-ai-api:ci .
```

---

### 3.6 Pre-commit hooks
**Priority:** MEDIUM
**New file:** `.pre-commit-config.yaml`

```yaml
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.6.9
    hooks:
      - id: ruff
      - id: ruff-format
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.6.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-added-large-files
        args: [--maxkb=2048]
      - id: detect-private-key
```

Document `pre-commit install` in `README.md`.

---

### 3.7 Test coverage backfill
**Priority:** HIGH
**Files:** new tests under `tests/routers/`

Current coverage: no HTTP-level tests for pipeline, history, or quiz routes. No ownership isolation tests.

**Targets (use `httpx.AsyncClient(transport=ASGITransport(app=app))` + `mongomock-motor`):**

| File | Tests |
|------|-------|
| `tests/routers/test_pipeline.py` | happy path, path traversal → 400, SSRF → 400, oversize → 400, job not found → 404, incomplete graph → 409 |
| `tests/routers/test_session.py` | session lifecycle, tutor chat safe errors (400/502 no internal detail) |
| `tests/routers/test_history.py` | pagination cursor, user A cannot read user B jobs |
| `tests/routers/test_quiz_extract.py` | rate limit → 429, oversize PDF → 400 |
| `tests/core/shared/test_url_fetch.py` | all SSRF vectors from task 1.2 |

Coverage gate: `--cov-fail-under=70`, enforced in CI.

---

### 3.8 MongoDB connection pool + TLS config
**Priority:** LOW
**File:** `api/core/shared/mongo_store.py`, `api/config.py`

No connection pool size is configured; Motor uses its default (`maxPoolSize=100`). Under load this may need tuning.

**Add to `api/config.py`:**
```python
mongo_max_pool_size: int = 50
mongo_min_pool_size: int = 5
```

**Update Motor client construction:**
```python
AsyncIOMotorClient(
    mongo_url,
    serverSelectionTimeoutMS=5000,
    maxPoolSize=settings.mongo_max_pool_size,
    minPoolSize=settings.mongo_min_pool_size,
)
```

Add a `README.md` note: production URIs should use `mongodb+srv://` (TLS implicit) or include `?tls=true`; never set `tlsAllowInvalidCertificates=true`.

---

### 3.9 Persist in-memory sessions across restarts
**Priority:** MEDIUM
**File:** `api/core/learning/session.py` (`SessionManager._sessions`)

All active sessions live in process memory. A restart loses in-flight sessions not yet persisted (eviction triggers at 500 sessions, evicting the oldest 20%).

**Recommended fix — Redis-backed session cache:**
1. Add `redis_url: str | None = None` to `api/config.py`.
2. On session save: serialize `Session` state to JSON, `SET session:{id} <json> EX 21600` (6 h TTL).
3. On cache miss in `get_or_recover_session`: fall back to Mongo recovery (already implemented).
4. On Redis unavailable: fall back to in-memory (log a warning).

Lighter alternative (if Redis is not available): flush mastery state to Mongo every N answers, and document that exercise state (current question) is lost on restart.

---

### 3.10 Deduplicate `mongo_store` global vs repository layer
**Priority:** LOW (technical debt)
**Files:** `api/core/shared/mongo_store.py`, `api/repositories/`

Some callers use `mongo_store.load_pipeline_job(...)` (module-level shim), others use `PipelineRepository` directly. Two access paths make it harder to mock and test.

**Goal:** retire module-level shim functions one by one, replacing with repository instances injected via `app.state` + `Depends`. No need to do in one PR — track in a separate issue.

---

## Phase 4 — Stretch / Post-launch

> Estimated effort: **1 day**

### 4.1 OpenTelemetry distributed tracing
Add `opentelemetry-instrumentation-fastapi`, `opentelemetry-instrumentation-httpx`, `opentelemetry-instrumentation-pymongo`. Configure OTLP exporter behind `OTEL_EXPORTER_OTLP_ENDPOINT`. Attach request IDs from Phase 2.3 as span attributes.

### 4.2 Prometheus metrics
Add `prometheus-fastapi-instrumentator`. Expose `/metrics` (behind auth or internal-only port). Custom metrics to add:
- Pipeline job duration histogram
- LLM token usage counter (if available from provider)
- Exercise generation latency percentiles

### 4.3 Background job janitor (scheduled)
Extend startup janitor (Phase 1.3) to a periodic task (APScheduler or FastAPI lifespan background task):
- Delete `uploads/*` older than 24 h every hour.
- Mark Mongo pipeline jobs stuck in `RUNNING` for > `content_pipeline_job_timeout_sec` as `FAILED` with a `timed_out` error code.

### 4.4 Services layer
`api/services/` was removed (empty). Reintroduce with actual content: move thin orchestration out of routers (background task scheduling, cross-repo aggregations) into `services/pipeline_service.py`, `services/session_service.py`. Goal: routers do HTTP/validation only; services own business rules; `core/` owns domain model.

### 4.5 Documentation
- `docs/architecture.md` — pipeline runner, session manager, RAG store data-flow diagram.
- `docs/deployment.md` — env var matrix, uvicorn worker count guidance, Mongo pool tuning, Redis sizing.
- `docs/runbook.md` — LLM timeout recovery, Chroma corruption, Mongo connection drop, how to drain in-flight sessions before restart.

---

## Acceptance Gate

Before declaring the API production-ready, all of these must pass:

- [ ] `ruff check api tests` — clean
- [ ] `mypy api` — no new errors vs baseline
- [ ] `pytest --cov=api --cov-fail-under=70` — passes
- [ ] `docker build .` — succeeds, image runs as uid 1000
- [ ] `/api/live` → 200 always; `/api/ready` → 503 when Mongo is down
- [ ] SSRF vectors (`file://`, `169.254.169.254`, `10.x.x.x`) → 400
- [ ] Path traversal filenames → 400
- [ ] 11th LLM-endpoint request from same user in 60 s → 429 with `Retry-After`
- [ ] Restart server mid-session → client recovers state
- [ ] `ENVIRONMENT=prod` → `/docs` returns 404

---

## Excluded from this audit

The following were identified but are out of scope per project decision:

- **Secrets rotation** — `.env` credentials (Google API key, MongoDB Atlas URI, S3 keys, LangSmith key) must be rotated in the credential provider directly.
- **Auth hardening** — `x-user-id` header model and `INTERNAL_SERVICE_TOKEN` gating.
- **CORS** — wildcard origin + `allow_credentials=True` combination.
