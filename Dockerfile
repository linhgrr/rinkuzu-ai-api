# Pinned uv for reproducible locked installs from uv.lock.
# Runtime venv lives at the identical absolute path across stages: /opt/venv
# Global ARG so multi-stage FROM can expand UV_VERSION (Docker forbids ${} in COPY --from).
ARG UV_VERSION=0.11.29
FROM ghcr.io/astral-sh/uv:${UV_VERSION} AS uv

# ── Stage 1: builder ──────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

COPY --from=uv /uv /uvx /bin/

ENV PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=0 \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_LOCKED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Dependency layer (project install deferred for better cache hits).
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    --mount=type=bind,source=README.md,target=README.md \
    uv sync --locked --no-dev --no-install-project

# Install the application package into /opt/venv (non-editable for relocatable copy).
COPY pyproject.toml README.md uv.lock ./
COPY api ./api
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --no-editable

# Fail closed on dependency metadata conflicts before baking runtime.
# Production images omit pip (no dev extra); pin check to the project venv explicitly
# (bare `uv pip check` can resolve to system Python instead of UV_PROJECT_ENVIRONMENT).
RUN uv pip check --python /opt/venv/bin/python

# ── Stage: performance (dev + otel extras, same lock, same venv path) ─────────
FROM python:3.11-slim AS perf

COPY --from=uv /uv /uvx /bin/

ENV PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=0 \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_LOCKED=1 \
    VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:$PATH"

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml README.md uv.lock ./
COPY api ./api
COPY tests/test_benchmark_smoke.py tests/locustfile.py ./tests/
COPY scripts ./scripts
COPY models ./models

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --extra dev --extra otel

# Dev+otel installs pip via pip-audit; keep both fail-closed checks aligned with CI.
RUN uv pip check --python /opt/venv/bin/python \
    && /opt/venv/bin/python -m pip check
RUN /opt/venv/bin/python scripts/dependency_runtime_smoke.py

# ── Stage 2: runtime ──────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    # Identical absolute path as builder so the copied venv stays valid.
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    PATH="/opt/venv/bin:$PATH" \
    HF_HOME=/home/user/.cache/huggingface \
    TRANSFORMERS_CACHE=/home/user/.cache/huggingface \
    PYTHONPATH=/home/user/app

RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd -m -u 1000 user

# Copy the locked virtualenv at the same absolute path used during build.
COPY --from=builder /opt/venv /opt/venv

USER user

WORKDIR /home/user/app
COPY --chown=user:user . .

# Production dependency/runtime smoke (CPU torch + checkpoint import).
RUN python scripts/dependency_runtime_smoke.py

# Pre-cache models into user-owned cache dir so they're writable at runtime
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')" && \
    python -c "import underthesea; underthesea.pos_tag('Chào thế giới')"

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
    CMD curl -fsS http://localhost:7860/api/live || exit 1

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "7860"]
