# Content Pipeline Migration Map

## Goal

This document maps the legacy `content-processor` service structure into the
new unified backend package under `api/core/content_pipeline/`.

The legacy `content-processor/` folder has been removed from the repo. The old
paths below are kept as historical migration references only.

## Target Backend Package

```text
api/core/content_pipeline/
  domain/
  application/
  infrastructure/
  interfaces/
```

## Migration Map

### Domain

- `content-processor/src/llm/schemas.py`
  Destination: `api/core/content_pipeline/domain/`
  Notes: concept and relation payload types belong to the domain boundary.

### Application

- `api/core/content_pipeline/orchestrator.py`
  Destination: keep here temporarily, then split by use case and stage.
- `content-processor/src/api/services.py`
  Destination: `api/core/content_pipeline/application/`
  Notes: extract orchestration logic only, not FastAPI concerns.

### Infrastructure

- `content-processor/src/processors/loaders/*`
  Destination: `api/core/content_pipeline/infrastructure/loaders/`
- `content-processor/src/processors/chunkers/*`
  Destination: `api/core/content_pipeline/infrastructure/chunkers/`
- `content-processor/src/llm/extract_chain.py`
  Destination: `api/core/content_pipeline/infrastructure/llm/`
- `content-processor/src/llm/postprocess.py`
  Destination: `api/core/content_pipeline/infrastructure/llm/`
- `content-processor/src/embed/*`
  Destination: `api/core/content_pipeline/infrastructure/embeddings/`
- `content-processor/src/graph/*`
  Destination: `api/core/content_pipeline/infrastructure/graph/`
- `content-processor/src/storage/chroma_store.py`
  Destination: `api/core/content_pipeline/infrastructure/storage/`
- `content-processor/src/prompts/*`
  Destination: `api/core/content_pipeline/infrastructure/prompts/`

### Interfaces

- `content-processor/src/api/routes.py`
  Destination: existing backend routers or `api/core/content_pipeline/interfaces/http/`
- `content-processor/src/api/models.py`
  Destination: backend schemas or interface DTOs
- `content-processor/src/api/dependencies.py`
  Destination: main backend dependency wiring
- `content-processor/src/api/main.py`
  Destination: removed after route integration

### Shared or Miscellaneous

- `content-processor/src/utils/*`
  Destination: relocate only if a helper has a single clear owner
- `content-processor/src/config/__init__.py`
  Destination: merge into `api/config.py` or a backend settings submodule

## Duplicate Ownership To Resolve

- Pipeline orchestration currently exists in both:
  - `api/core/content_pipeline/orchestrator.py`
  - `content-processor/src/api/services.py`
- FastAPI API surface currently exists in both:
  - main backend routers
  - `content-processor/src/api/*`

## Dependency Alignment Notes

- Root `requirements.txt` and `content-processor/requirements_api.txt` are mostly compatible.
- Versions differ materially for:
  - `fastapi`
  - `uvicorn`
  - `langchain-core`
- `api/requirements.txt` is older than root requirements in several places and should not become the source of truth during unification.

## Circular Import Risks To Watch

- `api/core/content_pipeline/orchestrator.py` depends on `mongo_store`.
- Routers depend on `api.core.content_pipeline`.
- Future application/infrastructure splits must avoid importing routers or FastAPI types downward.

## Dependency Direction Rule

Allowed:

- interfaces -> application
- application -> domain
- application -> infrastructure through explicit ports/adapters
- infrastructure -> domain

Not allowed:

- domain -> FastAPI
- domain -> storage clients
- application -> router modules
- infrastructure -> router modules
