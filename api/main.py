"""
main.py — FastAPI app entry point for Adaptive Learning Demo.
"""

from typing import Any, cast

from bson import ObjectId
from fastapi import FastAPI
from fastapi.encoders import ENCODERS_BY_TYPE
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi
from fastapi.responses import ORJSONResponse
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from .config import get_settings
from .domains.assistant import router as ask_rin_router
from .domains.content_pipeline import router as pipeline_router
from .domains.learning import history_router, knowledge_router
from .domains.learning import router as session_router
from .domains.quiz.router import drafts_router as quiz_drafts_router
from .exceptions import error_json_response, register_exception_handlers
from .lifespan import lifespan
from .middleware.request_context import RequestContextMiddleware
from .rate_limit import limiter
from .routers import admin_ocr_keys as admin_ocr_keys_router
from .routers import admin_usage as admin_usage_router
from .schemas.common import InfoResponse, ReadinessResponse, StandardResponse, ok
from .shared import mongo_store

ENCODERS_BY_TYPE.setdefault(ObjectId, str)


settings = get_settings()
_is_dev = settings.environment == "dev"
app = FastAPI(
    title="ALSS-LEPC Adaptive Learning API",
    description="Adaptive Learning System with SAINT KT + vanilla DQN RL",
    version="1.0.0",
    default_response_class=ORJSONResponse,
    contact={
        "name": "Rinkuzu Team",
        "url": "https://github.com/rinkuzu/rinkuzu-ai-api",
    },
    license_info={
        "name": "MIT",
        "url": "https://opensource.org/licenses/MIT",
    },
    lifespan=lifespan,
    docs_url="/docs" if _is_dev else None,
    redoc_url="/redoc" if _is_dev else None,
    openapi_url="/openapi.json" if _is_dev else None,
)


# Explicit Header(...) params that FastAPI emits as optional (default=None).
# Generated FE clients must treat these as required on protected operations.
_PROXY_HEADER_NAMES = frozenset({"x-service-token", "x-user-id", "x-user-role"})

_PUBLIC_OPENAPI_OPERATIONS: dict[str, set[str]] = {
    "/api/ready": {"get"},
    "/api/health": {"get"},
    "/api/info": {"get"},
    "/api/v1/pipeline/status": {"get"},
}


def _build_openapi_security() -> tuple[dict[str, dict[str, str]], list[dict[str, list[str]]]]:
    # Always advertise the service-token scheme. Runtime still fails closed when
    # the token is unconfigured (500) or invalid (401); docs must not hide the
    # requirement just because a local process has no token set.
    schemes: dict[str, dict[str, str]] = {
        "XUserIdHeader": {
            "type": "apiKey",
            "in": "header",
            "name": "x-user-id",
            "description": "Authenticated user id forwarded by the frontend proxy.",
        },
        "XServiceTokenHeader": {
            "type": "apiKey",
            "in": "header",
            "name": "x-service-token",
            "description": "Shared internal token used by the frontend proxy when calling the backend API.",
        },
    }
    requirement: dict[str, list[str]] = {
        "XUserIdHeader": [],
        "XServiceTokenHeader": [],
    }
    return schemes, [requirement]


def _without_null_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Collapse a nullable one-type schema while preserving its metadata."""
    any_of = schema.get("anyOf")
    if not isinstance(any_of, list):
        return schema
    non_null = [item for item in any_of if isinstance(item, dict) and item.get("type") != "null"]
    if len(non_null) != 1 or len(non_null) == len(any_of):
        return schema
    return {
        **non_null[0],
        **{key: value for key, value in schema.items() if key not in {"anyOf", "default"}},
    }


def _require_proxy_headers_in_openapi(openapi_schema: dict[str, Any]) -> None:
    """Mark explicit proxy header parameters required on protected operations.

    Runtime Header defaults and 401/403 behavior stay unchanged; this only
    tightens the OpenAPI contract so generated FE types enforce the headers.
    Public operations (``security=[]``) are left alone and are not marked
    required for these headers.
    """
    for path, path_item in (openapi_schema.get("paths") or {}).items():
        if not isinstance(path_item, dict):
            continue
        for method, operation in path_item.items():
            if method.startswith("x-") or not isinstance(operation, dict):
                continue
            if method in _PUBLIC_OPENAPI_OPERATIONS.get(path, set()):
                continue
            if operation.get("security") == []:
                continue
            for param in operation.get("parameters") or []:
                if not isinstance(param, dict):
                    continue
                name = (param.get("name") or "").lower()
                if param.get("in") == "header" and name in _PROXY_HEADER_NAMES:
                    param["required"] = True
                    schema = param.get("schema")
                    if not isinstance(schema, dict):
                        continue
                    param["schema"] = _without_null_schema(schema)


def custom_openapi() -> dict[str, Any]:
    if app.openapi_schema:
        return app.openapi_schema

    openapi_schema = get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        routes=app.routes,
    )

    security_schemes, security = _build_openapi_security()
    components = openapi_schema.setdefault("components", {})
    components.setdefault("securitySchemes", {}).update(security_schemes)
    openapi_schema["servers"] = [
        {
            "url": "/",
            "description": "Same-origin deployment behind the current API host.",
        }
    ]
    openapi_schema["security"] = security

    for path, methods in _PUBLIC_OPENAPI_OPERATIONS.items():
        path_item = openapi_schema.get("paths", {}).get(path, {})
        for method in methods:
            operation = path_item.get(method)
            if operation is not None:
                operation["security"] = []

    _require_proxy_headers_in_openapi(openapi_schema)

    app.openapi_schema = openapi_schema
    return openapi_schema


app.openapi = custom_openapi  # type: ignore[method-assign]

register_exception_handlers(app)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, cast("Any", _rate_limit_exceeded_handler))

# Middleware — outermost first (last added = outermost)
app.add_middleware(RequestContextMiddleware)

_cors_origins = list(settings.cors_origins)
_cors_allow_credentials = True
if "*" in _cors_origins:
    if settings.environment == "prod":
        raise RuntimeError(
            "CORS_ORIGINS cannot be ['*'] in production — set an explicit allowlist."
        )
    # Browsers reject `Access-Control-Allow-Origin: *` paired with credentials. Drop
    # credentials so the dev wildcard does not produce an invalid header combination.
    _cors_allow_credentials = False

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=_cors_allow_credentials,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(session_router.router)
app.include_router(knowledge_router.router)
app.include_router(pipeline_router.router)
app.include_router(history_router.router)
app.include_router(quiz_drafts_router)
app.include_router(admin_ocr_keys_router.router)
app.include_router(admin_usage_router.router)
app.include_router(ask_rin_router.router)


@app.get("/api/live", include_in_schema=False)
async def liveness() -> Any:
    """Kubernetes liveness probe — always 200 while the process is running."""
    return ok({"status": "ok"})


def _build_readiness_payload() -> tuple[dict, bool]:
    cfg = get_settings()
    models_loaded = getattr(app.state, "session_manager", None) is not None
    models_ready = models_loaded if cfg.load_models else True
    mongo_available = mongo_store.is_available()
    pipeline_service_ready = getattr(app.state, "content_pipeline_service", None) is not None
    content_pipeline_available = bool(getattr(app.state, "content_processor_available", False))
    ready = mongo_available and models_ready and pipeline_service_ready
    payload = {
        "status": "ok" if ready else "degraded",
        "ready": ready,
        "mongo_available": mongo_available,
        "models_enabled": cfg.load_models,
        "models_loaded": models_loaded,
        "content_pipeline_available": content_pipeline_available,
        "content_pipeline_service_ready": pipeline_service_ready,
    }
    return payload, ready


@app.get("/api/ready", response_model=StandardResponse[ReadinessResponse])
async def readiness() -> Any:
    """Kubernetes readiness probe — 503 until all dependencies are up."""
    payload, ready = _build_readiness_payload()
    if not ready:
        return error_json_response(
            code="service_unavailable",
            message="Service unavailable",
            detail="Adaptive API is not ready",
            status_code=503,
            meta=payload,
        )
    return ok(payload)


@app.get("/api/health", response_model=StandardResponse[ReadinessResponse])
async def health() -> Any:
    """Backwards-compat alias for /api/ready."""
    payload, ready = _build_readiness_payload()
    if not ready:
        return error_json_response(
            code="service_unavailable",
            message="Service unavailable",
            detail="Adaptive API is not ready",
            status_code=503,
            meta=payload,
        )
    return ok(payload)


@app.get("/api/info", response_model=StandardResponse[InfoResponse])
async def info() -> Any:
    cfg = get_settings()
    manager = getattr(app.state, "session_manager", None)
    return ok(
        {
            "models_enabled": cfg.load_models,
            "models_loaded": manager is not None,
            "n_concepts": manager.n_concepts if manager else 0,
            "mongo_available": mongo_store.is_available(),
            "content_pipeline_available": bool(
                getattr(app.state, "content_processor_available", False)
            ),
        }
    )
