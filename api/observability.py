"""Optional OpenTelemetry bootstrap helpers."""

from __future__ import annotations

import os
from typing import Any

from loguru import logger


def _load_otel_components() -> dict[str, Any]:
    from opentelemetry import trace
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
    from opentelemetry.instrumentation.pymongo import PymongoInstrumentor
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor

    return {
        "trace": trace,
        "OTLPSpanExporter": OTLPSpanExporter,
        "FastAPIInstrumentor": FastAPIInstrumentor,
        "PymongoInstrumentor": PymongoInstrumentor,
        "Resource": Resource,
        "TracerProvider": TracerProvider,
        "ConsoleSpanExporter": ConsoleSpanExporter,
        "SimpleSpanProcessor": SimpleSpanProcessor,
    }


def setup_otel(app: Any, settings: Any) -> Any | None:
    if not getattr(settings, "otel_enabled", False):
        app.state.otel_provider = None
        return None

    components = _load_otel_components()
    resource = components["Resource"].create({"service.name": settings.otel_service_name})
    provider = components["TracerProvider"](resource=resource)
    exporter_endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")
    exporter = (
        components["OTLPSpanExporter"](endpoint=exporter_endpoint)
        if exporter_endpoint
        else components["ConsoleSpanExporter"]()
    )
    provider.add_span_processor(components["SimpleSpanProcessor"](exporter))
    components["trace"].set_tracer_provider(provider)
    components["FastAPIInstrumentor"].instrument_app(app, tracer_provider=provider)
    try:
        components["PymongoInstrumentor"]().instrument(tracer_provider=provider)
    except Exception as exc:  # pragma: no cover - best effort only
        logger.warning("[OTel] Pymongo instrumentation skipped: {}", exc)

    app.state.otel_provider = provider
    app.state.otel_exporter = exporter
    logger.info(
        "[OTel] enabled service_name={} exporter={}",
        settings.otel_service_name,
        type(exporter).__name__,
    )
    return provider


def shutdown_otel(app: Any) -> None:
    provider = getattr(app.state, "otel_provider", None)
    if provider is None:
        return
    try:
        provider.force_flush()
        provider.shutdown()
    except Exception as exc:  # pragma: no cover - best effort only
        logger.warning("[OTel] shutdown warning: {}", exc)
