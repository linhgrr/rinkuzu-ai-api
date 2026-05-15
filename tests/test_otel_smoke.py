from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.observability import setup_otel


def test_setup_otel_emits_request_spans(monkeypatch):
    exported_spans: list[dict[str, str]] = []

    class FakeExporter:
        def export(self, spans):
            exported_spans.extend(spans)

    class FakeProcessor:
        def __init__(self, exporter):
            self.exporter = exporter

    class FakeProvider:
        def __init__(self, resource):
            self.resource = resource
            self.processors = []

        def add_span_processor(self, processor):
            self.processors.append(processor)

        def force_flush(self):
            return None

        def shutdown(self):
            return None

    class FakeFastAPIInstrumentor:
        @staticmethod
        def instrument_app(app, tracer_provider):
            @app.middleware("http")
            async def emit_span(request, call_next):
                response = await call_next(request)
                for processor in tracer_provider.processors:
                    processor.exporter.export([{"name": f"{request.method} {request.url.path}"}])
                return response

    class FakePymongoInstrumentor:
        def instrument(self, **kwargs):
            return None

    monkeypatch.setattr(
        "api.observability._load_otel_components",
        lambda: {
            "trace": SimpleNamespace(set_tracer_provider=lambda _provider: None),
            "OTLPSpanExporter": lambda _endpoint=None: FakeExporter(),
            "FastAPIInstrumentor": FakeFastAPIInstrumentor,
            "PymongoInstrumentor": FakePymongoInstrumentor,
            "Resource": SimpleNamespace(create=lambda data: data),
            "TracerProvider": FakeProvider,
            "ConsoleSpanExporter": FakeExporter,
            "SimpleSpanProcessor": FakeProcessor,
        },
    )

    app = FastAPI()

    @app.get("/api/ready")
    async def ready():
        return {"ok": True}

    setup_otel(app, SimpleNamespace(otel_enabled=True, otel_service_name="rinkuzu-ai-api"))

    client = TestClient(app)
    response = client.get("/api/ready")

    assert response.status_code == 200
    assert exported_spans == [{"name": "GET /api/ready"}]
