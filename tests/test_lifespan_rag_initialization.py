from types import SimpleNamespace

from api import lifespan


def test_init_rag_stores_skips_embedding_model_when_models_are_disabled(monkeypatch):
    app = SimpleNamespace(state=SimpleNamespace())

    monkeypatch.setattr(lifespan, "get_settings", lambda: SimpleNamespace(load_models=False))
    monkeypatch.setattr(
        lifespan,
        "EmbeddingClient",
        lambda: (_ for _ in ()).throw(AssertionError("embedding model must not load")),
    )

    lifespan._init_rag_stores(app)

    assert app.state.chunk_chroma_store is None


def test_init_rag_stores_builds_store_when_models_are_enabled(monkeypatch):
    app = SimpleNamespace(state=SimpleNamespace())
    embedding_client = object()
    chunk_store = object()

    monkeypatch.setattr(lifespan, "get_settings", lambda: SimpleNamespace(load_models=True))
    monkeypatch.setattr(lifespan, "EmbeddingClient", lambda: embedding_client)
    monkeypatch.setattr(
        lifespan,
        "ChunkChromaStore",
        lambda *, embedding_client: chunk_store,
    )

    lifespan._init_rag_stores(app)

    assert app.state.chunk_chroma_store is chunk_store
