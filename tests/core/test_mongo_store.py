import asyncio

from api.shared import mongo_store


class _ClientStub:
    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


def test_mongo_store_is_available_reflects_state(monkeypatch):
    available = True
    monkeypatch.setitem(mongo_store._state, "available", available)
    assert mongo_store.is_available() is True

    unavailable = False
    monkeypatch.setitem(mongo_store._state, "available", unavailable)
    assert mongo_store.is_available() is False


def test_shutdown_mongo_closes_client_and_resets_state(monkeypatch):
    client = _ClientStub()
    monkeypatch.setitem(mongo_store._state, "client", client)
    available = True
    monkeypatch.setitem(mongo_store._state, "available", available)

    asyncio.run(mongo_store.shutdown_mongo())

    assert client.closed is True
    assert mongo_store._state["client"] is None
    assert mongo_store._state["available"] is False
