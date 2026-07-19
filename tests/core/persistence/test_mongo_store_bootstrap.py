from unittest.mock import AsyncMock

import pytest

from api.shared import mongo_store


@pytest.mark.asyncio
async def test_init_mongo_never_allows_automatic_index_drops(monkeypatch):
    fake_client = AsyncMock()
    fake_client.__getitem__.return_value = object()
    init_beanie = AsyncMock()

    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    monkeypatch.setattr(mongo_store, "AsyncMongoClient", lambda *_args, **_kwargs: fake_client)
    monkeypatch.setattr(mongo_store, "init_beanie", init_beanie)

    assert await mongo_store.init_mongo("mongodb://localhost:27017/test") is True
    assert init_beanie.await_args.kwargs["allow_index_dropping"] is False
    assert init_beanie.await_args.kwargs["skip_indexes"] is False
