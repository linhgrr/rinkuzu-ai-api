from types import SimpleNamespace
from unittest.mock import AsyncMock

from pymongo.errors import DuplicateKeyError
import pytest

from api.domains.assistant import repository
from api.exceptions import AppError


class _ConversationCollection:
    def __init__(self, *, claimed=None, existing=None, deleted_count=1):
        self.claimed = claimed
        self.existing = existing
        self.deleted_count = deleted_count
        self.find_one_and_update = AsyncMock(return_value=claimed)
        self.find_one = AsyncMock(return_value=existing)
        self.delete_one = AsyncMock(return_value=SimpleNamespace(deleted_count=deleted_count))


class _DeleteQuery:
    def __init__(self):
        self.delete = AsyncMock()


class _Field:
    __hash__ = object.__hash__

    def __eq__(self, other):
        return ("eq", other)


class _ConversationDocument:
    conversation_id = _Field()
    in_flight_request_id = _Field()


class _RequestDocument:
    user_id = _Field()
    client_request_id = _Field()
    conversation_id = _Field()
    find_one = AsyncMock(return_value=None)
    insert_mock = AsyncMock()

    def __init__(self, **values):
        self.__dict__.update(values)

    async def insert(self):
        return await self.insert_mock()


class _MessageDocument:
    user_id = _Field()
    conversation_id = _Field()


@pytest.mark.asyncio
async def test_begin_turn_releases_conversation_lock_on_duplicate_request(monkeypatch):
    collection = _ConversationCollection(claimed={"conversation_id": "conversation-1"})
    _ConversationDocument.get_pymongo_collection = lambda: collection
    _RequestDocument.find_one = AsyncMock(return_value=None)
    _RequestDocument.insert_mock = AsyncMock(side_effect=DuplicateKeyError("duplicate"))
    monkeypatch.setattr(repository, "AskRinConversationDocument", _ConversationDocument)
    monkeypatch.setattr(repository, "AskRinRequestDocument", _RequestDocument)
    monkeypatch.setattr(
        repository,
        "_get_or_create_conversation",
        AsyncMock(return_value={"conversation_id": "conversation-1"}),
    )
    release = AsyncMock()
    monkeypatch.setattr(repository, "_release_conversation", release)

    with pytest.raises(AppError, match="Wait for the current response"):
        await repository.begin_turn(
            user_id="user-1",
            context_id="quiz:context-1",
            client_request_id="request-1",
            message="Help",
        )

    release.assert_awaited_once_with("conversation-1", "request-1")


@pytest.mark.asyncio
async def test_delete_conversation_rejects_an_active_stream(monkeypatch):
    collection = _ConversationCollection(claimed=None, existing={"_id": "conversation-1"})
    _ConversationDocument.get_pymongo_collection = lambda: collection
    monkeypatch.setattr(repository, "AskRinConversationDocument", _ConversationDocument)

    with pytest.raises(AppError, match="Wait for the current response"):
        await repository.delete_conversation("user-1", "quiz:context-1")

    collection.delete_one.assert_not_awaited()


@pytest.mark.asyncio
async def test_delete_conversation_claims_then_deletes_owned_history(monkeypatch):
    collection = _ConversationCollection(claimed={"conversation_id": "conversation-1"})
    message_query = _DeleteQuery()
    request_query = _DeleteQuery()
    _ConversationDocument.get_pymongo_collection = lambda: collection
    _MessageDocument.find = lambda *args: message_query
    _RequestDocument.find = lambda *args: request_query
    monkeypatch.setattr(repository, "AskRinConversationDocument", _ConversationDocument)
    monkeypatch.setattr(repository, "AskRinMessageDocument", _MessageDocument)
    monkeypatch.setattr(repository, "AskRinRequestDocument", _RequestDocument)

    assert await repository.delete_conversation("user-1", "quiz:context-1") is True
    message_query.delete.assert_awaited_once()
    request_query.delete.assert_awaited_once()
    collection.delete_one.assert_awaited_once()
