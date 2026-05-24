"""MongoDB + Beanie bootstrap utilities."""

from __future__ import annotations

import inspect
import os
from typing import TYPE_CHECKING, Any, TypedDict

from beanie import init_beanie
from loguru import logger
from pymongo import AsyncMongoClient

from api.config import get_settings
from api.core.shared.persistence.documents import (
    DocumentChunkDocument,
    DocumentOCRRecordDocument,
    PipelineJobDocument,
    QuizDraftDocument,
    SubjectProgressDocument,
)

if TYPE_CHECKING:
    from pymongo.asynchronous.client_session import AsyncClientSession


class _MongoState(TypedDict):
    available: bool
    client: AsyncMongoClient[Any] | None


_state: _MongoState = {
    "available": False,
    "client": None,
}


async def init_mongo(mongodb_uri: str | None = None) -> bool:
    if not mongodb_uri:
        mongodb_uri = get_settings().mongodb_uri

    if not mongodb_uri:
        logger.warning("[MongoDB] MONGODB_URI not set — persistence disabled")
        _state["available"] = False
        return False

    allow_index_dropping = get_settings().environment in {"dev", "staging"}
    skip_indexes = bool(os.environ.get("PYTEST_CURRENT_TEST"))

    try:
        client: AsyncMongoClient[Any] = AsyncMongoClient(
            mongodb_uri,
            serverSelectionTimeoutMS=5000,
            connectTimeoutMS=5000,
        )
        await client.admin.command("ping")
        db = client["adaptive_learning"]
        await init_beanie(
            database=db,
            document_models=[
                PipelineJobDocument,
                SubjectProgressDocument,
                QuizDraftDocument,
                DocumentChunkDocument,
                DocumentOCRRecordDocument,
            ],
            allow_index_dropping=allow_index_dropping,
            skip_indexes=skip_indexes,
        )
        _state["client"] = client
        _state["available"] = True
        logger.info("[MongoDB] ✓ Connected to adaptive_learning database via Beanie")
    except Exception:
        logger.exception("[MongoDB] ✗ Could not connect — persistence disabled")
        _state["available"] = False
        _state["client"] = None
        return False
    return True


async def shutdown_mongo() -> None:
    client = _state.get("client")
    if client is None:
        return
    close_result = client.close()
    if inspect.isawaitable(close_result):
        await close_result
    _state["client"] = None
    _state["available"] = False


def start_session() -> AsyncClientSession:
    client = _state.get("client")
    if client is None:
        raise RuntimeError("MongoDB client not initialized")
    return client.start_session()


def is_available() -> bool:
    return bool(_state["available"])
