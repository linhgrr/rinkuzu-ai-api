"""Minimal coverage for ordinary Chroma job delete."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from api.domains.content_pipeline.infrastructure.storage.chunk_chroma_store import (
    ChunkChromaStore,
)


def _store_with_collection(collection) -> ChunkChromaStore:
    store = ChunkChromaStore.__new__(ChunkChromaStore)
    store.collection_name = "document_chunks"
    store.chroma_client = SimpleNamespace(get_collection=lambda name: collection)
    store.vectorstore = None
    store.embedding_client = None
    store.persist_directory = "/tmp"
    return store


def test_delete_by_job_deletes_matching_ids_and_propagates_errors():
    collection = MagicMock()
    collection.get.return_value = {"ids": ["job-1_chunk_0", "job-1_chunk_1"]}
    store = _store_with_collection(collection)

    deleted = store.delete_by_job("job-1")
    assert deleted == 2
    collection.delete.assert_called_once_with(ids=["job-1_chunk_0", "job-1_chunk_1"])

    collection.get.side_effect = RuntimeError("chroma down")
    with pytest.raises(RuntimeError, match="chroma down"):
        store.delete_by_job("job-1")


def test_delete_by_job_generation_never_deletes_other_generation():
    collection = MagicMock()
    collection.get.return_value = {"ids": ["job-1:generation:2:chunk:0"]}
    store = _store_with_collection(collection)

    deleted = store.delete_by_job("job-1", generation=2)

    assert deleted == 1
    collection.get.assert_called_once_with(
        where={
            "$and": [
                {"job_id": {"$eq": "job-1"}},
                {"generation": {"$eq": 2}},
            ]
        }
    )
    collection.delete.assert_called_once_with(ids=["job-1:generation:2:chunk:0"])


def test_retrieve_is_generation_scoped_and_propagates_failures():
    store = _store_with_collection(MagicMock())
    store.vectorstore = MagicMock()
    store.vectorstore.similarity_search_with_score.return_value = []

    assert store._retrieve_sync("question", "job-1", 3, 5) == []
    store.vectorstore.similarity_search_with_score.assert_called_once_with(
        query="question",
        k=5,
        filter={
            "$and": [
                {"job_id": {"$eq": "job-1"}},
                {"generation": {"$eq": 3}},
            ]
        },
    )

    store.vectorstore.similarity_search_with_score.side_effect = RuntimeError("chroma down")
    with pytest.raises(RuntimeError, match="chroma down"):
        store._retrieve_sync("question", "job-1", 3, 5)
