"""Minimal exception-propagation coverage for document chunk replace."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from api.shared.persistence import document_chunks as chunk_store


class _BoomBulk:
    async def __aenter__(self):
        raise RuntimeError("mongo bulk unavailable")

    async def __aexit__(self, exc_type, exc, tb):
        return False


@pytest.mark.asyncio
async def test_replace_job_chunks_propagates_storage_errors(monkeypatch):
    monkeypatch.setattr(
        chunk_store.DocumentChunkDocument,
        "bulk_writer",
        lambda **kwargs: _BoomBulk(),
    )
    chunk = SimpleNamespace(page_content="text", metadata={"chunk_index": 0})

    with pytest.raises(RuntimeError, match="mongo bulk unavailable"):
        await chunk_store.replace_job_chunks(
            job_id="job-1",
            subject_id="subj-1",
            generation=2,
            chunks=[chunk],
        )


@pytest.mark.asyncio
async def test_replace_job_chunks_empty_is_zero():
    assert (
        await chunk_store.replace_job_chunks(
            job_id="job-1", subject_id="s", generation=2, chunks=[]
        )
        == 0
    )
