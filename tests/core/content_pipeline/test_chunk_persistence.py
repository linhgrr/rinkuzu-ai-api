"""Minimal fail-closed coverage for chunk persistence stage."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from api.domains.content_pipeline.application.stages import chunk_persistence as stage
from api.domains.content_pipeline.domain.jobs import PipelineJob


@pytest.mark.asyncio
async def test_persist_document_chunks_propagates_mongo_failure(monkeypatch):
    job = PipelineJob(job_id="job-1", filename="a.pdf", subject_id="subj", retry_count=4)
    states: list[tuple] = []

    async def persist_job_state(job_arg, status, message, progress):
        states.append((job_arg.job_id, status, message, progress))

    async def boom(**kwargs):
        assert kwargs["generation"] == 4
        raise RuntimeError("mongo write failed")

    monkeypatch.setattr(stage, "replace_job_chunks", boom)
    chunk = SimpleNamespace(page_content="x", metadata={})

    with pytest.raises(RuntimeError, match="mongo write failed"):
        await stage.persist_document_chunks(
            job,
            chunks=[chunk],
            chunk_chroma_store=None,
            persist_job_state=persist_job_state,
        )

    assert states  # entered persisting state before failure


@pytest.mark.asyncio
async def test_persist_document_chunks_empty_returns_zero():
    job = PipelineJob(job_id="job-1", filename="a.pdf", subject_id="subj")
    persist = AsyncMock()
    assert (
        await stage.persist_document_chunks(
            job,
            chunks=[],
            chunk_chroma_store=None,
            persist_job_state=persist,
        )
        == 0
    )
    persist.assert_not_called()
