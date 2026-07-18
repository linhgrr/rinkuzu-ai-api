import asyncio

import pytest

from api.domains.content_pipeline.application.stages import (
    document_loading as document_loading_stage,
)
from api.domains.content_pipeline.application.stages.document_loading import load_document_chunks
from api.domains.content_pipeline.domain.jobs import PipelineJob, PipelineStatus
from api.shared.document_text import DocumentPageText, ExtractedDocumentText


def test_load_document_chunks_updates_progress_and_total_chunks():
    job = PipelineJob(job_id="job-1", filename="lesson.pdf", subject_id="algebra")
    calls: list[tuple[PipelineStatus, str, float]] = []
    document_text_out = {}

    async def persist_job_state(job_arg, status, step, progress):
        assert job_arg is job
        calls.append((status, step, progress))

    async def fake_run_blocking_stage(func, *args, stage_name, timeout_sec=None, **kwargs):
        del timeout_sec
        if stage_name == "document_hashing":
            return "hash-123"
        if stage_name == "document_size_stat":
            return 1234
        if stage_name == "document_chunking":
            assert args[1] == "algebra"
            assert args[0]["metadata"]["file_hash"] == "hash-123"
            return ["chunk-1", "chunk-2", "chunk-3"]
        return func(*args, **kwargs)

    async def fake_load_or_extract_document_text_cached(
        *,
        file_hash: str,
        file_name: str,
        extract_document_text,
        resolve_file_size_bytes,
        file_size_bytes=None,
    ):
        assert file_hash == "hash-123"
        assert file_name == "lesson.pdf"
        assert file_size_bytes is None
        extracted = await extract_document_text()
        resolved_size = await resolve_file_size_bytes()
        assert resolved_size == 1234
        return extracted

    async def fake_extract_document_text_from_file_with_key_pool(file_path: str):
        del file_path
        return ExtractedDocumentText(
            text="OCR text",
            pages=[DocumentPageText(page_number=1, text="OCR text")],
            metadata={"page_count": 1, "provider": "landingai", "model": "dpt-2-mini"},
        )

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(document_loading_stage, "run_blocking_stage", fake_run_blocking_stage)
    monkeypatch.setattr(
        document_loading_stage,
        "extract_document_text_from_file_with_key_pool",
        fake_extract_document_text_from_file_with_key_pool,
    )
    monkeypatch.setattr(
        document_loading_stage,
        "load_or_extract_document_text_cached",
        fake_load_or_extract_document_text_cached,
    )

    try:
        chunks = asyncio.run(
            load_document_chunks(
                job,
                file_path="fixtures/lesson.pdf",
                persist_job_state=persist_job_state,
                document_text_out=document_text_out,
            )
        )
    finally:
        monkeypatch.undo()

    assert chunks == ["chunk-1", "chunk-2", "chunk-3"]
    assert document_text_out["document_text"].text == "OCR text"
    assert job.total_chunks == 3
    assert job.total_pages == 1
    assert calls == [
        (PipelineStatus.LOADING, "Loading PDF...", 0.05),
        (PipelineStatus.LOADING, "Checking OCR cache...", 0.05),
        (PipelineStatus.CHUNKING, "Chunking PDF text (1 pages)...", 0.05),
        (PipelineStatus.CHUNKING, "PDF text chunked", 0.10),
    ]


def test_load_document_chunks_reuses_cached_ocr_record():
    job = PipelineJob(job_id="job-2", filename="lesson.pdf", subject_id="algebra")
    stage_names: list[str] = []

    async def persist_job_state(*_args):
        return None

    async def fake_run_blocking_stage(func, *args, stage_name, timeout_sec=None, **kwargs):
        del timeout_sec
        stage_names.append(stage_name)
        if stage_name == "document_hashing":
            return "hash-cached"
        if stage_name == "document_chunking":
            assert args[0]["metadata"]["ocr_cache_hit"] is True
            assert args[0]["metadata"]["file_hash"] == "hash-cached"
            return ["cached-chunk"]
        return func(*args, **kwargs)

    async def fake_load_or_extract_document_text_cached(
        *,
        file_hash: str,
        file_name: str,
        extract_document_text,
        resolve_file_size_bytes,
        file_size_bytes=None,
    ):
        del extract_document_text, resolve_file_size_bytes, file_size_bytes
        assert file_hash == "hash-cached"
        assert file_name == "lesson.pdf"
        return ExtractedDocumentText(
            text="Cached OCR text",
            pages=[
                DocumentPageText(page_number=1, text="Trang 1"),
                DocumentPageText(page_number=2, text="Trang 2"),
            ],
            metadata={
                "page_count": 2,
                "provider": "landingai",
                "model": "dpt-2-mini",
                "ocr_cache_hit": True,
                "file_hash": "hash-cached",
            },
        )

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(document_loading_stage, "run_blocking_stage", fake_run_blocking_stage)
    monkeypatch.setattr(
        document_loading_stage,
        "load_or_extract_document_text_cached",
        fake_load_or_extract_document_text_cached,
    )

    try:
        chunks = asyncio.run(
            load_document_chunks(
                job,
                file_path="fixtures/lesson.pdf",
                persist_job_state=persist_job_state,
            )
        )
    finally:
        monkeypatch.undo()

    assert chunks == ["cached-chunk"]
    assert job.total_chunks == 1
    assert job.total_pages == 2
    assert stage_names == ["document_hashing", "document_chunking"]
