from anyio import Path
import pytest

from api.domains.content_pipeline.application import source_fetch
from api.domains.content_pipeline.domain.errors import (
    PipelineInvalidSourceError,
    PipelineSourceDownloadError,
)


@pytest.mark.asyncio
async def test_download_source_from_s3_writes_file(monkeypatch, tmp_path):
    class FakeBody:
        def read(self):
            return b"%PDF-1.4 fake"

    class FakeS3:
        def get_object(self, Bucket, Key):
            return {"Body": FakeBody()}

    monkeypatch.setattr(source_fetch, "get_s3_client", FakeS3)
    monkeypatch.setattr(source_fetch, "_bucket_name", lambda: "bucket")
    dest = await source_fetch.download_source_to_dir("uploads/x/a.pdf", str(tmp_path))
    p = Path(dest)
    assert await p.exists()
    assert (await p.read_bytes()).startswith(b"%PDF")


@pytest.mark.asyncio
async def test_download_source_rejects_non_pdf(monkeypatch, tmp_path):
    class FakeBody:
        def read(self):
            return b"not a pdf"

    class FakeS3:
        def get_object(self, Bucket, Key):
            return {"Body": FakeBody()}

    monkeypatch.setattr(source_fetch, "get_s3_client", FakeS3)
    monkeypatch.setattr(source_fetch, "_bucket_name", lambda: "bucket")
    with pytest.raises(PipelineInvalidSourceError, match="not a valid PDF"):
        await source_fetch.download_source_to_dir("uploads/x/a.pdf", str(tmp_path))


@pytest.mark.asyncio
async def test_download_source_raises_when_unconfigured(monkeypatch, tmp_path):
    monkeypatch.setattr(source_fetch, "get_s3_client", lambda: None)
    monkeypatch.setattr(source_fetch, "_bucket_name", lambda: None)
    with pytest.raises(PipelineSourceDownloadError, match="not configured"):
        await source_fetch.download_source_to_dir("uploads/x/a.pdf", str(tmp_path))
