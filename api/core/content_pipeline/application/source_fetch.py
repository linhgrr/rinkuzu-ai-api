"""Re-download a job's source PDF from object storage for retry/recovery."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any
import uuid

from api.config import get_settings
from api.core.shared.s3 import get_s3_client


def _bucket_name() -> str | None:
    return get_settings().object_storage_bucket


def _read_object_bytes(s3_client: Any, bucket: str, key: str) -> bytes:
    resp = s3_client.get_object(Bucket=bucket, Key=key)
    body = resp.get("Body")
    return body.read() if body else b""


async def download_source_to_dir(s3_key: str, dest_dir: str) -> str:
    bucket = _bucket_name()
    s3_client = get_s3_client()
    if not bucket or s3_client is None:
        raise ValueError("Object storage is not configured")
    data = await asyncio.to_thread(_read_object_bytes, s3_client, bucket, s3_key)
    if not data.startswith(b"%PDF-"):
        raise ValueError("Source object is not a valid PDF")
    await asyncio.to_thread(lambda: Path(dest_dir).mkdir(parents=True, exist_ok=True))
    out = Path(dest_dir) / f"{uuid.uuid4().hex[:8]}_{Path(s3_key).name}"
    await asyncio.to_thread(out.write_bytes, data)
    return str(out)
