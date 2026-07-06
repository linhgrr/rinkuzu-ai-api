"""Lightweight S3 client helper shared by API domains."""

from __future__ import annotations

from typing import Any

import boto3
from botocore.client import Config

from api.config import get_settings


def get_s3_client() -> Any:
    settings = get_settings()
    if not settings.s3_available:
        return None

    return boto3.client(
        "s3",
        endpoint_url=settings.object_storage_client_endpoint,
        region_name=settings.object_storage_region,
        aws_access_key_id=settings.object_storage_access_key,
        aws_secret_access_key=settings.object_storage_secret_key,
        config=Config(s3={"addressing_style": settings.object_storage_addressing_style or "path"}),
    )


def get_quiz_draft_s3_client(endpoint_url: str | None = None) -> Any:
    """Build a bounded S3 client for request-independent quiz extraction jobs."""
    settings = get_settings()
    if not settings.s3_available:
        return None

    return boto3.client(
        "s3",
        endpoint_url=endpoint_url or settings.object_storage_client_endpoint,
        region_name=settings.object_storage_region,
        aws_access_key_id=settings.object_storage_access_key,
        aws_secret_access_key=settings.object_storage_secret_key,
        config=Config(
            connect_timeout=float(settings.object_storage_quiz_connect_timeout_sec),
            read_timeout=float(settings.object_storage_quiz_read_timeout_sec),
            retries={
                "max_attempts": int(settings.object_storage_quiz_retry_attempts),
                "mode": "standard",
            },
            s3={"addressing_style": settings.object_storage_addressing_style or "path"},
        ),
    )
