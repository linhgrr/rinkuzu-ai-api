"""Lightweight S3 client helper shared by API domains."""

from __future__ import annotations

import boto3
from botocore.client import Config

from api.config import get_settings


def get_s3_client():
    settings = get_settings()
    if not settings.s3_available:
        return None

    return boto3.client(
        "s3",
        endpoint_url=settings.object_storage_client_endpoint,
        region_name=settings.object_storage_region,
        aws_access_key_id=settings.object_storage_access_key,
        aws_secret_access_key=settings.object_storage_secret_key,
        config=Config(
            s3={"addressing_style": settings.object_storage_addressing_style or "path"}
        ),
    )
