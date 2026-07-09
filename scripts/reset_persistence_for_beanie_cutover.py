from __future__ import annotations

import argparse
import asyncio
from importlib import import_module
from pathlib import Path
import sys

from loguru import logger
from pymongo import AsyncMongoClient

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

MONGO_COLLECTIONS = [
    "al_pipeline_jobs",
    "al_subject_progress",
    "quiz_drafts",
    "al_document_chunks",
]


def _get_settings():
    return import_module("api.config").get_settings()


def _get_chunk_chroma_store_type() -> type:
    return import_module(
        "api.domains.content_pipeline.infrastructure.storage.chunk_chroma_store"
    ).ChunkChromaStore


async def reset_mongo() -> None:
    settings = _get_settings()
    mongodb_uri = settings.mongodb_uri
    if not mongodb_uri:
        raise RuntimeError("MONGODB_URI is not configured")
    client = AsyncMongoClient(mongodb_uri)
    try:
        db = client["adaptive_learning"]
        for name in MONGO_COLLECTIONS:
            await db[name].drop()
            logger.info("dropped mongo collection: {}", name)
    finally:
        await client.close()


def reset_chroma() -> None:
    chunk_chroma_store_type = _get_chunk_chroma_store_type()
    chunk_chroma_store_type().reset_collection()
    logger.info("reset chroma collection: document_chunks")


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--force",
        action="store_true",
        help="Actually drop Mongo collections and reset Chroma collections.",
    )
    args = parser.parse_args()
    if not args.force:
        raise SystemExit("Refusing to run without --force")
    await reset_mongo()
    reset_chroma()
    logger.info("beanie cutover reset complete")


if __name__ == "__main__":
    asyncio.run(main())
