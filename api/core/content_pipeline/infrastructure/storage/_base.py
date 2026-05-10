"""Shared ChromaDB initialization helper."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import chromadb
from chromadb.config import Settings as ChromaSettings
from langchain_chroma import Chroma
from loguru import logger

from api.core.content_pipeline.infrastructure.embed.embedding_client import EmbeddingClient


def init_chroma_store(
    *,
    collection_name: str,
    persist_directory: str,
    embedding_client: Any | None = None,
    log_label: str,
) -> tuple[chromadb.PersistentClient, Chroma, EmbeddingClient]:
    """Initialize a ChromaDB persistent client + LangChain vectorstore.

    Returns (chroma_client, vectorstore, embedding_client).
    """
    Path(persist_directory).mkdir(parents=True, exist_ok=True)

    if embedding_client is None:
        embedding_client = EmbeddingClient()

    chroma_client = chromadb.PersistentClient(
        path=persist_directory,
        settings=ChromaSettings(anonymized_telemetry=False, allow_reset=True),
    )

    vectorstore = Chroma(
        client=chroma_client,
        collection_name=collection_name,
        embedding_function=embedding_client,
    )

    logger.info(
        "{} initialized",
        log_label,
        collection=collection_name,
        persist_dir=persist_directory,
        embedding_model=embedding_client.model_name,
    )
    return chroma_client, vectorstore, embedding_client
