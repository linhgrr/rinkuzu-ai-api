"""
chunk_chroma_store.py — ChromaDB storage for document chunks (RAG source).
"""

import asyncio
from pathlib import Path
from typing import Any

import chromadb
from chromadb.config import Settings as ChromaSettings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from loguru import logger

from api.core.content_pipeline.infrastructure.embed.embedding_client import EmbeddingClient

_DEFAULT_PERSIST_DIRECTORY = str(Path(__file__).parent.parent.parent.parent / "chroma_db")


class ChunkChromaStore:
    """ChromaDB store for document chunks, used as RAG retrieval source for tutor chat."""

    def __init__(
        self,
        collection_name: str = "document_chunks",
        persist_directory: str | None = None,
        embedding_client: Any | None = None,
    ):
        self.persist_directory = persist_directory or _DEFAULT_PERSIST_DIRECTORY
        self.collection_name = collection_name

        Path(self.persist_directory).mkdir(parents=True, exist_ok=True)

        if embedding_client is None:
            embedding_client = EmbeddingClient()

        self.embedding_client = embedding_client

        self.chroma_client = chromadb.PersistentClient(
            path=self.persist_directory,
            settings=ChromaSettings(
                anonymized_telemetry=False,
                allow_reset=True,
            ),
        )

        self.vectorstore = Chroma(
            client=self.chroma_client,
            collection_name=self.collection_name,
            embedding_function=self.embedding_client,
        )

        logger.info(
            "ChunkChromaStore initialized",
            collection=self.collection_name,
            persist_dir=self.persist_directory,
            embedding_model=self.embedding_client.model_name,
        )

    def add_chunks(
        self,
        chunks: list[Document],
        job_id: str,
        subject_id: str,
    ) -> list[str]:
        """Add document chunks to ChromaDB.

        Args:
            chunks: List of LangChain Documents (from text chunker).
            job_id: Pipeline job ID (used for filtering at retrieval time).
            subject_id: Subject/topic ID.

        Returns:
            List of chunk IDs added.
        """
        if not chunks:
            return []

        ids = [f"{job_id}_chunk_{i}" for i in range(len(chunks))]
        texts = [c.page_content for c in chunks]
        metadatas = [
            {
                "job_id": job_id,
                "subject_id": subject_id,
                "chunk_index": c.metadata.get("chunk_index", i),
                "start_page": c.metadata.get("start_page", 0),
                "end_page": c.metadata.get("end_page", 0),
            }
            for i, c in enumerate(chunks)
        ]

        self.vectorstore.add_texts(texts=texts, ids=ids, metadatas=metadatas)
        logger.info(
            "Added {} chunks to ChromaDB",
            len(chunks),
            job_id=job_id,
            subject_id=subject_id,
        )
        return ids

    async def aretrieve(
        self,
        query: str,
        job_id: str,
        k: int = 3,
    ) -> list[Document]:
        """Async retrieval of relevant chunks for a query.

        Args:
            query: User's chat message.
            job_id: Filter to chunks from this pipeline job.
            k: Number of top chunks to return.

        Returns:
            List of top-k relevant Documents with page_content and metadata.
        """
        return await asyncio.to_thread(self._retrieve_sync, query, job_id, k)

    def _retrieve_sync(
        self,
        query: str,
        job_id: str,
        k: int = 3,
    ) -> list[Document]:
        """Synchronous retrieval (run in thread pool from async wrapper)."""
        try:
            results = self.vectorstore.similarity_search_with_score(
                query=query,
                k=k,
                filter={"job_id": job_id},
            )
            return [doc for doc, _score in results]
        except Exception:
            logger.exception("RAG retrieval failed")
            return []

