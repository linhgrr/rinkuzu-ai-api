"""
chunk_chroma_store.py — ChromaDB storage for document chunks (RAG source).
"""

import asyncio
from pathlib import Path
from typing import Any

from langchain_core.documents import Document
from loguru import logger

from ._base import init_chroma_store

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

        self.chroma_client, self.vectorstore, self.embedding_client = init_chroma_store(
            collection_name=self.collection_name,
            persist_directory=self.persist_directory,
            embedding_client=embedding_client,
            log_label="ChunkChromaStore",
        )

    def add_chunks(
        self,
        chunks: list[Document],
        job_id: str,
        subject_id: str,
    ) -> list[str]:
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

    def delete_by_job(self, job_id: str) -> int:
        try:
            collection = self.chroma_client.get_collection(self.collection_name)
            results = collection.get(where={"job_id": job_id})
            if results and results.get("ids"):
                collection.delete(ids=results["ids"])
                deleted_count = len(results["ids"])
                logger.info("Deleted {} chunk vectors for job {}", deleted_count, job_id)
                return deleted_count
        except Exception:
            logger.exception("Error deleting chunks for job {}", job_id)
            raise
        return 0

    def replace_chunks(self, chunks: list[Document], job_id: str, subject_id: str) -> list[str]:
        self.delete_by_job(job_id)
        return self.add_chunks(chunks, job_id, subject_id)

    async def aretrieve(
        self,
        query: str,
        job_id: str,
        k: int = 3,
    ) -> list[Document]:
        return await asyncio.to_thread(self._retrieve_sync, query, job_id, k)

    def _retrieve_sync(
        self,
        query: str,
        job_id: str,
        k: int = 3,
    ) -> list[Document]:
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

    def reset_collection(self) -> None:
        try:
            self.chroma_client.delete_collection(self.collection_name)
        except Exception:
            logger.exception("Error resetting chunk collection")
            raise
        self.chroma_client, self.vectorstore, self.embedding_client = init_chroma_store(
            collection_name=self.collection_name,
            persist_directory=self.persist_directory,
            embedding_client=self.embedding_client,
            log_label="ChunkChromaStore",
        )
