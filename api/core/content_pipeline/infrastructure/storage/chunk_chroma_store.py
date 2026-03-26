"""
chunk_chroma_store.py — ChromaDB storage for document chunks (RAG source).
"""

from typing import Any, Dict, List, Optional

from langchain_core.documents import Document
from langchain_chroma import Chroma
import chromadb
from chromadb.config import Settings as ChromaSettings
from loguru import logger

from .chroma_store import ConceptChromaStore


class ChunkChromaStore:
    """ChromaDB store for document chunks, used as RAG retrieval source for tutor chat."""

    def __init__(
        self,
        collection_name: str = "document_chunks",
        persist_directory: Optional[str] = None,
        embedding_client: Optional[Any] = None,
    ):
        if persist_directory is None:
            from pathlib import Path
            persist_directory = str(
                Path(__file__).parent.parent.parent.parent / "chroma_db"
            )

        self.persist_directory = persist_directory
        self.collection_name = collection_name

        import os
        os.makedirs(self.persist_directory, exist_ok=True)

        if embedding_client is None:
            from ..embed.embedding_client import EmbeddingClient
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
        chunks: List[Document],
        job_id: str,
        subject_id: str,
    ) -> List[str]:
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
            f"Added {len(chunks)} chunks to ChromaDB",
            job_id=job_id,
            subject_id=subject_id,
        )
        return ids

    async def aretrieve(
        self,
        query: str,
        job_id: str,
        k: int = 3,
    ) -> List[Document]:
        """Async retrieval of relevant chunks for a query.

        Args:
            query: User's chat message.
            job_id: Filter to chunks from this pipeline job.
            k: Number of top chunks to return.

        Returns:
            List of top-k relevant Documents with page_content and metadata.
        """
        import asyncio
        return await asyncio.to_thread(self._retrieve_sync, query, job_id, k)

    def _retrieve_sync(
        self,
        query: str,
        job_id: str,
        k: int = 3,
    ) -> List[Document]:
        """Synchronous retrieval (run in thread pool from async wrapper)."""
        try:
            results = self.vectorstore.similarity_search_with_score(
                query=query,
                k=k,
                filter={"job_id": job_id},
            )
            docs = [doc for doc, _score in results]
            logger.debug(
                f"RAG retrieval returned {len(docs)} chunks for query",
                job_id=job_id,
                k=k,
            )
            return docs
        except Exception as exc:
            logger.warning(f"RAG retrieval failed: {exc}")
            return []

    def as_retriever(self, **kwargs):
        """Expose as LangChain retriever for future chain usage."""
        return self.vectorstore.as_retriever(**kwargs)
