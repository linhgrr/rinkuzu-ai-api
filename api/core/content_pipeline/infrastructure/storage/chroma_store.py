"""ChromaDB storage for concepts with LangChain integration."""

from pathlib import Path
from typing import Any

import chromadb
from chromadb.config import Settings as ChromaSettings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from loguru import logger

from api.core.content_pipeline.infrastructure.embed.embedding_client import EmbeddingClient
from api.core.content_pipeline.infrastructure.llm.schemas import Concept


class ConceptChromaStore:
    """ChromaDB store for concept knowledge base with LangChain."""

    def __init__(
        self,
        collection_name: str = "concepts",
        persist_directory: str | None = None,
        embedding_client: EmbeddingClient | None = None
    ):
        """
        Khởi tạo ChromaDB store với LangChain integration.

        Args:
            collection_name: Tên collection trong ChromaDB
            persist_directory: Thư mục lưu trữ dữ liệu ChromaDB (default: ./chroma_db)
            embedding_client: EmbeddingClient instance (nếu None, sẽ tạo mới)
        """
        self.collection_name = collection_name

        # Set default persist directory
        if persist_directory is None:
            persist_directory = str(
                Path(__file__).parent.parent.parent / "chroma_db")

        self.persist_directory = persist_directory
        Path(self.persist_directory).mkdir(parents=True, exist_ok=True)

        # Initialize embedding client
        if embedding_client is None:
            self.embedding_client = EmbeddingClient()
        else:
            self.embedding_client = embedding_client

        # Initialize ChromaDB client with persistent storage
        self.chroma_client = chromadb.PersistentClient(
            path=self.persist_directory,
            settings=ChromaSettings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )

        # Initialize LangChain Chroma vectorstore
        # Sử dụng trực tiếp EmbeddingClient vì nó đã implement Embeddings interface
        self.vectorstore = Chroma(
            client=self.chroma_client,
            collection_name=self.collection_name,
            embedding_function=self.embedding_client,
        )

        logger.info(
            "ConceptChromaStore initialized with LangChain Chroma",
            collection=self.collection_name,
            persist_dir=self.persist_directory,
            embedding_model=self.embedding_client.model_name
        )

    def add_concepts(
        self,
        concepts: list[Concept],
        subject_id: str
    ) -> list[str]:
        """
        Thêm concepts vào ChromaDB.

        Args:
            concepts: List các concepts cần thêm
            subject_id: Subject ID để filter

        Returns:
            List các document IDs đã được thêm
        """
        if not concepts:
            logger.warning("No concepts to add to ChromaDB")
            return []

        documents = []
        ids = []

        for concept in concepts:
            # Tạo document content cho semantic search
            # Kết hợp name, definition và examples để tăng chất lượng search
            content_parts = [f"Concept: {concept.name}"]

            if concept.definition:
                content_parts.append(f"Definition: {concept.definition}")

            if concept.examples:
                # Giới hạn 3 examples để tránh content quá dài
                examples_text = "; ".join(concept.examples[:3])
                content_parts.append(f"Examples: {examples_text}")

            page_content = "\n".join(content_parts)

            # Tạo metadata để filter và retrieve
            metadata = {
                "concept_id": concept.concept_id,
                "subject_id": subject_id,
                "name": concept.name,
                "definition": concept.definition or "",
                "num_examples": len(concept.examples) if concept.examples else 0,
                "num_relations": len(concept.relations) if concept.relations else 0,
            }

            # Thêm topic nếu có
            if hasattr(concept, "topic") and concept.topic:
                metadata["topic"] = concept.topic

            # Tạo LangChain Document
            doc = Document(
                page_content=page_content,
                metadata=metadata
            )

            documents.append(doc)
            ids.append(concept.concept_id)

        # Thêm vào vectorstore bằng LangChain Chroma
        try:
            added_ids = self.vectorstore.add_documents(
                documents=documents,
                ids=ids
            )

            logger.info(
                f"Added {len(added_ids)} concepts to ChromaDB via LangChain",
                collection=self.collection_name,
                subject_id=subject_id
            )

            return added_ids

        except Exception as e:
            logger.error(f"Error adding concepts to ChromaDB: {e}")
            raise

    def search_concepts(
        self,
        query: str,
        subject_id: str | None = None,
        k: int = 5,
        filter_dict: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        """
        Tìm kiếm concepts sử dụng semantic search.

        Args:
            query: Query text để search
            subject_id: Filter theo subject ID (optional)
            k: Số lượng kết quả trả về
            filter_dict: Các metadata filters bổ sung

        Returns:
            List các search results với concept data và similarity scores
        """
        # Build filter cho metadata
        where_filter = filter_dict or {}
        if subject_id:
            where_filter["subject_id"] = subject_id

        # Thực hiện similarity search với LangChain Chroma
        try:
            if where_filter:
                results = self.vectorstore.similarity_search_with_score(
                    query=query,
                    k=k,
                    filter=where_filter
                )
            else:
                results = self.vectorstore.similarity_search_with_score(
                    query=query,
                    k=k
                )

            # Format kết quả
            formatted_results = []
            for doc, score in results:
                result = {
                    "concept_id": doc.metadata.get("concept_id"),
                    "name": doc.metadata.get("name"),
                    "definition": doc.metadata.get("definition"),
                    "subject_id": doc.metadata.get("subject_id"),
                    "score": float(score),
                    "content": doc.page_content
                }

                # Thêm topic nếu có
                if "topic" in doc.metadata:
                    result["topic"] = doc.metadata["topic"]

                formatted_results.append(result)

            logger.debug(
                f"Semantic search returned {len(formatted_results)} results",
                query=query,
                subject_id=subject_id,
                k=k
            )

            return formatted_results

        except Exception as e:
            logger.error(f"Error searching concepts: {e}")
            raise

    def get_concept_by_id(
        self,
        concept_id: str
    ) -> dict[str, Any] | None:
        """
        Retrieve a concept by its ID.

        Args:
            concept_id: Concept ID to retrieve

        Returns:
            Concept data or None if not found
        """
        try:
            results = self.vectorstore.get(ids=[concept_id])

            if results and results["ids"]:
                # Return first result
                return {
                    "concept_id": results["ids"][0],
                    "metadata": results["metadatas"][0] if results["metadatas"] else {},
                    "content": results["documents"][0] if results["documents"] else ""
                }

            return None

        except Exception as e:
            logger.error(f"Error retrieving concept {concept_id}: {e}")
            return None

    def delete_by_subject(self, subject_id: str) -> int:
        """
        Delete all concepts for a subject.

        Args:
            subject_id: Subject ID to delete

        Returns:
            Number of concepts deleted
        """
        try:
            # Get all concept IDs for this subject
            collection = self.chroma_client.get_collection(
                self.collection_name)
            results = collection.get(where={"subject_id": subject_id})

            if results and results["ids"]:
                # Delete by IDs
                collection.delete(ids=results["ids"])
                deleted_count = len(results["ids"])

                logger.info(
                    f"Deleted {deleted_count} concepts for subject {subject_id}"
                )

                return deleted_count

            return 0

        except Exception as e:
            logger.error(
                f"Error deleting concepts for subject {subject_id}: {e}")
            raise

    def get_collection_stats(self) -> dict[str, Any]:
        """
        Get statistics about the collection.

        Returns:
            Dictionary with collection statistics
        """
        try:
            collection = self.chroma_client.get_collection(
                self.collection_name)
            count = collection.count()

            return {
                "collection_name": self.collection_name,
                "total_concepts": count,
                "persist_directory": self.persist_directory,
                "embedding_model": self.embedding_client.model_name
            }

        except Exception as e:
            logger.error(f"Error getting collection stats: {e}")
            return {
                "collection_name": self.collection_name,
                "error": str(e)
            }

    def reset_collection(self):
        """Xóa và tạo lại collection (sử dụng cẩn thận!)."""
        try:
            self.chroma_client.delete_collection(self.collection_name)
            logger.warning(f"Deleted collection: {self.collection_name}")

            # Recreate vectorstore với LangChain Chroma
            self.vectorstore = Chroma(
                client=self.chroma_client,
                collection_name=self.collection_name,
                embedding_function=self.embedding_client,
            )

            logger.info(f"Recreated collection: {self.collection_name}")

        except Exception as e:
            logger.error(f"Error resetting collection: {e}")
            raise

    def as_retriever(self, **kwargs):
        """
        Chuyển vectorstore thành retriever để sử dụng với LangChain chains.

        Args:
            **kwargs: Arguments được pass vào retriever (search_type, search_kwargs, etc.)

        Returns:
            VectorStoreRetriever instance

        Example:
            retriever = store.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 5, "filter": {"subject_id": "math"}}
            )
        """
        return self.vectorstore.as_retriever(**kwargs)
