"""ChromaDB storage for concepts with LangChain integration."""

from pathlib import Path
from typing import Any

from langchain_chroma import Chroma
from langchain_core.documents import Document
from loguru import logger

from api.domains.content_pipeline.infrastructure.embed.embedding_client import EmbeddingClient
from api.domains.content_pipeline.infrastructure.llm.schemas import Concept

from ._base import init_chroma_store


class ConceptChromaStore:
    """ChromaDB store for concept knowledge base with LangChain."""

    def __init__(
        self,
        collection_name: str = "concepts",
        persist_directory: str | None = None,
        embedding_client: EmbeddingClient | None = None,
    ) -> None:
        self.collection_name = collection_name

        if persist_directory is None:
            persist_directory = str(Path(__file__).parent.parent.parent / "chroma_db")

        self.persist_directory = persist_directory

        self.chroma_client, self.vectorstore, self.embedding_client = init_chroma_store(
            collection_name=self.collection_name,
            persist_directory=self.persist_directory,
            embedding_client=embedding_client,
            log_label="ConceptChromaStore",
        )

    def add_concepts(self, concepts: list[Concept], subject_id: str) -> list[str]:
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
            doc = Document(page_content=page_content, metadata=metadata)

            documents.append(doc)
            ids.append(concept.concept_id)

        # Thêm vào vectorstore bằng LangChain Chroma
        try:
            added_ids: list[str] = self.vectorstore.add_documents(documents=documents, ids=ids)

            logger.info(
                "Added {} concepts to ChromaDB via LangChain",
                len(added_ids),
                collection=self.collection_name,
                subject_id=subject_id,
            )

        except Exception:
            logger.exception("Error adding concepts to ChromaDB")
            raise
        else:
            return added_ids

    def search_concepts(
        self,
        query: str,
        subject_id: str | None = None,
        k: int = 5,
        filter_dict: dict[str, Any] | None = None,
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
                    query=query, k=k, filter=where_filter
                )
            else:
                results = self.vectorstore.similarity_search_with_score(query=query, k=k)

            # Format kết quả
            formatted_results = []
            for doc, score in results:
                result = {
                    "concept_id": doc.metadata.get("concept_id"),
                    "name": doc.metadata.get("name"),
                    "definition": doc.metadata.get("definition"),
                    "subject_id": doc.metadata.get("subject_id"),
                    "score": float(score),
                    "content": doc.page_content,
                }

                # Thêm topic nếu có
                if "topic" in doc.metadata:
                    result["topic"] = doc.metadata["topic"]

                formatted_results.append(result)

            logger.debug(
                "Semantic search returned {} results",
                len(formatted_results),
                query=query,
                subject_id=subject_id,
                k=k,
            )

        except Exception:
            logger.exception("Error searching concepts")
            raise
        else:
            return formatted_results

    def get_concept_by_id(self, concept_id: str) -> dict[str, Any] | None:
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
                    "content": results["documents"][0] if results["documents"] else "",
                }

        except Exception:
            logger.exception("Error retrieving concept {}", concept_id)
            return None
        else:
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
            collection = self.chroma_client.get_collection(self.collection_name)
            results = collection.get(where={"subject_id": subject_id})

            if results and results["ids"]:
                collection.delete(ids=results["ids"])
                deleted_count = len(results["ids"])
                logger.info("Deleted {} concepts for subject {}", deleted_count, subject_id)
            else:
                deleted_count = 0

        except Exception:
            logger.exception("Error deleting concepts for subject {}", subject_id)
            raise
        else:
            return deleted_count

    def get_collection_stats(self) -> dict[str, Any]:
        """
        Get statistics about the collection.

        Returns:
            Dictionary with collection statistics
        """
        try:
            collection = self.chroma_client.get_collection(self.collection_name)
            count = collection.count()
            stats = {
                "collection_name": self.collection_name,
                "total_concepts": count,
                "persist_directory": self.persist_directory,
                "embedding_model": self.embedding_client.model_name,
            }

        except Exception as e:
            logger.exception("Error getting collection stats")
            return {"collection_name": self.collection_name, "error": str(e)}
        else:
            return stats

    def reset_collection(self) -> Any:
        """Xóa và tạo lại collection (sử dụng cẩn thận!)."""
        try:
            self.chroma_client.delete_collection(self.collection_name)
            logger.warning("Deleted collection: {}", self.collection_name)

            # Recreate vectorstore với LangChain Chroma
            self.vectorstore = Chroma(
                client=self.chroma_client,
                collection_name=self.collection_name,
                embedding_function=self.embedding_client,
            )

            logger.info("Recreated collection: {}", self.collection_name)

        except Exception:
            logger.exception("Error resetting collection")
            raise
