"""Dependencies for FastAPI dependency injection."""

from llm import get_llm
from llm.extract_chain import ExtractionChain
from embed.embedding_client import EmbeddingClient
from storage.chroma_store import ConceptChromaStore
from api.config import api_settings
from fastapi import Depends, HTTPException, status
from typing import Optional
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))


# Singleton instances
_extraction_chain: Optional[ExtractionChain] = None
_embedding_client: Optional[EmbeddingClient] = None
_chroma_store: Optional[ConceptChromaStore] = None


def get_extraction_chain() -> ExtractionChain:
    """
    Get or create extraction chain singleton.

    Returns:
        ExtractionChain instance
    """
    global _extraction_chain

    if _extraction_chain is None:
        try:
            llm = get_llm(
                temperature=0.1,
            )
            _extraction_chain = ExtractionChain(client=llm)
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to initialize extraction chain: {str(e)}"
            )

    return _extraction_chain


def get_embedding_client() -> EmbeddingClient:
    """
    Get or create embedding client singleton.

    Returns:
        EmbeddingClient instance
    """
    global _embedding_client

    if _embedding_client is None:
        try:
            _embedding_client = EmbeddingClient()
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to initialize embedding client: {str(e)}"
            )

    return _embedding_client


def get_chroma_store() -> ConceptChromaStore:
    """
    Get or create ChromaDB store singleton.

    Returns:
        ConceptChromaStore instance
    """
    global _chroma_store

    if _chroma_store is None:
        try:
            embedding_client = get_embedding_client()
            _chroma_store = ConceptChromaStore(
                collection_name=api_settings.chroma_collection_name,
                persist_directory=api_settings.chroma_persist_dir,
                embedding_client=embedding_client
            )
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to initialize ChromaDB store: {str(e)}"
            )

    return _chroma_store


def verify_components() -> dict:
    """
    Verify all components are initialized properly.

    Returns:
        Dictionary with component status
    """
    components = {}

    try:
        chain = get_extraction_chain()
        components["extraction_chain"] = "OK"
    except Exception as e:
        components["extraction_chain"] = f"ERROR: {str(e)}"

    try:
        embeddings = get_embedding_client()
        components["embedding_client"] = f"OK - {embeddings.model_name}"
    except Exception as e:
        components["embedding_client"] = f"ERROR: {str(e)}"

    try:
        store = get_chroma_store()
        components["chroma_store"] = "OK"
    except Exception as e:
        components["chroma_store"] = f"ERROR: {str(e)}"

    return components
