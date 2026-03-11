"""Configuration for API."""

from pydantic_settings import BaseSettings
from typing import Optional


class APISettings(BaseSettings):
    """API configuration settings."""

    # API settings
    api_title: str = "Knowledge Graph Builder API"
    api_version: str = "1.0.0"
    api_description: str = "API for building knowledge graphs from PDF documents"

    # Server settings
    host: str = "0.0.0.0"
    port: int = 8000
    reload: bool = True

    # CORS
    cors_origins: list = ["*"]

    # File upload
    max_upload_size: int = 50 * 1024 * 1024  # 50MB
    allowed_extensions: list = [".pdf"]
    upload_dir: str = "./uploads"

    # Processing settings
    default_chunk_size: int = 1500
    default_chunk_overlap: int = 200
    default_batch_size: int = 5
    default_max_workers: int = 8
    default_prs_threshold: float = 0.6
    default_min_confidence: float = 0.5
    default_similarity_threshold: float = 0.9
    # How many previously-extracted concept names to include in each batch prompt
    # to help the LLM avoid duplicates and understand cross-batch relations.
    default_max_previous_concepts: int = 50

    # ChromaDB settings
    chroma_persist_dir: str = "./chroma_db"
    chroma_collection_name: str = "concepts"

    # Job settings
    max_concurrent_jobs: int = 3
    job_timeout_seconds: int = 3600  # 1 hour

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "allow"  # Allow extra fields from .env


api_settings = APISettings()
