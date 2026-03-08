"""Configuration settings for content processor."""

from pydantic import ConfigDict
from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    """Application settings."""

    model_config = ConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="allow"  
    )

    # Chunking settings
    chunk_size: int = 1500
    chunk_overlap: int = 200

    # Embedding settings
    embedding_model: Optional[str] = "huyydangg/DEk21_hcmute_embedding"
    max_seq_length: Optional[int] = 512
    embedding_batch_size: int = 32
    use_vi_tokenizer: bool = False

    # LLM settings
    gemini_api_keys: Optional[str] = None

    prs_threshold: float = 0.75  # Threshold for prerequisite ranking

# Global settings instance
settings = Settings()
