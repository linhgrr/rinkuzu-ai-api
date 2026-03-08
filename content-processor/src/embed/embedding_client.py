"""Embedding client compatible with LangChain Embeddings interface."""

from sentence_transformers import SentenceTransformer
import torch
from pyvi import ViTokenizer
from loguru import logger
from typing import List, Optional
from langchain_core.embeddings import Embeddings
from config import settings


class EmbeddingClient(Embeddings):
    """
    Embedding client using SentenceTransformers, compatible with LangChain.
    
    Kế thừa từ langchain_core.embeddings.Embeddings để tích hợp trực tiếp
    với Langchain Chroma và các components khác của LangChain.
    """
    
    def __init__(
        self,
        model_name: Optional[str] = None,
        use_vi_tokenizer: Optional[bool] = None,
        batch_size: Optional[int] = None,
    ):
        """
        Khởi tạo EmbeddingClient.
        
        Args:
            model_name: Tên model embedding từ HuggingFace hoặc local
            use_vi_tokenizer: Có sử dụng ViTokenizer cho tiếng Việt không
            batch_size: Kích thước batch cho embedding batch
        """
        self.model_name = model_name or settings.embedding_model
        if not self.model_name:
            raise ValueError(
                "Embedding model name must be specified in settings.")

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer(self.model_name, device=self.device)

        self.use_vi_tokenizer = use_vi_tokenizer if use_vi_tokenizer is not None else getattr(
            settings, "use_vi_tokenizer", False)

        self.batch_size = batch_size or getattr(
            settings, "embedding_batch_size", 32)

        if hasattr(settings, "max_seq_length") and settings.max_seq_length:
            try:
                self.model.max_seq_length = settings.max_seq_length
            except Exception as e:
                logger.warning(f"Cannot set max_seq_length: {e}")
        
        logger.info(
            f"EmbeddingClient initialized: model={self.model_name}, "
            f"device={self.device}, batch_size={self.batch_size}, "
            f"use_vi_tokenizer={self.use_vi_tokenizer}"
        )

    def _maybe_tokenize(self, text: str) -> str:
        """Áp dụng ViTokenizer nếu được bật."""
        if not text:
            return ""
        return ViTokenizer.tokenize(text) if self.use_vi_tokenizer else text

    def embed_query(self, text: str) -> List[float]:
        """
        Embed một query text đơn lẻ.
        
        Method này được LangChain sử dụng cho query embedding.
        
        Args:
            text: Text cần embed
            
        Returns:
            List[float]: Vector embedding
        """
        if not text:
            logger.warning("Empty text provided for embedding computation.")
            return []
        
        text = self._maybe_tokenize(text)

        emb = self.model.encode(
            text,
            convert_to_tensor=False,
            normalize_embeddings=True,
            batch_size=1,
        )
        return emb.astype(float).tolist()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embed nhiều documents.
        
        Method này được LangChain sử dụng cho document embedding.
        
        Args:
            texts: List các texts cần embed
            
        Returns:
            List[List[float]]: List các vector embeddings
        """
        if not texts:
            return []
        
        prepped = [self._maybe_tokenize(t or "") for t in texts]

        embs = self.model.encode(
            prepped,
            convert_to_tensor=False,
            normalize_embeddings=True,
            batch_size=self.batch_size,
        )
        return [row.astype(float).tolist() for row in embs]
    
    # Backward compatibility methods (cho các phần khác của dự án)
    def encode(self, texts, **kwargs):
        """
        Backward compatibility method.
        
        Một số phần của dự án có thể gọi trực tiếp .encode()
        """
        if isinstance(texts, str):
            texts = [texts]
        
        return self.model.encode(
            [self._maybe_tokenize(t or "") for t in texts],
            convert_to_tensor=False,
            normalize_embeddings=True,
            batch_size=self.batch_size,
            **kwargs
        )
