"""Embedding client compatible with LangChain Embeddings interface."""

import functools
from threading import Lock
from typing import cast

from langchain_core.embeddings import Embeddings
from loguru import logger
from pyvi import ViTokenizer
from sentence_transformers import SentenceTransformer
import torch

from api.config import settings


class _ModelHandle:
    def __init__(self, model: SentenceTransformer) -> None:
        self.model = model
        self.lock = Lock()


class EmbeddingClient(Embeddings):
    """
    Embedding client using SentenceTransformers, compatible with LangChain.

    Kế thừa từ langchain_core.embeddings.Embeddings để tích hợp trực tiếp
    với Langchain Chroma và các components khác của LangChain.
    """

    @staticmethod
    @functools.lru_cache(maxsize=4)
    def _load_model_handle(
        model_name: str,
        device: str,
        max_seq_length: int | None,
    ) -> _ModelHandle:
        model = SentenceTransformer(model_name, device=device)
        if max_seq_length:
            try:
                model.max_seq_length = max_seq_length
            except Exception as exc:  # pragma: no cover - defensive fallback
                logger.warning("Cannot set max_seq_length: {}", exc)
        return _ModelHandle(model)

    def __init__(
        self,
        model_name: str | None = None,
        *,
        use_vi_tokenizer: bool | None = None,
        batch_size: int | None = None,
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
            raise ValueError("Embedding model name must be specified in settings.")

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model_handle = self._load_model_handle(
            self.model_name,
            self.device,
            settings.max_seq_length,
        )
        self.model = self._model_handle.model

        self.use_vi_tokenizer = (
            use_vi_tokenizer if use_vi_tokenizer is not None else settings.use_vi_tokenizer
        )

        self.batch_size = batch_size or settings.embedding_batch_size

        logger.info(
            "EmbeddingClient initialized: model={}, device={}, batch_size={}, use_vi_tokenizer={}",
            self.model_name,
            self.device,
            self.batch_size,
            self.use_vi_tokenizer,
        )

    def _maybe_tokenize(self, text: str) -> str:
        """Áp dụng ViTokenizer nếu được bật."""
        if not text:
            return ""
        return ViTokenizer.tokenize(text) if self.use_vi_tokenizer else text

    def embed_query(self, text: str) -> list[float]:
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

        with self._model_handle.lock:
            emb = self.model.encode(
                text,
                convert_to_tensor=False,
                normalize_embeddings=True,
                batch_size=1,
            )
        return cast("list[float]", emb.astype(float).tolist())

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
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

        with self._model_handle.lock:
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

        with self._model_handle.lock:
            return self.model.encode(
                [self._maybe_tokenize(t or "") for t in texts],
                convert_to_tensor=False,
                normalize_embeddings=True,
                batch_size=self.batch_size,
                **kwargs,
            )
