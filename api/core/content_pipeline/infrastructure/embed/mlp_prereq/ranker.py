"""MLPPrerequisiteRanker — drop-in replacement for the legacy PRS ranker.

Encodes concept names with BAAI/bge-m3 (SentenceTransformer, L2-normalized),
runs the PrerequisiteClassifier on every ordered pair, and returns pairs whose
predicted probability is above the threshold.

The encoding path mirrors module1_update training EXACTLY: the MLP was trained
on SentenceTransformer("BAAI/bge-m3").encode(..., normalize_embeddings=True)
embeddings, so inference must use the same encoder + normalization. Using a
different pooling/normalization would shift the embedding distribution and the
trained decision boundary would silently misfire.

Singleton: encoder + MLP weights load once per process (lazy on first
``rank()`` call). CPU-only by default — works on servers without a GPU.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, ClassVar

import torch
from torch import Tensor

from .model import PrerequisiteClassifier

logger = logging.getLogger(__name__)

ENCODER_NAME = "BAAI/bge-m3"
EMBED_DIM = 1024  # BGE-M3 hidden size (XLM-RoBERTa-large backbone)
ENCODE_BATCH = 16
PREDICT_BATCH = 1024
MIN_CONCEPT_COUNT = 2


def _concept_to_text(concept: Any) -> str:
    """Build the text encoded by BGE-M3 for a concept.

    The MLP was trained on LectureBank with name-only embeddings; we keep that
    contract at inference time. Empirically, concatenating the definition shifts
    the embedding distribution far enough that the trained decision boundary
    underfits — so we deliberately ignore the definition here.
    """
    name = (getattr(concept, "name", "") or "").strip()
    if name:
        return name
    return (getattr(concept, "definition", "") or "").strip()


class MLPPrerequisiteRanker:
    """Singleton ranker. Use ``MLPPrerequisiteRanker.load(weights_path)``."""

    _instance: ClassVar[MLPPrerequisiteRanker | None] = None

    def __init__(self, weights_path: Path, device: str = "cpu") -> None:
        self._weights_path = Path(weights_path)
        self._device = torch.device(device)
        self._encoder: Any | None = None
        self._model: PrerequisiteClassifier | None = None

    @classmethod
    def load(cls, weights_path: Path, device: str = "cpu") -> MLPPrerequisiteRanker:
        if cls._instance is None:
            inst = cls(weights_path, device=device)
            inst._load_components()
            cls._instance = inst
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Drop the cached singleton. Intended for tests."""
        cls._instance = None

    def _load_components(self) -> None:
        from sentence_transformers import SentenceTransformer

        if not self._weights_path.exists():
            raise FileNotFoundError(
                f"MLP weights not found at {self._weights_path}. "
                f"Expected file copied from bc_with_code/results/results/best_model.pth."
            )

        logger.info("Loading BGE-M3 SentenceTransformer encoder for MLP ranker.")
        encoder = SentenceTransformer(ENCODER_NAME, device=str(self._device))
        self._encoder = encoder

        logger.info("Loading MLP weights from %s", self._weights_path)
        ckpt = torch.load(
            self._weights_path,
            map_location=self._device,
            weights_only=False,
        )
        state = ckpt.get("model_state_dict", ckpt)

        model = PrerequisiteClassifier(embedding_dim=EMBED_DIM, hidden_dim=512, dropout=0.3)
        model.load_state_dict(state)
        model.eval()
        model.to(self._device)
        self._model = model

    def _ensure_loaded(self) -> None:
        if self._model is None or self._encoder is None:
            self._load_components()

    def _encode(self, texts: list[str]) -> Tensor:
        assert self._encoder is not None
        # Mirror training: SentenceTransformer.encode with L2-normalization.
        vecs = self._encoder.encode(
            texts,
            batch_size=ENCODE_BATCH,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return torch.as_tensor(vecs).float()

    def rank(
        self,
        concepts: list[Any],
        threshold: float,
    ) -> list[tuple[str, str]]:
        """Return ordered (concept_id_a, concept_id_b) pairs with p(A→B) ≥ threshold."""
        if not concepts or len(concepts) < MIN_CONCEPT_COUNT:
            return []

        self._ensure_loaded()
        assert self._model is not None

        texts = [_concept_to_text(c) for c in concepts]
        ids = [c.concept_id for c in concepts]

        embeddings = self._encode(texts)

        n = len(concepts)
        idx_a: list[int] = []
        idx_b: list[int] = []
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                idx_a.append(i)
                idx_b.append(j)

        if not idx_a:
            return []

        idx_a_t = torch.tensor(idx_a, dtype=torch.long)
        idx_b_t = torch.tensor(idx_b, dtype=torch.long)
        emb_a = embeddings.index_select(0, idx_a_t)
        emb_b = embeddings.index_select(0, idx_b_t)

        probs = torch.empty(len(idx_a), dtype=torch.float32)
        with torch.no_grad():
            for start in range(0, len(idx_a), PREDICT_BATCH):
                end = start + PREDICT_BATCH
                logits = self._model(
                    emb_a[start:end].to(self._device),
                    emb_b[start:end].to(self._device),
                ).squeeze(-1)
                probs[start:end] = torch.sigmoid(logits).cpu()

        keep = probs >= threshold
        return [
            (ids[idx_a[k]], ids[idx_b[k]])
            for k in torch.nonzero(keep, as_tuple=False).flatten().tolist()
        ]
