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

from dataclasses import dataclass
import json
import logging
from pathlib import Path
from typing import Any, ClassVar

import torch
from torch import Tensor

from api.core.content_pipeline.domain.relations import RelationCandidate

from .model import PrerequisiteClassifier

logger = logging.getLogger(__name__)

ENCODER_NAME = "BAAI/bge-m3"
EMBED_DIM = 1024  # BGE-M3 hidden size (XLM-RoBERTa-large backbone)
ENCODE_BATCH = 16
PREDICT_BATCH = 1024
MIN_CONCEPT_COUNT = 2
EXPECTED_INPUT_TEXT_MODE = "name+definition"
EXPECTED_PAIR_MODE = "concat_rich"
EXPECTED_DATASET_PREFIX = "ViMath"


@dataclass(frozen=True, slots=True)
class PrerequisiteModelMetadata:
    """Inference contract saved next to the ViMath prerequisite checkpoint."""

    dataset: str
    encoder: str
    input_text_mode: str
    pair_mode: str
    embedding_dim: int
    input_dim: int
    threshold: float

    @classmethod
    def from_path(cls, metadata_path: Path) -> PrerequisiteModelMetadata:
        if not metadata_path.exists():
            raise FileNotFoundError(f"Prerequisite metadata not found at {metadata_path}")
        with metadata_path.open("r", encoding="utf-8") as file_obj:
            payload = json.load(file_obj)
        metadata = cls(
            dataset=str(payload.get("dataset", "")),
            encoder=str(payload.get("encoder", "")),
            input_text_mode=str(payload.get("input_text_mode", "")),
            pair_mode=str(payload.get("pair_mode", "")),
            embedding_dim=int(payload.get("embedding_dim", 0)),
            input_dim=int(payload.get("input_dim", 0)),
            threshold=float(payload.get("threshold", 0.0)),
        )
        metadata.validate()
        return metadata

    def validate(self) -> None:
        errors: list[str] = []
        if not self.dataset.startswith(EXPECTED_DATASET_PREFIX):
            errors.append(f"dataset={self.dataset!r}")
        if self.encoder != ENCODER_NAME:
            errors.append(f"encoder={self.encoder!r}")
        if self.input_text_mode != EXPECTED_INPUT_TEXT_MODE:
            errors.append(f"input_text_mode={self.input_text_mode!r}")
        if self.pair_mode != EXPECTED_PAIR_MODE:
            errors.append(f"pair_mode={self.pair_mode!r}")
        if self.embedding_dim != EMBED_DIM:
            errors.append(f"embedding_dim={self.embedding_dim!r}")
        if self.input_dim != EMBED_DIM * 4:
            errors.append(f"input_dim={self.input_dim!r}")
        if not 0.0 <= self.threshold <= 1.0:
            errors.append(f"threshold={self.threshold!r}")
        if errors:
            raise ValueError(
                "Invalid prerequisite model metadata for ViMath inference: " + ", ".join(errors)
            )


def _default_metadata_path(weights_path: Path) -> Path:
    if weights_path.name.endswith(".pth"):
        return weights_path.with_name(f"{weights_path.name[:-4]}.metadata.json")
    return weights_path.with_suffix(weights_path.suffix + ".metadata.json")


def _concept_to_text(concept: Any) -> str:
    """Build the text encoded by BGE-M3 for a concept.

    The ViMath checkpoint was trained with name+definition text. Keeping this
    exact input shape matters because the MLP threshold is calibrated on it.
    """
    name = (getattr(concept, "name", "") or "").strip()
    definition = (getattr(concept, "definition", "") or "").strip()
    if name and definition:
        return f"{name}: {definition}"
    return name or definition


def _build_pair_features(emb_a: Tensor, emb_b: Tensor) -> Tensor:
    return torch.cat([emb_a, emb_b, torch.abs(emb_a - emb_b), emb_a * emb_b], dim=-1)


class MLPPrerequisiteRanker:
    """Singleton ranker. Use ``MLPPrerequisiteRanker.load(weights_path)``."""

    _instance: ClassVar[MLPPrerequisiteRanker | None] = None

    def __init__(self, weights_path: Path, device: str = "cpu") -> None:
        self._weights_path = Path(weights_path)
        self._device = torch.device(device)
        self._encoder: Any | None = None
        self._model: PrerequisiteClassifier | None = None
        self._metadata: PrerequisiteModelMetadata | None = None

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
                "Expected the ViMath prerequisite checkpoint in api/models."
            )

        metadata = PrerequisiteModelMetadata.from_path(_default_metadata_path(self._weights_path))
        self._metadata = metadata

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

        model = PrerequisiteClassifier(
            input_dim=metadata.input_dim,
            hidden_dim=512,
            dropout=0.3,
        )
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
        threshold: float | None,
    ) -> list[RelationCandidate]:
        """Return ordered (concept_id_a, concept_id_b) pairs with p(A→B) ≥ threshold."""
        if not concepts or len(concepts) < MIN_CONCEPT_COUNT:
            return []

        self._ensure_loaded()
        assert self._model is not None
        assert self._metadata is not None
        cutoff = self._metadata.threshold if threshold is None else threshold

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
                features = _build_pair_features(
                    emb_a[start:end].to(self._device),
                    emb_b[start:end].to(self._device),
                )
                logits = self._model(features).squeeze(-1)
                probs[start:end] = torch.sigmoid(logits).cpu()

        keep = probs >= cutoff
        return [
            RelationCandidate(
                source_id=ids[idx_a[k]],
                target_id=ids[idx_b[k]],
                sources=frozenset({"mlp"}),
                ranker_score=float(probs[k].item()),
            )
            for k in torch.nonzero(keep, as_tuple=False).flatten().tolist()
        ]
