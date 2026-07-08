"""pca.py — Apply the training-time PCA transform to concept embeddings.

The DQN policy was trained with per-concept observation features that include a
16-dim PCA reduction of the 768-dim SentenceTransformer concept embeddings
(``paraphrase-multilingual-mpnet-base-v2``). The PCA was fit ONCE on the Junyi
concept embeddings at train time; serving must reuse that exact transform so the
observation distribution matches what the frozen policy expects.

The fitted transform (``components_`` + ``mean_``) is persisted next to the model
checkpoints as ``concept_pca16.npz``. Applying it is a linear projection followed
by per-row L2 normalization — mirroring ``saint_rl/generate_pca_embeddings.py``.

# ponytail: linear PCA transform inline, no sklearn dependency at serving.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from loguru import logger
import numpy as np

from api.config import settings

_EXPECTED_NDIM = 2


def _default_pca_path() -> Path:
    # Sits next to saint_best.pt / dqn_best.pt (settings.saint_path -> models/).
    return Path(settings.saint_path).with_name("concept_pca16.npz")


@lru_cache(maxsize=1)
def _load_transform() -> tuple[np.ndarray, np.ndarray]:
    """Load (components (D, 768), mean (768,)) once per process."""
    path = _default_pca_path()
    if not path.exists():
        raise FileNotFoundError(
            f"PCA transform not found at {path}. Expected concept_pca16.npz "
            "(fit on Junyi concept embeddings) alongside the model checkpoints."
        )
    data = np.load(path)
    components = data["components"].astype(np.float32)  # (pca_dim, 768)
    mean = data["mean"].astype(np.float32)  # (768,)
    logger.info(
        "[PCA] Loaded concept PCA transform from {} (components {}, mean {})",
        path,
        components.shape,
        mean.shape,
    )
    return components, mean


def apply_concept_pca(embeddings: np.ndarray) -> np.ndarray:
    """Project (N, 768) concept embeddings to (N, pca_dim), L2-normalized per row.

    Matches training: ``(emb - mean_) @ components_.T`` then per-row L2 norm.
    """
    emb = np.asarray(embeddings, dtype=np.float32)
    if emb.ndim != _EXPECTED_NDIM:
        raise ValueError(f"Expected 2D embeddings, got shape {emb.shape}")

    components, mean = _load_transform()
    if emb.shape[1] != mean.shape[0]:
        raise ValueError(f"Embedding dim {emb.shape[1]} != PCA input dim {mean.shape[0]}.")

    projected = (emb - mean) @ components.T  # (N, pca_dim)
    norms = np.maximum(np.linalg.norm(projected, axis=1, keepdims=True), 1e-8)
    normalized: np.ndarray = (projected / norms).astype(np.float32)
    return normalized


def pca_dim() -> int:
    """Return the PCA output dimensionality of the persisted transform."""
    components, _ = _load_transform()
    return int(components.shape[0])
