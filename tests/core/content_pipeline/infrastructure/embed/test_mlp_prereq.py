"""Unit tests for MLPPrerequisiteRanker.

Mocks sentence-transformers + torch.load so CI never downloads BAAI/bge-m3
or touches the actual MLP weights file.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import pytest
import torch

from api.domains.content_pipeline.infrastructure.embed.mlp_prereq import (
    MLPPrerequisiteRanker,
)
from api.domains.content_pipeline.infrastructure.embed.mlp_prereq import ranker as ranker_mod

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture(autouse=True)
def _reset_singleton():
    MLPPrerequisiteRanker.reset()
    yield
    MLPPrerequisiteRanker.reset()


def _build_concept(concept_id: str, name: str):
    return SimpleNamespace(concept_id=concept_id, name=name, definition=f"{name} definition")


def _make_loaded_ranker(monkeypatch, weights_path: Path, prob_for_pair):
    """Build a ranker with mocked encoder, tokenizer, model, and weight loader."""
    weights_path.write_bytes(b"")  # placeholder so existence check passes
    weights_path.with_name(f"{weights_path.name[:-4]}.metadata.json").write_text(
        """
{
  "dataset": "ViMath-Prereq-Module1-v1",
  "encoder": "BAAI/bge-m3",
  "input_text_mode": "name+definition",
  "pair_mode": "concat_rich",
  "embedding_dim": 1024,
  "input_dim": 4096,
  "threshold": 0.235
}
""".strip(),
        encoding="utf-8",
    )

    fake_encoder = MagicMock(name="encoder")
    fake_encoder.eval.return_value = fake_encoder
    fake_encoder.to.return_value = fake_encoder

    monkeypatch.setattr(
        "sentence_transformers.SentenceTransformer",
        lambda *_a, **_kw: fake_encoder,
    )

    fake_mlp = MagicMock(name="mlp")
    fake_mlp.eval.return_value = fake_mlp
    fake_mlp.to.return_value = fake_mlp

    def _forward(features):
        assert features.shape[-1] == 4096
        # Return logits whose sigmoid hits prob_for_pair for each row.
        target = torch.tensor(prob_for_pair, dtype=torch.float32)
        # logit such that sigmoid(logit) ≈ target
        eps = 1e-6
        target_clamped = target.clamp(eps, 1 - eps)
        return torch.log(target_clamped / (1 - target_clamped)).unsqueeze(-1)

    fake_mlp.side_effect = _forward
    monkeypatch.setattr(ranker_mod, "PrerequisiteClassifier", lambda **_kw: fake_mlp)

    monkeypatch.setattr(torch, "load", lambda *_a, **_kw: {"model_state_dict": {}})

    # Patch _encode to bypass the SentenceTransformer pipeline (BGE-M3 = 1024-d)
    def _fake_encode(self, texts):
        return torch.eye(len(texts), 1024)

    monkeypatch.setattr(ranker_mod.MLPPrerequisiteRanker, "_encode", _fake_encode)

    return MLPPrerequisiteRanker.load(weights_path)


def test_rank_empty_concepts_returns_empty(tmp_path: Path, monkeypatch):
    ranker = _make_loaded_ranker(monkeypatch, tmp_path / "w.pth", prob_for_pair=[0.0])
    assert ranker.rank([], threshold=0.5) == []


def test_rank_single_concept_returns_empty(tmp_path: Path, monkeypatch):
    ranker = _make_loaded_ranker(monkeypatch, tmp_path / "w.pth", prob_for_pair=[0.0])
    one = [_build_concept("c1", "Linear Algebra")]
    assert ranker.rank(one, threshold=0.5) == []


def test_rank_returns_pairs_above_threshold(tmp_path: Path, monkeypatch):
    # 3 concepts → 3*2 = 6 ordered pairs.
    # Make pairs at index 0 and 3 above threshold, others below.
    probs = [0.9, 0.1, 0.2, 0.8, 0.2, 0.1]
    ranker = _make_loaded_ranker(monkeypatch, tmp_path / "w.pth", prob_for_pair=probs)

    concepts = [
        _build_concept("c1", "Linear Algebra"),
        _build_concept("c2", "Matrix Multiplication"),
        _build_concept("c3", "Eigenvalues"),
    ]
    pairs = ranker.rank(concepts, threshold=0.5)

    # Pair ordering deterministic: (i,j) for i in 0..N, j in 0..N, i!=j
    # idx 0 → (c1,c2), idx 3 → (c2,c3)
    first_pair = next(pair for pair in pairs if pair.source_id == "c1" and pair.target_id == "c2")
    assert first_pair.sources == frozenset({"mlp"})
    assert first_pair.ranker_score == pytest.approx(0.9)
    assert any(pair.source_id == "c2" and pair.target_id == "c3" for pair in pairs)
    assert len(pairs) == 2


def test_rank_uses_metadata_threshold_when_threshold_is_none(tmp_path: Path, monkeypatch):
    probs = [0.3, 0.2]
    ranker = _make_loaded_ranker(monkeypatch, tmp_path / "w.pth", prob_for_pair=probs)
    concepts = [_build_concept("c1", "A"), _build_concept("c2", "B")]

    pairs = ranker.rank(concepts, threshold=None)

    assert [(pair.source_id, pair.target_id) for pair in pairs] == [("c1", "c2")]


def test_rank_threshold_above_all_probs_returns_empty(tmp_path: Path, monkeypatch):
    probs = [0.4, 0.3, 0.2, 0.1, 0.0, 0.4]
    ranker = _make_loaded_ranker(monkeypatch, tmp_path / "w.pth", prob_for_pair=probs)
    concepts = [
        _build_concept("c1", "A"),
        _build_concept("c2", "B"),
        _build_concept("c3", "C"),
    ]
    assert ranker.rank(concepts, threshold=0.5) == []


def test_load_is_singleton(tmp_path: Path, monkeypatch):
    weights = tmp_path / "w.pth"
    first = _make_loaded_ranker(monkeypatch, weights, prob_for_pair=[0.0])
    second = MLPPrerequisiteRanker.load(weights)
    assert first is second


def test_load_missing_weights_raises(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(
        "transformers.AutoTokenizer.from_pretrained",
        lambda *_a, **_kw: MagicMock(),
    )
    monkeypatch.setattr(
        "transformers.AutoModel.from_pretrained",
        lambda *_a, **_kw: MagicMock(),
    )
    with pytest.raises(FileNotFoundError):
        MLPPrerequisiteRanker.load(tmp_path / "does_not_exist.pth")


def test_load_invalid_metadata_raises(tmp_path: Path, monkeypatch):
    weights_path = tmp_path / "w.pth"
    weights_path.write_bytes(b"")
    weights_path.with_name("w.metadata.json").write_text(
        """
{
  "dataset": "LectureBank",
  "encoder": "BAAI/bge-m3",
  "input_text_mode": "name",
  "pair_mode": "concat",
  "embedding_dim": 1024,
  "input_dim": 2048,
  "threshold": 0.5
}
""".strip(),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Invalid prerequisite model metadata"):
        MLPPrerequisiteRanker.load(weights_path)
