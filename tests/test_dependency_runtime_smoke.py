"""Tests for production dependency/runtime smoke guard."""

from __future__ import annotations

from pathlib import Path

import pytest

from scripts import dependency_runtime_smoke as smoke


def test_check_banned_distributions_empty_on_clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    class _FakeDist:
        def __init__(self, name: str) -> None:
            self.metadata = {"Name": name}

    monkeypatch.setattr(
        smoke.importlib.metadata,
        "distributions",
        lambda: [_FakeDist("torch"), _FakeDist("numpy"), _FakeDist("setuptools")],
    )
    assert smoke.check_banned_distributions() == []


def test_check_banned_distributions_flags_nvidia_and_triton(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _FakeDist:
        def __init__(self, name: str) -> None:
            self.metadata = {"Name": name}

    monkeypatch.setattr(
        smoke.importlib.metadata,
        "distributions",
        lambda: [
            _FakeDist("torch"),
            _FakeDist("nvidia-cublas-cu12"),
            _FakeDist("triton"),
            _FakeDist("cuda-toolkit"),
            _FakeDist("nvidia-ml-py"),  # monitoring-only; must not be banned
        ],
    )
    found = smoke.check_banned_distributions()
    assert "nvidia-cublas-cu12" in found
    assert "triton" in found
    assert "cuda-toolkit" in found
    assert "nvidia-ml-py" not in found


def test_check_torch_cpu_rejects_cuda_tag(monkeypatch: pytest.MonkeyPatch) -> None:
    class _Version:
        cuda = "12.4"

    class _Cuda:
        @staticmethod
        def is_available() -> bool:
            return False

    class _FakeTorch:
        __version__ = "2.13.0+cu124"
        version = _Version()
        cuda = _Cuda()

    import sys

    monkeypatch.setitem(sys.modules, "torch", _FakeTorch)
    errors = smoke.check_torch_cpu()
    assert errors
    assert any("CUDA" in e or "cpu" in e.lower() for e in errors)


def test_check_torch_cpu_accepts_cpu_tag(monkeypatch: pytest.MonkeyPatch) -> None:
    class _Version:
        cuda = None

    class _Cuda:
        @staticmethod
        def is_available() -> bool:
            return False

    class _FakeTorch:
        __version__ = "2.13.0+cpu"
        version = _Version()
        cuda = _Cuda()

    import sys

    monkeypatch.setitem(sys.modules, "torch", _FakeTorch)
    assert smoke.check_torch_cpu() == []


def test_check_torch_cpu_rejects_stale_cpu_version(monkeypatch: pytest.MonkeyPatch) -> None:
    class _Version:
        cuda = None

    class _Cuda:
        @staticmethod
        def is_available() -> bool:
            return False

    class _FakeTorch:
        __version__ = "2.12.1+cpu"
        version = _Version()
        cuda = _Cuda()

    import sys

    monkeypatch.setitem(sys.modules, "torch", _FakeTorch)
    errors = smoke.check_torch_cpu()
    assert any("2.13" in error for error in errors)


def test_check_model_checkpoints_missing_files(tmp_path: Path) -> None:
    errors = smoke.check_model_checkpoints(models_dir=tmp_path)
    assert any("SAINT" in e for e in errors)
    assert any("DQN" in e for e in errors)
    assert any("MLP" in e for e in errors)


def test_run_smoke_against_repo_models() -> None:
    """Integration: real CPU torch + committed checkpoints (skip only if models absent)."""
    models = Path(__file__).resolve().parents[1] / "models"
    required = (
        models / "saint_best.pt",
        models / "dqn_best.pt",
        models / "prereq_vimath_bgem3_namedef_concat_rich.pth",
    )
    if not all(p.is_file() for p in required):
        pytest.fail(
            f"required model checkpoints missing; smoke must fail closed (looked under {models})"
        )
    errors = smoke.run_smoke(models_dir=models)
    assert errors == [], errors
