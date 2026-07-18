"""Production dependency/runtime smoke guard.

Fails closed unless:
  * torch is a CPU build (``+cpu`` local version tag, or CUDA-free darwin wheel)
  * CUDA is not linked into the torch binary
  * banned CUDA/NVIDIA/triton distributions are absent from the environment
  * SAINT, DQN, and MLP checkpoints load on CPU

Intended to run after a frozen ``uv sync`` in CI and Docker gates.
"""

from __future__ import annotations

import importlib.metadata
import os
from pathlib import Path
import secrets
import sys

# Match api.config._MIN_SERVICE_TOKEN_LENGTH without importing Settings early.
_MIN_SERVICE_TOKEN_LENGTH = 32

# CUDA runtime / NVIDIA wheel redistribs and triton must be absent.
# Note: nvidia-ml-py (NVML bindings, pulled by some profilers) is monitoring-only
# and is intentionally not banned.
_BANNED_DIST_PREFIXES = (
    "nvidia-cublas",
    "nvidia-cuda",
    "nvidia-cudnn",
    "nvidia-cufft",
    "nvidia-cufile",
    "nvidia-curand",
    "nvidia-cusolver",
    "nvidia-cusparse",
    "nvidia-cusparselt",
    "nvidia-nccl",
    "nvidia-nvjitlink",
    "nvidia-nvshmem",
    "nvidia-nvtx",
    "cuda-bindings",
    "cuda-pathfinder",
    "cuda-python",
    "cuda-toolkit",
)
_BANNED_DIST_NAMES = frozenset(
    {
        "triton",
        "cudatoolkit",
    }
)

_REPO_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_MODELS = _REPO_ROOT / "models"
_EXPECTED_TORCH_MAJOR_MINOR = (2, 13)


def _ensure_ephemeral_settings_env() -> None:
    """Provide a non-placeholder INTERNAL_SERVICE_TOKEN in-process only.

    Settings load on some api package imports. Generate a disposable token when
    unset/invalid so the smoke guard never depends on a local .env. Never print it.
    """
    os.environ.setdefault("ENVIRONMENT", "dev")
    current = os.environ.get("INTERNAL_SERVICE_TOKEN", "")
    if (
        len(current) >= _MIN_SERVICE_TOKEN_LENGTH
        and "change-me" not in current.casefold()
        and "placeholder" not in current.casefold()
    ):
        return
    os.environ["INTERNAL_SERVICE_TOKEN"] = secrets.token_urlsafe(_MIN_SERVICE_TOKEN_LENGTH)


def _dist_name(dist: importlib.metadata.Distribution) -> str:
    name = dist.metadata["Name"] if dist.metadata is not None else dist.name
    return str(name).lower().replace("_", "-")


def check_banned_distributions() -> list[str]:
    """Return installed distribution names that are banned for CPU production."""
    found: list[str] = []
    for dist in importlib.metadata.distributions():
        name = _dist_name(dist)
        if name in _BANNED_DIST_NAMES or any(name.startswith(p) for p in _BANNED_DIST_PREFIXES):
            found.append(name)
    return sorted(set(found))


def check_torch_cpu() -> list[str]:
    """Validate torch is a CPU build without CUDA linkage."""
    errors: list[str] = []
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - environment misconfiguration
        return [f"torch import failed: {exc}"]

    version = str(getattr(torch, "__version__", ""))
    cuda_version = getattr(getattr(torch, "version", None), "cuda", None)

    public_version = version.split("+", maxsplit=1)[0]
    try:
        major, minor, *_rest = (int(part) for part in public_version.split("."))
    except ValueError:
        errors.append(f"torch version is not parseable: {version!r}")
    else:
        if (major, minor) != _EXPECTED_TORCH_MAJOR_MINOR:
            errors.append(
                f"torch version must match the frozen 2.13.x runtime contract; got {version!r}"
            )

    # Linux/Windows CPU wheels use the +cpu local tag. Official macOS CPU wheels
    # from the pytorch-cpu index may omit it; reject any CUDA local tag always.
    has_cuda_tag = "+cu" in version or "+cuda" in version.lower()
    has_cpu_tag = "+cpu" in version
    if has_cuda_tag:
        errors.append(f"torch must be a CPU build; CUDA local version tag present: {version!r}")
    elif not has_cpu_tag and cuda_version is not None:
        errors.append(
            f"torch must be a CPU build (+cpu tag or null torch.version.cuda); "
            f"got version={version!r} cuda={cuda_version!r}"
        )

    if cuda_version is not None:
        errors.append(f"torch.version.cuda must be None for CPU builds; got {cuda_version!r}")

    # Fail closed if a CUDA-capable binary is present even without a GPU.
    if hasattr(torch, "cuda") and torch.cuda.is_available():
        errors.append("torch.cuda.is_available() is True; expected CPU-only runtime")

    return errors


def check_model_checkpoints(models_dir: Path | None = None) -> list[str]:
    """Load SAINT, DQN, and MLP checkpoints on CPU and verify state dicts apply."""
    _ensure_ephemeral_settings_env()
    import torch

    from api.domains.content_pipeline.infrastructure.embed.mlp_prereq.model import (
        PrerequisiteClassifier,
    )
    from api.domains.content_pipeline.infrastructure.embed.mlp_prereq.ranker import (
        PrerequisiteModelMetadata,
        _default_metadata_path,
    )
    from api.domains.learning.models import load_dqn_model, load_saint_model

    root = models_dir if models_dir is not None else _DEFAULT_MODELS
    errors: list[str] = []
    device = torch.device("cpu")

    saint_path = root / "saint_best.pt"
    dqn_path = root / "dqn_best.pt"
    mlp_path = root / "prereq_vimath_bgem3_namedef_concat_rich.pth"

    for label, path in (
        ("SAINT", saint_path),
        ("DQN", dqn_path),
        ("MLP", mlp_path),
    ):
        if not path.is_file():
            errors.append(f"{label} checkpoint missing: {path}")

    if errors:
        return errors

    try:
        model, _concept_map, _cfg = load_saint_model(str(saint_path), device)
        if model is None:
            errors.append("SAINT load returned no model")
    except Exception as exc:
        errors.append(f"SAINT checkpoint load failed: {exc}")

    try:
        q_net, _info = load_dqn_model(str(dqn_path), device)
        if q_net is None:
            errors.append("DQN load returned no model")
    except Exception as exc:
        errors.append(f"DQN checkpoint load failed: {exc}")

    try:
        metadata = PrerequisiteModelMetadata.from_path(_default_metadata_path(mlp_path))
        ckpt = torch.load(mlp_path, map_location=device, weights_only=False)
        state = ckpt.get("model_state_dict", ckpt)
        mlp = PrerequisiteClassifier(
            input_dim=metadata.input_dim,
            hidden_dim=512,
            dropout=0.3,
        )
        mlp.load_state_dict(state)
        mlp.eval()
    except Exception as exc:
        errors.append(f"MLP checkpoint load failed: {exc}")

    return errors


def run_smoke(models_dir: Path | None = None) -> list[str]:
    """Run all smoke checks; return a list of error strings (empty = pass)."""
    errors: list[str] = []
    errors.extend(check_torch_cpu())
    errors.extend(check_banned_distributions())
    errors.extend(check_model_checkpoints(models_dir=models_dir))
    return errors


def main(argv: list[str] | None = None) -> int:
    del argv  # reserved for future CLI flags
    errors = run_smoke()
    if errors:
        sys.stderr.write("dependency/runtime smoke FAILED:\n")
        for err in errors:
            sys.stderr.write(f"  - {err}\n")
        return 1
    sys.stdout.write(
        "dependency/runtime smoke OK: CPU torch, no banned CUDA/NVIDIA/triton, checkpoints load\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
