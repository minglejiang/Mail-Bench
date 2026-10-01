"""Auditable runtime certificates for real simulator and policy execution."""

from __future__ import annotations

import importlib.metadata
import os
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any, Optional, Union

from .manifest import semantic_sha256


PathLike = Union[str, Path]


def _git_bytes(root: Path, *args: str) -> Optional[bytes]:
    """Run git and return raw stdout, so undecodable content cannot raise."""
    try:
        completed = subprocess.run(
            ["git", "-C", str(root), *args],
            check=True,
            capture_output=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return completed.stdout


def _git_output(root: Path, *args: str) -> Optional[str]:
    raw = _git_bytes(root, *args)
    if raw is None:
        return None
    return raw.decode("utf-8", errors="replace").strip()


def _package_version(distribution: str) -> Optional[str]:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def collect_runtime_fingerprint(benchmark_root: PathLike) -> dict[str, Any]:
    """Collect stable provenance plus hardware/runtime facts without failing on CPU CI."""
    benchmark = Path(benchmark_root).resolve()
    # Hashed as raw bytes: a working tree with non-UTF-8 content must not be
    # able to break certificate collection, and the digest must not depend on
    # a lossy decode.
    source_diff = _git_bytes(benchmark, "diff", "--binary", "HEAD")
    payload: dict[str, Any] = {
        "schema_version": 2,
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "python_executable": str(Path(sys.executable).resolve()),
        "platform": platform.platform(),
        "packages": {
            name: _package_version(name)
            for name in ("numpy", "torch", "mujoco", "robosuite", "robocasa")
        },
        "rendering": {
            "MUJOCO_GL": os.environ.get("MUJOCO_GL"),
            "PYOPENGL_PLATFORM": os.environ.get("PYOPENGL_PLATFORM"),
        },
        "benchmark_git_commit": _git_output(benchmark, "rev-parse", "HEAD"),
        "benchmark_worktree_dirty": bool(_git_output(benchmark, "status", "--porcelain")),
        "benchmark_diff_hash": semantic_sha256(source_diff) if source_diff else None,
    }

    try:
        import torch

        payload["cuda_version"] = torch.version.cuda
        payload["cuda_available"] = bool(torch.cuda.is_available())
        payload["gpu_names"] = [
            torch.cuda.get_device_name(index)
            for index in range(torch.cuda.device_count())
        ]
    except ImportError:
        payload["cuda_version"] = None
        payload["cuda_available"] = False
        payload["gpu_names"] = []
    payload["fingerprint_hash"] = semantic_sha256(payload)
    return payload
