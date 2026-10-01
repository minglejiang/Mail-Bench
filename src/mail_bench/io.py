"""Atomic, idempotent result persistence for resumable evaluation."""

from __future__ import annotations

import json
import os
import re
import tempfile
from pathlib import Path
from typing import Union

from .manifest import (
    FaultManifest,
    ResultRecord,
    canonical_json,
    semantic_sha256,
    validate_result,
)


PathLike = Union[str, Path]
RUN_ID_RE = re.compile(r"^[A-Za-z0-9_.-]+$")


class ResultConflictError(RuntimeError):
    """Raised when a completed run id already contains different bytes."""


def semantic_run_key(result: ResultRecord, manifest: FaultManifest) -> str:
    """Identity of one evaluated cell, independent of a caller-chosen run id."""
    validated = validate_result(result, manifest)
    return semantic_run_key_fields(
        manifest,
        method_id=validated.method_id,
        checkpoint_hash=validated.checkpoint_hash,
        method_config_hash=validated.method_config_hash,
        schedule_hash=validated.schedule_hash,
        runner_config_hash=validated.runner_config_hash,
        dataset_authorization_hash=validated.dataset_authorization_hash,
    )


def semantic_run_key_fields(
    manifest: FaultManifest,
    *,
    method_id: str,
    checkpoint_hash: str,
    method_config_hash: str,
    schedule_hash: str,
    runner_config_hash: str,
    dataset_authorization_hash: str,
) -> str:
    """Precompute a semantic run key before launching an expensive rollout."""
    required = {
        "method_id": method_id,
        "checkpoint_hash": checkpoint_hash,
        "method_config_hash": method_config_hash,
        "schedule_hash": schedule_hash,
        "runner_config_hash": runner_config_hash,
        "dataset_authorization_hash": dataset_authorization_hash,
    }
    missing = [name for name, value in required.items() if not value]
    if missing:
        raise ValueError(f"semantic run identity missing fields: {missing}")
    return semantic_sha256({"manifest_hash": manifest.semantic_hash(), **required})


class AtomicResultWriter:
    """Write one validated result per file with idempotent resume semantics."""

    def __init__(self, root: PathLike) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def result_path(self, run_id: str) -> Path:
        if not RUN_ID_RE.fullmatch(run_id):
            raise ValueError("run_id may contain only letters, digits, dot, underscore and hyphen")
        return self.root / f"{run_id}.json"

    def write(self, run_id: str, result: ResultRecord, manifest: FaultManifest) -> Path:
        """Validate and atomically write, accepting byte-identical retries."""
        validated = validate_result(result, manifest)
        payload = canonical_json(
            {
                "manifest_hash": manifest.semantic_hash(),
                "result": validated,
            }
        ) + "\n"
        target = self.result_path(run_id)
        if target.exists():
            existing = target.read_text(encoding="utf-8")
            if existing == payload:
                return target
            raise ResultConflictError(f"run id {run_id!r} already exists with different content")

        fd, temporary_name = tempfile.mkstemp(prefix=f".{run_id}.", suffix=".tmp", dir=self.root)
        temporary = Path(temporary_name)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                fh.write(payload)
                fh.flush()
                os.fsync(fh.fileno())
            try:
                os.link(temporary, target)
            except FileExistsError:
                existing = target.read_text(encoding="utf-8")
                if existing != payload:
                    raise ResultConflictError(
                        f"run id {run_id!r} already exists with different content"
                    )
            directory_fd = os.open(self.root, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        finally:
            if temporary.exists():
                temporary.unlink()
        return target

    def read(self, run_id: str, manifest: FaultManifest) -> ResultRecord:
        """Read and validate one completed atomic result against its manifest."""
        path = self.result_path(run_id)
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ResultConflictError(f"run id {run_id!r} is not a readable result") from exc
        if not isinstance(payload, dict) or payload.get("manifest_hash") != manifest.semantic_hash():
            raise ResultConflictError(f"run id {run_id!r} has a different manifest identity")
        result_payload = payload.get("result")
        if not isinstance(result_payload, dict):
            raise ResultConflictError(f"run id {run_id!r} has no result record")
        try:
            return validate_result(ResultRecord(**result_payload), manifest)
        except (TypeError, ValueError) as exc:
            raise ResultConflictError(f"run id {run_id!r} contains an invalid result") from exc

    def write_semantic(self, result: ResultRecord, manifest: FaultManifest) -> Path:
        """Write under the canonical semantic key, preventing renamed duplicates."""
        return self.write(semantic_run_key(result, manifest), result, manifest)
