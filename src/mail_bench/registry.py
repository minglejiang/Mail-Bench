"""Canonical operator and dataset-registry loading and validation."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Mapping, Optional, Union


PathLike = Union[str, Path]
SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")
EXECUTION_STAGES = {f"stage_{i}" for i in range(6)}


class RegistryError(ValueError):
    """Raised when a registry or its use violates the frozen protocol."""


def load_yaml(path: PathLike) -> dict[str, Any]:
    """Load one YAML mapping, keeping PyYAML optional for the core package."""
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("PyYAML is required to load MAIL-Bench registries") from exc
    with Path(path).open("r", encoding="utf-8") as fh:
        value = yaml.safe_load(fh)
    if not isinstance(value, dict):
        raise RegistryError(f"{path}: expected a mapping at the top level")
    return value


@dataclass(frozen=True)
class DatasetValidation:
    dataset_id: Optional[str]
    stage: str
    mock: bool
    pinned: bool
    revision: Optional[str] = None
    inventory_sha256: Optional[str] = None
    accessed_on: Optional[str] = None


def validate_dataset(
    registry: Mapping[str, Any],
    dataset_id: Optional[str],
    *,
    stage: str,
    mock: bool = False,
) -> DatasetValidation:
    """Validate dataset permission and provenance, failing closed for real runs.

    A mock dataset is accepted only at ``stage_0``; every real run is subject
    to the registry policy.
    """
    if stage not in EXECUTION_STAGES:
        raise RegistryError(f"unknown execution stage {stage!r}")
    if mock:
        if stage != "stage_0":
            raise RegistryError("mock dataset bypass is allowed only in stage_0")
        return DatasetValidation(None, stage, True, False)
    if not dataset_id:
        raise RegistryError("a real run requires dataset_id")
    datasets = registry.get("datasets")
    if not isinstance(datasets, list):
        raise RegistryError("dataset registry must contain a datasets list")
    matches = [d for d in datasets if isinstance(d, dict) and d.get("id") == dataset_id]
    if len(matches) != 1:
        raise RegistryError(f"dataset {dataset_id!r} must appear exactly once in the registry")
    entry = matches[0]
    decision = str(entry.get("decision", ""))
    academic = str(entry.get("academic_evaluation", ""))
    blocked_words = ("hold", "pending", "exclude", "wait")
    if any(word in decision for word in blocked_words) or academic == "hold":
        raise RegistryError(f"dataset {dataset_id!r} is blocked by decision={decision!r}")

    policy = registry.get("policy") or {}
    pin_required = bool(policy.get("unpinned_entries_blocked_from_experiments", True))
    pinned = entry.get("pin_status") == "pinned"
    if pin_required and not pinned:
        raise RegistryError(f"dataset {dataset_id!r} is not pinned")
    if policy.get("require_version_pin", True) and not entry.get("revision"):
        raise RegistryError(f"dataset {dataset_id!r} has no revision pin")
    digest = entry.get("inventory_sha256")
    if policy.get("require_inventory_sha256", True) and not (
        isinstance(digest, str) and SHA256_RE.fullmatch(digest)
    ):
        raise RegistryError(f"dataset {dataset_id!r} has no valid inventory_sha256")
    accessed_on = entry.get("accessed_on")
    try:
        date.fromisoformat(str(accessed_on))
    except ValueError as exc:
        raise RegistryError(f"dataset {dataset_id!r} has no valid accessed_on date") from exc
    if policy.get("require_official_source", True) and not entry.get("source"):
        raise RegistryError(f"dataset {dataset_id!r} has no official source")
    return DatasetValidation(
        dataset_id,
        stage,
        False,
        pinned,
        revision=str(entry["revision"]),
        inventory_sha256=str(digest).lower(),
        accessed_on=str(accessed_on),
    )


def load_and_validate_dataset(
    registry_path: PathLike,
    dataset_id: Optional[str],
    *,
    stage: str,
    mock: bool = False,
) -> DatasetValidation:
    """Load a dataset registry and validate one execution request."""
    registry = load_yaml(registry_path)
    authorization = validate_dataset(registry, dataset_id, stage=stage, mock=mock)
    if authorization.mock:
        return authorization
    entry = next(
        item for item in registry["datasets"]
        if isinstance(item, dict) and item.get("id") == dataset_id
    )
    reference = entry.get("inventory_manifest")
    if (registry.get("policy") or {}).get("require_inventory_manifest", True) and not reference:
        raise RegistryError(f"dataset {dataset_id!r} has no inventory_manifest")
    if reference:
        registry_file = Path(registry_path).resolve()
        inventory_path = Path(str(reference))
        if not inventory_path.is_absolute():
            inventory_path = registry_file.parent / inventory_path
            if not inventory_path.exists():
                inventory_path = registry_file.parent.parent / str(reference)
        try:
            payload = json.loads(inventory_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RegistryError(
                f"dataset {dataset_id!r} inventory_manifest is unreadable"
            ) from exc
        if not isinstance(payload, dict) or payload.get("schema_version") != 1:
            raise RegistryError(f"dataset {dataset_id!r} inventory_manifest has invalid schema")
        from .manifest import semantic_sha256

        calculated = semantic_sha256(
            {"schema_version": 1, "files": payload.get("files")}
        )
        expected = {
            "dataset_id": dataset_id,
            "revision": authorization.revision,
            "accessed_on": authorization.accessed_on,
            "inventory_sha256": authorization.inventory_sha256,
        }
        mismatches = [key for key, value in expected.items() if payload.get(key) != value]
        if calculated != authorization.inventory_sha256:
            mismatches.append("calculated_inventory_sha256")
        if mismatches:
            raise RegistryError(
                f"dataset {dataset_id!r} inventory_manifest mismatch: {sorted(set(mismatches))}"
            )
    return authorization
