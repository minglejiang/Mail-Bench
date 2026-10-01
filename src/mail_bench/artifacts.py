"""Two levels of assurance that the bytes being run are the bytes that were pinned.

Rehashing sixteen gigabytes before every cohort would be absurd, and trusting a
timestamp would be negligent, so verification is split in two.

A **full verification** reads every byte, recomputes every file's SHA256 and the
inventory digest, and compares them against the frozen manifest. It is done once
when an artifact first becomes runnable -- and, for a loader that is known to
rewrite checkpoints, once again after loading, which turns "the loader should not
mutate the artifact" into something demonstrated rather than assumed.

A successful full verification writes a **receipt**: the inventory digest it
matched, plus size, mtime and inode for every file. The receipt is not a second
identity. Identity remains the upstream revision and the inventory digest; the
receipt only records that this particular tree on this particular machine once
passed a byte-level check.

Isolating the network is not enough for a checkpoint whose Python is loaded
through ``trust_remote_code``: transformers copies those files into
``HF_MODULES_CACHE`` and imports them from there, so a stale cache can execute
code that is not the code in the verified snapshot. The cache is therefore
emptied per process and, after loading, the modules actually imported are hashed
against the snapshot's own files.

A **fast guard** then runs at cohort startup. It stats every file and compares
against the receipt, and cryptographically re-checks the small files that decide
what the model computes. It reads no weights, so it is cheap, and it is not
verification: any drift raises and demands a full rehash rather than warning and
continuing.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

from .inventory import DatasetInventory, build_inventory
from .manifest import canonical_json

PathLike = Any

RECEIPT_SCHEMA_VERSION = 1


class ArtifactVerificationRequired(RuntimeError):
    """The tree changed since it was verified; a full rehash has to settle it."""


class ArtifactMismatch(RuntimeError):
    """The bytes are not the pinned bytes. No scientific run may proceed."""


@dataclass(frozen=True)
class VerificationReceipt:
    schema_version: int
    repo: str
    revision: str
    inventory_sha256: str
    verified_on: str
    files: Mapping[str, Mapping[str, int]]
    critical_sha256: Mapping[str, str]

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "repo": self.repo,
            "revision": self.revision,
            "inventory_sha256": self.inventory_sha256,
            "verified_on": self.verified_on,
            "files": {name: dict(stat) for name, stat in self.files.items()},
            "critical_sha256": dict(self.critical_sha256),
        }


def _file_stat(path: Path) -> dict[str, int]:
    """Size, mtime and inode.

    Size and mtime alone are forgeable: a file can be replaced and its timestamp
    restored with ``touch -r``. The inode is not cryptographic proof either, but
    together they answer the only question a startup guard asks -- has this tree
    changed at the filesystem level since it was hashed?
    """
    status = path.stat()
    return {"size": int(status.st_size),
            "mtime_ns": int(status.st_mtime_ns),
            "inode": int(status.st_ino)}


def _sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def full_verify(root: PathLike, manifest: PathLike, *, dataset_id: str,
                revision: str) -> DatasetInventory:
    """Recompute the whole tree and refuse anything the manifest did not pin."""
    root_path = Path(root)
    published = json.loads(Path(manifest).read_text(encoding="utf-8"))
    observed = build_inventory(root_path, dataset_id=dataset_id, revision=revision,
                               accessed_on=published["accessed_on"])
    if observed.inventory_sha256 != published["inventory_sha256"]:
        want = {item["relative_path"]: (item["size"], item["sha256"])
                for item in published["files"]}
        have = {item.relative_path: (item.size, item.sha256) for item in observed.files}
        differences = [name for name in sorted(set(want) | set(have))
                       if want.get(name) != have.get(name)]
        raise ArtifactMismatch(
            f"{dataset_id} at {root_path} does not match its pinned inventory "
            f"({published['inventory_sha256']} expected, {observed.inventory_sha256} "
            f"observed); differing paths: {', '.join(differences[:10])}"
        )
    return observed


def build_receipt(root: PathLike, inventory: DatasetInventory, *, repo: str,
                  critical_files: Sequence[str],
                  verified_on: Optional[str] = None) -> VerificationReceipt:
    """Record what a full verification saw, so a startup can check it cheaply."""
    root_path = Path(root)
    missing = [name for name in critical_files if not (root_path / name).is_file()]
    if missing:
        raise ArtifactMismatch(
            f"declared execution-critical files are absent: {', '.join(missing)}"
        )
    return VerificationReceipt(
        schema_version=RECEIPT_SCHEMA_VERSION,
        repo=repo,
        revision=inventory.revision,
        inventory_sha256=inventory.inventory_sha256,
        verified_on=verified_on or date.today().isoformat(),
        files={item.relative_path: _file_stat(root_path / item.relative_path)
               for item in inventory.files},
        critical_sha256={name: _sha256(root_path / name) for name in sorted(critical_files)},
    )


def write_receipt(receipt: VerificationReceipt, output: PathLike) -> Path:
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(canonical_json(receipt.as_dict()), encoding="utf-8")
    return path


def read_receipt(path: PathLike) -> VerificationReceipt:
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    if document.get("schema_version") != RECEIPT_SCHEMA_VERSION:
        raise ArtifactVerificationRequired(
            f"receipt schema {document.get('schema_version')!r} is not "
            f"{RECEIPT_SCHEMA_VERSION}; re-verify rather than guess its meaning"
        )
    return VerificationReceipt(
        schema_version=document["schema_version"],
        repo=document["repo"],
        revision=document["revision"],
        inventory_sha256=document["inventory_sha256"],
        verified_on=document["verified_on"],
        files=document["files"],
        critical_sha256=document["critical_sha256"],
    )


def fast_guard(root: PathLike, receipt: VerificationReceipt, *,
               expected_inventory_sha256: Optional[str] = None) -> dict[str, Any]:
    """Cheap startup check. Not verification, and never a warning.

    Every file is stat-ed and every declared execution-critical file is hashed.
    Anything that disagrees raises :class:`ArtifactVerificationRequired`: the only
    way out is a full rehash, which either restores the receipt or forbids the
    run outright.
    """
    root_path = Path(root)
    if expected_inventory_sha256 and receipt.inventory_sha256 != expected_inventory_sha256:
        raise ArtifactVerificationRequired(
            f"receipt was written for inventory {receipt.inventory_sha256}, but the "
            f"roster pins {expected_inventory_sha256}"
        )

    present = {path.relative_to(root_path).as_posix()
               for path in root_path.rglob("*") if path.is_file()}
    recorded = set(receipt.files)
    if present != recorded:
        added = sorted(present - recorded)[:10]
        removed = sorted(recorded - present)[:10]
        raise ArtifactVerificationRequired(
            f"the artifact tree gained {added or 'nothing'} and lost "
            f"{removed or 'nothing'} since it was verified"
        )

    for name, expected in sorted(receipt.files.items()):
        observed = _file_stat(root_path / name)
        if observed != dict(expected):
            raise ArtifactVerificationRequired(
                f"{name} changed on disk since verification "
                f"(recorded {dict(expected)}, found {observed})"
            )

    for name, expected_digest in sorted(receipt.critical_sha256.items()):
        observed_digest = _sha256(root_path / name)
        if observed_digest != expected_digest:
            raise ArtifactVerificationRequired(
                f"{name} hashes to {observed_digest}, not the verified {expected_digest}"
            )

    # Deliberately not "artifact_verified": that word belongs to the full
    # byte-level check alone, or provenance becomes ambiguous.
    return {
        "artifact_fast_guard_passed": True,
        "artifact_verification_receipt_valid": True,
        "inventory_sha256": receipt.inventory_sha256,
        "files_checked": len(receipt.files),
        "critical_files_hashed": len(receipt.critical_sha256),
    }


def make_read_only(root: PathLike) -> int:
    """Drop write permission across a verified tree. A second layer, not a proof."""
    root_path = Path(root)
    changed = 0
    for path in list(root_path.rglob("*")) + [root_path]:
        mode = path.stat().st_mode
        stripped = mode & ~0o222
        if stripped != mode:
            os.chmod(path, stripped)
            changed += 1
    return changed


def inventory_digest(root: PathLike, *, dataset_id: str, revision: str,
                     accessed_on: str = "1970-01-01") -> str:
    """The digest alone, for before/after comparisons around a model load."""
    return build_inventory(root, dataset_id=dataset_id, revision=revision,
                           accessed_on=accessed_on).inventory_sha256


def prepare_modules_cache(path: PathLike) -> Path:
    """Empty and recreate the dynamic-module cache for this process.

    ``trust_remote_code`` copies a checkpoint's Python into this directory and
    imports it from there. Reusing a previous run's copy would mean a verified
    snapshot on disk and somebody else's code in memory, so the directory is
    rebuilt every time. Call this before transformers is imported.
    """
    import shutil

    cache = Path(path)
    if cache.exists():
        shutil.rmtree(cache)
    cache.mkdir(parents=True)
    return cache


def loaded_remote_code(modules_cache: PathLike) -> dict[str, str]:
    """SHA256 of every dynamic module that was actually imported."""
    import sys

    cache = Path(modules_cache).resolve()
    digests: dict[str, str] = {}
    for module in list(sys.modules.values()):
        location = getattr(module, "__file__", None)
        if not location:
            continue
        path = Path(location).resolve()
        if cache in path.parents and path.is_file():
            digests[path.name] = _sha256(path)
    return digests


def verify_remote_code(snapshot: PathLike, modules_cache: PathLike,
                       expected_files: Iterable[str]) -> dict[str, str]:
    """Prove the executed code is the code the inventory pinned.

    Verifying the snapshot alone leaves a gap: the model runs whatever was
    imported, which is a copy. This closes it by hashing the imported copies and
    requiring each to equal the snapshot file of the same name, and by refusing a
    load in which a declared module was never imported at all.
    """
    snapshot_path = Path(snapshot)
    imported = loaded_remote_code(modules_cache)
    expected = list(expected_files)

    absent = [name for name in expected if name not in imported]
    if absent:
        raise ArtifactMismatch(
            f"declared remote-code modules were never imported from the module "
            f"cache: {', '.join(absent)}"
        )
    for name in expected:
        pinned_digest = _sha256(snapshot_path / name)
        if imported[name] != pinned_digest:
            raise ArtifactMismatch(
                f"the imported {name} hashes to {imported[name]}, but the verified "
                f"snapshot holds {pinned_digest}: the running code is not the "
                f"pinned code"
            )
    return {name: imported[name] for name in expected}


__all__ = [
    "ArtifactMismatch",
    "ArtifactVerificationRequired",
    "VerificationReceipt",
    "build_receipt",
    "fast_guard",
    "full_verify",
    "inventory_digest",
    "loaded_remote_code",
    "make_read_only",
    "prepare_modules_cache",
    "read_receipt",
    "verify_remote_code",
    "write_receipt",
]
