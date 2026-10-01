"""Deterministic dataset inventory generation for registry pinning."""

from __future__ import annotations

import argparse
import hashlib
import os
import stat
import subprocess
import tempfile
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Iterable, Optional, Union

from .manifest import canonical_json, semantic_sha256


PathLike = Union[str, Path]
CHUNK_SIZE = 4 * 1024 * 1024


@dataclass(frozen=True)
class InventoryFile:
    relative_path: str
    size: int
    sha256: str


@dataclass(frozen=True)
class DatasetInventory:
    schema_version: int
    dataset_id: str
    revision: str
    accessed_on: str
    file_count: int
    total_bytes: int
    inventory_sha256: str
    files: tuple[InventoryFile, ...]


def _file_sha256(path: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        before = os.fstat(handle.fileno())
        if not stat.S_ISREG(before.st_mode):
            raise ValueError(f"dataset inventory supports regular files only: {path}")
        while chunk := handle.read(CHUNK_SIZE):
            digest.update(chunk)
        after = os.fstat(handle.fileno())
    current = path.lstat()

    def identity(value: os.stat_result) -> tuple[int, int, int, int]:
        return value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns

    if identity(before) != identity(after) or identity(after) != identity(current):
        raise RuntimeError(f"file changed while inventory was generated: {path}")
    return after.st_size, digest.hexdigest()


def build_inventory(
    root: PathLike,
    *,
    dataset_id: str,
    revision: str,
    accessed_on: str,
    exclude: Iterable[PathLike] = (),
    relative_paths: Optional[Iterable[str]] = None,
) -> DatasetInventory:
    """Hash a portable sorted inventory of every regular file below ``root``."""
    root_path = Path(root).resolve()
    if not root_path.is_dir():
        raise ValueError(f"dataset root is not a directory: {root_path}")
    if not dataset_id or not revision:
        raise ValueError("dataset_id and revision are required")
    try:
        date.fromisoformat(accessed_on)
    except ValueError as exc:
        raise ValueError("accessed_on must be an ISO date") from exc
    excluded = {Path(path).resolve() for path in exclude}
    files: list[InventoryFile] = []
    if relative_paths is None:
        candidates = list(root_path.rglob("*"))
    else:
        names = list(relative_paths)
        if len(set(names)) != len(names):
            raise ValueError("dataset inventory relative_paths contains duplicates")
        candidates = []
        for name in names:
            relative = Path(name)
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError(f"dataset inventory path escapes its root: {name}")
            candidates.append(root_path / relative)
    for path in sorted(candidates, key=lambda item: item.relative_to(root_path).as_posix()):
        if path.is_symlink():
            raise ValueError(f"dataset inventory rejects symlinks: {path}")
        if path.resolve() in excluded:
            continue
        if path.is_dir():
            continue
        if not path.is_file():
            raise ValueError(f"dataset inventory supports regular files only: {path}")
        size, digest = _file_sha256(path)
        files.append(InventoryFile(path.relative_to(root_path).as_posix(), size, digest))
    if not files:
        raise ValueError("dataset inventory contains no files")
    inventory_hash = semantic_sha256(
        {"schema_version": 1, "files": files}
    )
    return DatasetInventory(
        schema_version=1,
        dataset_id=dataset_id,
        revision=revision,
        accessed_on=accessed_on,
        file_count=len(files),
        total_bytes=sum(item.size for item in files),
        inventory_sha256=inventory_hash,
        files=tuple(files),
    )


def git_tracked_paths(root: PathLike, revision: str) -> tuple[str, ...]:
    """Return tracked paths only when ``root`` is clean at the exact revision."""
    root_path = Path(root).resolve()

    def git(*args: str) -> bytes:
        try:
            return subprocess.run(
                ["git", "-C", str(root_path), *args],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            ).stdout
        except (OSError, subprocess.CalledProcessError) as exc:
            raise ValueError(f"cannot inspect git-tracked inventory at {root_path}") from exc

    head = git("rev-parse", "HEAD").decode("ascii").strip()
    if head != revision:
        raise ValueError(f"git HEAD {head} does not match requested revision {revision}")
    if git("status", "--porcelain", "--untracked-files=no"):
        raise ValueError("git-tracked inventory requires a clean tracked worktree")
    raw_paths = git("ls-files", "-z")
    paths = tuple(os.fsdecode(part) for part in raw_paths.split(b"\0") if part)
    if not paths:
        raise ValueError("git-tracked inventory contains no files")
    return paths


def write_inventory(inventory: DatasetInventory, output: PathLike) -> Path:
    """Atomically replace a generated inventory with canonical JSON."""
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = canonical_json(inventory) + "\n"
    fd, temporary_name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            os.fchmod(handle.fileno(), 0o644)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
        directory_fd = os.open(target.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary.exists():
            temporary.unlink()
    return target


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, help="read-only dataset root")
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--accessed-on", required=True, help="ISO date, YYYY-MM-DD")
    parser.add_argument("--output", required=True, help="inventory JSON path")
    parser.add_argument(
        "--git-tracked",
        action="store_true",
        help="hash only files tracked by a clean git checkout at --revision",
    )
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    relative_paths = git_tracked_paths(args.root, args.revision) if args.git_tracked else None
    inventory = build_inventory(
        args.root,
        dataset_id=args.dataset_id,
        revision=args.revision,
        accessed_on=args.accessed_on,
        exclude=(args.output,),
        relative_paths=relative_paths,
    )
    path = write_inventory(inventory, args.output)
    print(
        f"inventory={path} files={inventory.file_count} bytes={inventory.total_bytes} "
        f"sha256={inventory.inventory_sha256}"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
