import json
import os
from pathlib import Path
import subprocess

import pytest

from mail_bench.inventory import build_inventory, git_tracked_paths, main


ROOT = Path(__file__).resolve().parents[1]


def test_inventory_is_sorted_portable_and_content_sensitive(tmp_path):
    root = tmp_path / "dataset"
    (root / "nested").mkdir(parents=True)
    (root / "z.bin").write_bytes(b"z")
    (root / "nested" / "a.bin").write_bytes(b"alpha")
    first = build_inventory(
        root, dataset_id="fixture", revision="commit-1", accessed_on="2026-08-29"
    )
    second = build_inventory(
        root, dataset_id="fixture", revision="commit-1", accessed_on="2026-08-29"
    )
    assert first == second
    assert [item.relative_path for item in first.files] == ["nested/a.bin", "z.bin"]
    assert first.total_bytes == 6 and len(first.inventory_sha256) == 64

    (root / "z.bin").write_bytes(b"changed")
    changed = build_inventory(
        root, dataset_id="fixture", revision="commit-1", accessed_on="2026-08-29"
    )
    assert changed.inventory_sha256 != first.inventory_sha256


def test_inventory_rejects_symlinks_and_empty_roots(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(ValueError, match="no files"):
        build_inventory(empty, dataset_id="fixture", revision="r1", accessed_on="2026-08-29")
    (empty / "real").write_text("data", encoding="utf-8")
    (empty / "link").symlink_to(empty / "real")
    with pytest.raises(ValueError, match="symlinks"):
        build_inventory(empty, dataset_id="fixture", revision="r1", accessed_on="2026-08-29")


def test_inventory_cli_writes_canonical_output_and_excludes_itself(tmp_path, capsys):
    root = tmp_path / "dataset"
    root.mkdir()
    (root / "episode.dat").write_bytes(b"episode")
    output = root / "inventory.json"
    args = [
        "--root", str(root),
        "--dataset-id", "fixture",
        "--revision", "commit-1",
        "--accessed-on", "2026-08-29",
        "--output", str(output),
    ]
    assert main(args) == 0
    first_bytes = output.read_bytes()
    assert main(args) == 0
    assert output.read_bytes() == first_bytes
    assert os.stat(output).st_mode & 0o777 == 0o644
    payload = json.loads(first_bytes)
    assert payload["file_count"] == 1
    assert payload["files"][0]["relative_path"] == "episode.dat"
    assert "sha256=" in capsys.readouterr().out


def test_git_tracked_inventory_excludes_untracked_and_rejects_dirty_files(tmp_path):
    root = tmp_path / "checkout"
    root.mkdir()

    def git(*args):
        return subprocess.run(
            ["git", "-C", str(root), *args],
            check=True,
            stdout=subprocess.PIPE,
        ).stdout.decode("ascii").strip()

    git("init", "-q")
    git("config", "user.email", "test@example.com")
    git("config", "user.name", "Test")
    (root / "tracked.txt").write_text("tracked", encoding="utf-8")
    git("add", "tracked.txt")
    git("commit", "-qm", "fixture")
    revision = git("rev-parse", "HEAD")
    (root / "ignored.cache").write_text("local", encoding="utf-8")
    paths = git_tracked_paths(root, revision)
    assert paths == ("tracked.txt",)
    inventory = build_inventory(
        root,
        dataset_id="fixture",
        revision=revision,
        accessed_on="2026-08-29",
        relative_paths=paths,
    )
    assert inventory.file_count == 1

    (root / "tracked.txt").write_text("dirty", encoding="utf-8")
    with pytest.raises(ValueError, match="clean tracked worktree"):
        git_tracked_paths(root, revision)


