"""Two levels of assurance, and the discipline that keeps them apart.

The expensive check has to catch a changed byte; the cheap one has to catch a
changed tree without reading it; and neither may be allowed to speak in the
other's name.
"""

import json
import os
from pathlib import Path

import pytest

from mail_bench.artifacts import (
    ArtifactMismatch,
    ArtifactVerificationRequired,
    build_receipt,
    fast_guard,
    full_verify,
    inventory_digest,
    make_read_only,
    read_receipt,
    write_receipt,
)
from mail_bench.inventory import build_inventory, write_inventory

CRITICAL = ("config.json", "modeling.py", "dataset_statistics.json")


def snapshot(root: Path) -> Path:
    root.mkdir(parents=True)
    (root / "config.json").write_text('{"model_type": "model"}', encoding="utf-8")
    (root / "modeling.py").write_text("class Model: pass\n", encoding="utf-8")
    (root / "dataset_statistics.json").write_text('{"q01": [0.0]}', encoding="utf-8")
    (root / "weights.safetensors").write_bytes(b"\x01\x02" * 4096)
    return root


def pinned(root: Path, manifest: Path, revision="a" * 40):
    inventory = build_inventory(root, dataset_id="fixture", revision=revision,
                                accessed_on="2026-08-30")
    write_inventory(inventory, manifest)
    return inventory


def verified(tmp_path):
    root = snapshot(tmp_path / "artifact")
    manifest = tmp_path / "fixture.json"
    pinned(root, manifest)
    inventory = full_verify(root, manifest, dataset_id="fixture", revision="a" * 40)
    receipt = build_receipt(root, inventory, repo="owner/repo", critical_files=CRITICAL)
    return root, manifest, receipt


def test_a_changed_byte_is_caught_and_names_the_file(tmp_path):
    root = snapshot(tmp_path / "artifact")
    manifest = tmp_path / "fixture.json"
    pinned(root, manifest)
    full_verify(root, manifest, dataset_id="fixture", revision="a" * 40)

    payload = bytearray((root / "weights.safetensors").read_bytes())
    payload[0] ^= 0x01                       # same size, different content
    (root / "weights.safetensors").write_bytes(bytes(payload))
    with pytest.raises(ArtifactMismatch, match="weights.safetensors"):
        full_verify(root, manifest, dataset_id="fixture", revision="a" * 40)


def test_a_receipt_survives_a_round_trip(tmp_path):
    root, _, receipt = verified(tmp_path)
    path = write_receipt(receipt, tmp_path / "receipt.json")
    restored = read_receipt(path)
    assert restored.inventory_sha256 == receipt.inventory_sha256
    assert set(restored.files) == set(receipt.files)
    assert restored.critical_sha256 == receipt.critical_sha256
    assert fast_guard(root, restored)["artifact_fast_guard_passed"] is True


def test_the_fast_guard_never_calls_itself_verification(tmp_path):
    root, _, receipt = verified(tmp_path)
    result = fast_guard(root, receipt)
    # "artifact_verified" belongs to the byte-level check alone.
    assert "artifact_verified" not in result
    assert result["artifact_verification_receipt_valid"] is True
    assert result["files_checked"] == 4
    assert result["critical_files_hashed"] == 3


def test_a_rewritten_critical_file_is_caught_without_reading_the_weights(tmp_path):
    root, _, receipt = verified(tmp_path)
    # A critical file is rewritten in place.
    (root / "modeling.py").write_text("class Model: pass  # patched\n",
                                                encoding="utf-8")
    with pytest.raises(ArtifactVerificationRequired, match="modeling.py"):
        fast_guard(root, receipt)


def test_a_replaced_file_with_a_restored_timestamp_is_still_caught(tmp_path):
    root, _, receipt = verified(tmp_path)
    target = root / "weights.safetensors"
    status = target.stat()
    # cp then `touch -r`: same size, same mtime, different inode.
    replacement = root / "replacement.tmp"
    replacement.write_bytes(b"\x03\x04" * 4096)
    os.replace(replacement, target)
    os.utime(target, ns=(status.st_atime_ns, status.st_mtime_ns))
    assert target.stat().st_size == status.st_size
    assert target.stat().st_mtime_ns == status.st_mtime_ns
    with pytest.raises(ArtifactVerificationRequired, match="weights.safetensors"):
        fast_guard(root, receipt)


def test_an_added_or_removed_file_is_caught(tmp_path):
    root, _, receipt = verified(tmp_path)
    # A stray file appears beside the pinned ones.
    (root / "config.json.bak").write_text("{}", encoding="utf-8")
    with pytest.raises(ArtifactVerificationRequired, match="gained"):
        fast_guard(root, receipt)


def test_a_receipt_for_a_different_pin_is_refused(tmp_path):
    root, _, receipt = verified(tmp_path)
    with pytest.raises(ArtifactVerificationRequired, match="roster pins"):
        fast_guard(root, receipt, expected_inventory_sha256="b" * 64)


def test_a_receipt_from_an_unknown_schema_demands_re_verification(tmp_path):
    root, _, receipt = verified(tmp_path)
    path = write_receipt(receipt, tmp_path / "receipt.json")
    document = json.loads(path.read_text(encoding="utf-8"))
    document["schema_version"] = 99
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ArtifactVerificationRequired, match="schema"):
        read_receipt(path)


def test_a_receipt_requires_every_declared_critical_file_to_exist(tmp_path):
    root = snapshot(tmp_path / "artifact")
    manifest = tmp_path / "fixture.json"
    inventory = pinned(root, manifest)
    with pytest.raises(ArtifactMismatch, match="generation_config.json"):
        build_receipt(root, inventory, repo="owner/repo",
                      critical_files=(*CRITICAL, "generation_config.json"))


def test_the_digest_is_stable_across_a_read_only_pass(tmp_path):
    root, _, _ = verified(tmp_path)
    before = inventory_digest(root, dataset_id="fixture", revision="a" * 40)
    assert make_read_only(root) > 0
    after = inventory_digest(root, dataset_id="fixture", revision="a" * 40)
    # Permissions are not content: locking the tree must not change its identity.
    assert before == after
    for path in root.rglob("*"):
        assert not (path.stat().st_mode & 0o222), path


# --------------------------------------------------------------------------
# trust_remote_code: what runs is a copy, so verifying the snapshot is not enough
# --------------------------------------------------------------------------

REMOTE_CODE = ("modeling.py", "configuration.py")


def fake_import(cache: Path, name: str, source: str):
    """Stand in for what transformers does: copy the file, then import it."""
    import sys
    import types

    target = cache / "somehash" / name
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(source, encoding="utf-8")
    module = types.ModuleType(f"transformers_modules.{name[:-3]}")
    module.__file__ = str(target)
    sys.modules[module.__name__] = module
    return module


@pytest.fixture
def clean_modules(monkeypatch):
    import sys

    before = set(sys.modules)
    yield
    for name in set(sys.modules) - before:
        del sys.modules[name]


def test_the_module_cache_is_rebuilt_rather_than_reused(tmp_path):
    from mail_bench.artifacts import prepare_modules_cache

    cache = tmp_path / "modules"
    cache.mkdir()
    stale = cache / "stale_module.py"
    stale.write_text("# a previous run's code\n", encoding="utf-8")

    prepare_modules_cache(cache)
    # A previous run's copy must not survive into this one.
    assert cache.is_dir()
    assert not stale.exists()
    assert list(cache.iterdir()) == []


def test_the_code_that_ran_is_the_code_that_was_pinned(tmp_path, clean_modules):
    from mail_bench.artifacts import verify_remote_code

    snapshot = tmp_path / "artifact"
    snapshot.mkdir()
    cache = tmp_path / "modules"
    cache.mkdir()
    for name in REMOTE_CODE:
        source = f"# {name}\nclass Thing: pass\n"
        (snapshot / name).write_text(source, encoding="utf-8")
        fake_import(cache, name, source)

    digests = verify_remote_code(snapshot, cache, REMOTE_CODE)
    assert set(digests) == set(REMOTE_CODE)


def test_a_stale_cached_module_is_caught_even_when_the_snapshot_is_intact(
    tmp_path, clean_modules
):
    from mail_bench.artifacts import ArtifactMismatch, verify_remote_code

    snapshot = tmp_path / "artifact"
    snapshot.mkdir()
    cache = tmp_path / "modules"
    cache.mkdir()
    for name in REMOTE_CODE:
        (snapshot / name).write_text(f"# {name}\nclass Thing: pass\n", encoding="utf-8")
    fake_import(cache, "configuration.py",
                "# configuration.py\nclass Thing: pass\n")
    # The snapshot is untouched, but an older copy is what got imported.
    fake_import(cache, "modeling.py",
                "# modeling.py\nclass Thing:\n    pass  # last week's build\n")

    with pytest.raises(ArtifactMismatch, match="running code is not the pinned code"):
        verify_remote_code(snapshot, cache, REMOTE_CODE)


def test_a_module_that_was_never_imported_is_not_silently_accepted(
    tmp_path, clean_modules
):
    from mail_bench.artifacts import ArtifactMismatch, verify_remote_code

    snapshot = tmp_path / "artifact"
    snapshot.mkdir()
    cache = tmp_path / "modules"
    cache.mkdir()
    for name in REMOTE_CODE:
        (snapshot / name).write_text(f"# {name}\n", encoding="utf-8")
    fake_import(cache, "configuration.py", "# configuration.py\n")

    with pytest.raises(ArtifactMismatch, match="never imported"):
        verify_remote_code(snapshot, cache, REMOTE_CODE)
