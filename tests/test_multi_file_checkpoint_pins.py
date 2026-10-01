"""Pins for checkpoints that are not one file.

A model whose weights are several shards plus auxiliary files cannot be
identified by hashing one file. The roster pins such a checkpoint by the digest
of its verified inventory, and the loader admits it only after those bytes have
been recomputed where the checkpoint is stored.
"""

from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")

ROOT = Path(__file__).resolve().parents[1]


def test_the_mail_roster_baselines_are_admitted_on_their_verified_inventories():
    """The MAIL roster pins multi-file snapshots by inventory digest and one
    checkpoint per model for the one suite. The loader accepts an inventory
    identity for per-suite entries and for suite-wide baselines alike, so
    --model-id resolves both shapes the same way."""
    from mail_bench.cohort import CohortError, load_model_entry
    import yaml
    roster = ROOT / "configs" / "mail_bench_roster.yaml"
    entries = {e["id"]: e for e in yaml.safe_load(roster.read_text())["models"]}
    # Every reference baseline (main_table) resolves its identity from its
    # verified inventory; the set is read from the roster.
    main_table = [e["id"] for e in yaml.safe_load(roster.read_text())["models"]
                  if e.get("main_table")]
    assert main_table, "the roster names no main-table model"
    for model_id in main_table:
        resolved = load_model_entry(roster, model_id, suite="atomic_seen")
        assert resolved.main_table is True
        assert resolved.checkpoint_sha256 == entries[model_id]["inventory_sha256"]
    # An entry not marked main_table is refused as a main-table run, and says so.
    for entry in yaml.safe_load(roster.read_text())["models"]:
        if entry.get("main_table"):
            continue
        with pytest.raises(CohortError, match="not a main-table entry"):
            load_model_entry(roster, entry["id"], suite="atomic_seen")
        assert load_model_entry(roster, entry["id"], suite="atomic_seen",
                                require_main_table=False).main_table is False
    # An inventory whose bytes were never recomputed here is refused.
    doc = yaml.safe_load(roster.read_text())
    next(e for e in doc["models"] if e["id"] == main_table[0])["artifact_verified"] = False
    unverified = ROOT / "results" / "_tmp_unverified_roster.yaml"
    try:
        unverified.write_text(yaml.safe_dump(doc, sort_keys=False, allow_unicode=True))
        with pytest.raises(CohortError, match="not been verified"):
            load_model_entry(unverified, main_table[0], suite="atomic_seen")
    finally:
        unverified.unlink(missing_ok=True)



def test_every_pinned_checkpoint_ships_the_receipt_it_claims():
    """A roster entry that says it was verified must let a reader check it.

    ``verified_receipt`` names the file recording which bytes were recomputed
    where the checkpoint is stored. The receipt travels with the pin, and its digest
    has to be the pin.
    """
    import json
    import yaml

    roster_path = ROOT / "configs" / "mail_bench_roster.yaml"
    roster = yaml.safe_load(roster_path.read_text())
    entries = roster.get("models") or []
    claimed = [e for e in entries if e.get("verified_receipt")]
    assert claimed, "no roster entry names a verification receipt"
    for entry in claimed:
        receipt_path = ROOT / entry["verified_receipt"]
        assert receipt_path.exists(), f"{entry['id']} names a receipt that is not in the repository"
        receipt = json.loads(receipt_path.read_text())["receipt"]
        assert receipt["inventory_sha256"] == entry["inventory_sha256"], (
            f"{entry['id']}: the roster pins one inventory and its receipt records another"
        )
        # The receipt is provenance for anyone, so it carries no path from the
        # machine that produced it.
        assert "/mnt/" not in receipt_path.read_text()
