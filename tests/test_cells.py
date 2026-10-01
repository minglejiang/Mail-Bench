"""The cell reader: the one door every result comes through."""

import json

import pytest

from mail_bench.aggregate import AggregationError, load_cells, parse_cell_id


def cell(tmp_path, name, *, task, episode, mode, cameras, onset, success,
         action_hash="a", profile="p1", method="m1", **extra):
    identifier = (
        f"{task}|ep{episode}|{mode}|miss={'+'.join(cameras) or 'none'}"
        f"|onset={onset:g}|dur=end"
    )
    result = {
        "cell_id": identifier,
        "method_id": method,
        "checkpoint_hash": "c" * 64,
        "runner_config_hash": profile,
        "method_config_hash": extra.pop("method_config", "mc"),
        "dataset_authorization_hash": extra.pop("authorization", "auth"),
        "policy_seed": extra.pop("policy_seed", 1),
        "clean_or_fault": "clean" if mode == "healthy" else "fault",
        "success_or_valid": success,
        "action_or_prediction_hash": action_hash,
        # Equal on the real runners; the protocol counts executed actions.
        "action_execution_count": extra.get("steps", 10),
        "environment_step_count": extra.pop("steps", 10),
        "onset_basis": "healthy_reference",
        "onset_fallback": False,
        "fault_cell_valid": True,
        "subset_ranking_eligible": extra.pop("rankable", True),
        "ranking_eligible": True,
    }
    result.update(extra)
    path = tmp_path / f"{name}.json"
    path.write_text(json.dumps({"manifest_hash": "h", "result": result}), encoding="utf-8")
    return path


def certificate(tmp_path, *, visible=("cam0", "cam1"), checkpoint="c" * 64, name="certificate",
                method="m1", method_config="mc", profile="p1", authorization="auth"):
    """The driver writes this next to the cells; the aggregator reads the scope back."""
    (tmp_path / f"{name}.json").write_text(
        json.dumps({
            "measurement_identity": {
                "method_id": method,
                "checkpoint_hash": checkpoint,
                "method_config_hash": method_config,
                "runner_config_hash": profile,
                "dataset_authorization_hash": authorization,
            },
            "policy_declaration": {
                "checkpoint_sha256": checkpoint,
                "policy_visible_cameras": list(visible),
            },
        }),
        encoding="utf-8",
    )


def build(tmp_path, units, *, fault_success, replicate_hashes=("a", "b", "c"), cameras=("cam0",)):
    """A cohort of ``units`` units, each with 3 healthy replicates and one fault cell."""
    for index, (task, episode, healthy_ok) in enumerate(units):
        for replicate in range(3):
            cell(
                tmp_path, f"h{index}_{replicate}", task=task, episode=episode, mode="healthy",
                cameras=(), onset=0.0, success=healthy_ok,
                action_hash=replicate_hashes[replicate], policy_seed=replicate,
            )
        cell(
            tmp_path, f"f{index}", task=task, episode=episode, mode="hard_missing",
            cameras=cameras, onset=0.45, success=fault_success(task, episode),
        )
    certificate(tmp_path)
    return tmp_path


def test_cell_id_round_trips_the_frozen_format():
    parsed = parse_cell_id("task_0|ep3|hard_missing|miss=a+b|onset=0.45|dur=end")
    assert parsed["task"] == "task_0"
    assert parsed["episode_index"] == 3
    assert parsed["faulted_cameras"] == ("a", "b")
    assert parsed["onset_fraction"] == 0.45
    assert parse_cell_id("t|ep0|healthy|miss=none|onset=0|dur=end")["faulted_cameras"] == ()
    with pytest.raises(AggregationError, match="frozen format"):
        parse_cell_id("not a cell id")


def test_audit_failures_are_reported_rather_than_silently_dropped(tmp_path):
    (tmp_path / "broken.json").write_text("{not json", encoding="utf-8")
    build(tmp_path, [("t1", 0, True)], fault_success=lambda *_: True)
    rows, failures = load_cells(tmp_path)
    assert {failure.kind for failure in failures} == {"unreadable_result"}
    # The rest of the cohort still loads; three healthy replicates plus one fault cell.
    assert len(rows) == 4


def test_two_files_claiming_one_identity_are_not_both_counted(tmp_path):
    build(tmp_path, [("t0", 0, True)], fault_success=lambda *_: True)
    copy = (tmp_path / "f0.json").read_text(encoding="utf-8")
    (tmp_path / "f0_again.json").write_text(copy, encoding="utf-8")
    rows, failures = load_cells(tmp_path)
    assert {failure.kind for failure in failures} == {"duplicate_semantic_key"}
    assert len([row for row in rows if not row.is_healthy]) == 1
