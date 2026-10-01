import json

import pytest

from mail_bench.manifest import (
    PROTOCOL_VERSION,
    FaultManifest,
    ManifestError,
    ResultError,
    ResultRecord,
    canonical_json,
    semantic_sha256,
    validate_manifest,
    validate_result,
)


def make_manifest(**overrides):
    base = dict(
        dataset_version="fixture-v1",
        split="test",
        scene_or_task="task_a",
        episode_or_sequence=3,
        camera_ids=["cam0", "cam1", "cam2"],
        faulted_camera_ids=["cam0"],
        perturbation_family="availability",
        fault_mode="hard_missing",
        onset_fraction=0.3,
        duration_fraction=1.0,
        environment_seed=1,
        policy_seed=2,
        fault_seed=3,
        official_horizon=400,
        adapter_version="fake-adapter-v1",
        protocol_version=PROTOCOL_VERSION,
        onset_step=120,
        onset_basis="healthy_reference",
        healthy_reference_success=True,
        onset_reference_hash="a" * 64,
        onset_fallback=False,
        subset_ranking_eligible=True,
    )
    base.update(overrides)
    return FaultManifest(**base)


def test_canonical_json_key_order_and_floats():
    a = {"b": 1.0, "a": [0.1, {"z": 2, "y": frozenset({"q", "p"})}]}
    b = {"a": [0.1, {"y": frozenset({"p", "q"}), "z": 2}], "b": 1.0}
    assert canonical_json(a) == canonical_json(b) == '{"a":[0.1,{"y":["p","q"],"z":2}],"b":1.0}'
    assert semantic_sha256(a) == semantic_sha256(b)
    with pytest.raises(ValueError):
        canonical_json({"x": float("nan")})
    with pytest.raises(TypeError):
        canonical_json({"x": object()})


def test_manifest_hash_stable_across_key_order():
    m1 = make_manifest()
    m2 = FaultManifest(**dict(reversed(list(m1.to_dict().items()))))
    assert m1.semantic_hash() == m2.semantic_hash()
    assert semantic_sha256(json.loads(canonical_json(m1))) == semantic_sha256(m1)
    assert make_manifest(onset_fraction=0.45).semantic_hash() != m1.semantic_hash()


def test_validate_manifest_ok_and_cell_id():
    m = validate_manifest(make_manifest())
    assert m.cell_id and "miss=cam0" in m.cell_id and "onset=0.3" in m.cell_id


def test_validate_manifest_missing_required():
    with pytest.raises(ManifestError, match="policy_seed"):
        validate_manifest(make_manifest(policy_seed=None))


def test_validate_manifest_onset_inconsistent_with_horizon():
    with pytest.raises(ManifestError, match="onset_step"):
        validate_manifest(make_manifest(onset_step=400))
    with pytest.raises(ManifestError, match="end_step"):
        validate_manifest(make_manifest(onset_step=10, end_step=10))


def test_validate_manifest_unknown_mode_and_bad_cameras():
    with pytest.raises(ManifestError, match="fault_mode"):
        validate_manifest(make_manifest(fault_mode="teleport"))
    with pytest.raises(ManifestError, match="faulted_camera_ids"):
        validate_manifest(make_manifest(faulted_camera_ids=["nose"]))
    with pytest.raises(ManifestError, match="stale_k"):
        validate_manifest(make_manifest(fault_mode="stale_k"))
    with pytest.raises(ManifestError, match="alias"):
        validate_manifest(make_manifest(fault_mode="stale", stale_k=3))
    assert validate_manifest(make_manifest(fault_mode="stale_k", stale_k=3)).stale_k == 3


def make_result(m, **overrides):
    base = dict(
        cell_id=m.cell_id, method_id="policy", checkpoint_hash="abc", clean_or_fault="fault",
        official_metrics={"success": 1.0}, success_or_valid=True, completed=True,
        environment_step_count=300, policy_query_count=60, action_execution_count=300,
        faulted_observations_produced=180, faulted_observations_consumed=180,
        environment_seed=1, policy_seed=2, fault_seed=3,
        onset_basis=m.onset_basis, healthy_reference_success=m.healthy_reference_success,
        fault_reached=True, onset_reference_hash=m.onset_reference_hash,
        onset_fallback=m.onset_fallback,
    )
    base.update(overrides)
    return ResultRecord(**base)


def test_validate_result_ok():
    m = validate_manifest(make_manifest())
    r = validate_result(make_result(m, manifest_hash=m.semantic_hash()), m)
    assert r.fault_cell_valid is True and r.invalid_reason is None
    assert r.recovery_eligible is True and r.ranking_eligible is True


def test_validate_result_rejects_mismatch():
    m = validate_manifest(make_manifest())
    with pytest.raises(ResultError, match="cell_id"):
        validate_result(make_result(m, cell_id="other"), m)
    with pytest.raises(ResultError, match="policy_seed"):
        validate_result(make_result(m, policy_seed=99), m)
    with pytest.raises(ResultError, match="manifest_hash"):
        validate_result(make_result(m, manifest_hash="00"), m)
    with pytest.raises(ResultError, match="clean_or_fault"):
        validate_result(make_result(m, clean_or_fault="oops"), m)


def test_validate_result_zero_consumed_marks_invalid():
    m = validate_manifest(make_manifest())
    r = validate_result(make_result(m, faulted_observations_consumed=0), m)
    assert r.fault_cell_valid is False and "zero" in r.invalid_reason
    assert r.fault_reached is True and r.recovery_eligible is True
    assert r.ranking_eligible is False
    r2 = validate_result(make_result(m, faulted_observations_consumed=None), m)
    assert r2.fault_cell_valid is False


def test_result_rejects_impossible_produced_consumed_counts():
    m = validate_manifest(make_manifest())
    with pytest.raises(ResultError, match="cannot exceed"):
        validate_result(make_result(m, faulted_observations_produced=1,
                                    faulted_observations_consumed=2), m)
    with pytest.raises(ResultError, match="before the fault is reached"):
        validate_result(make_result(m, fault_reached=False,
                                    faulted_observations_produced=1,
                                    faulted_observations_consumed=0), m)
    with pytest.raises(ResultError, match="must be an integer"):
        validate_result(make_result(m, faulted_observations_produced="1"), m)
    with pytest.raises(ResultError, match="integer or None"):
        validate_result(make_result(m, faulted_observations_consumed=1.0), m)


def test_onset_audit_consistency_and_fault_reached_vs_consumed():
    with pytest.raises(ManifestError, match="onset_fallback"):
        validate_manifest(make_manifest(onset_fallback=True))
    with pytest.raises(ManifestError, match="SHA256"):
        validate_manifest(make_manifest(onset_reference_hash="not-a-hash"))
    with pytest.raises(ManifestError, match="protocol_version"):
        validate_manifest(make_manifest(protocol_version="mail_bench_perturbation_v1_1"))
    # A fault cell only exists behind a successful healthy rollout, so the
    # "fallback onset" is refused outright.
    with pytest.raises(ManifestError, match="healthy_reference_success"):
        validate_manifest(make_manifest(
            onset_basis="healthy_reference", healthy_reference_success=False,
        ))
    cell = validate_manifest(make_manifest(
        onset_basis="healthy_reference",
        healthy_reference_success=True,
        subset_ranking_eligible=True,
    ))
    reached_not_consumed = validate_result(
        make_result(cell, healthy_reference_success=True, fault_reached=True,
                    faulted_observations_consumed=0),
        cell,
    )
    # The healthy reference succeeded and the fault was reached, so the cell meets
    # the recovery precondition -- but it consumed no faulted observation, so it is
    # invalid and cannot be ranked. Validity is what disqualifies it, not the onset.
    assert reached_not_consumed.fault_cell_valid is False
    assert reached_not_consumed.invalid_reason == (
        "fault arm consumed zero faulted observations")
    assert reached_not_consumed.ranking_eligible is False
    with pytest.raises(ResultError, match="before the fault is reached"):
        validate_result(make_result(cell, healthy_reference_success=True,
                                    fault_reached=False,
                                    faulted_observations_consumed=1), cell)
    with pytest.raises(ResultError, match="healthy_reference_success mismatch"):
        validate_result(make_result(cell, healthy_reference_success=False), cell)
    with pytest.raises(ResultError, match="fault_reached must be boolean"):
        validate_result(make_result(cell, healthy_reference_success=True,
                                    fault_reached="yes",
                                    faulted_observations_consumed=0), cell)


def test_total_visual_loss_is_a_ranked_condition():
    """MAIL-Bench ranks it.

    Three of the ten conditions are total visual loss. A policy that keeps
    working with no camera at all is telling us how much it was using vision,
    and that belongs in the score rather than beside it as an unranked floor.

    Ranking eligibility is the caller's to declare -- it depends on the policy's
    camera scope, which a manifest cannot see -- so the faulted set does not
    force it either way.
    """
    every_camera = ["cam0", "cam1", "cam2"]
    ranked = validate_manifest(make_manifest(
        faulted_camera_ids=every_camera, subset_ranking_eligible=True,
    ))
    result = validate_result(make_result(ranked), ranked)
    assert result.fault_cell_valid is True
    assert result.subset_ranking_eligible is True
    # A diagnostic combination outside the ranked grid is still expressible.
    unranked = validate_manifest(make_manifest(
        faulted_camera_ids=every_camera, subset_ranking_eligible=False,
    ))
    assert validate_result(make_result(unranked), unranked).ranking_eligible is False


def healthy_overrides(**overrides):
    base = dict(
        faulted_camera_ids=[],
        fault_mode="healthy",
        onset_fraction=0.0,
        onset_step=None,
        end_step=None,
        onset_basis="not_applicable",
        healthy_reference_success=False,
        onset_fallback=False,
        subset_ranking_eligible=False,
    )
    base.update(overrides)
    return base


def test_healthy_cell_is_a_first_class_availability_cell():
    healthy = validate_manifest(make_manifest(**healthy_overrides()))
    assert healthy.cell_id == (
        "task_a|ep3|healthy|miss=none|onset=0|dur=end"
    )
    assert healthy.onset_step is None
    assert healthy.subset_ranking_eligible is False


def test_healthy_mode_and_empty_subset_must_agree():
    with pytest.raises(ManifestError, match="exactly when no camera is faulted"):
        validate_manifest(make_manifest(fault_mode="healthy"))
    with pytest.raises(ManifestError, match="exactly when no camera is faulted"):
        validate_manifest(make_manifest(**healthy_overrides(fault_mode="hard_missing")))


def test_healthy_cell_rejects_onset_provenance_and_ranking():
    with pytest.raises(ManifestError, match="onset_basis 'not_applicable'"):
        validate_manifest(
            make_manifest(
                **healthy_overrides(
                    onset_basis="healthy_reference", healthy_reference_success=True
                )
            )
        )
    with pytest.raises(ManifestError, match="cannot claim onset provenance"):
        validate_manifest(make_manifest(**healthy_overrides(onset_fallback=True)))
    with pytest.raises(ManifestError, match="must not resolve an onset_step"):
        validate_manifest(make_manifest(**healthy_overrides(onset_step=120)))
    with pytest.raises(ManifestError, match="never a ranked camera subset"):
        validate_manifest(make_manifest(**healthy_overrides(subset_ranking_eligible=True)))


def test_not_applicable_onset_basis_is_reserved_for_healthy_cells():
    with pytest.raises(ManifestError, match="reserved for healthy cells"):
        validate_manifest(
            make_manifest(onset_basis="not_applicable", healthy_reference_success=False)
        )


def test_clean_or_fault_must_match_the_manifest_fault_mode():
    # A fault rollout labelled clean would skip every fault audit field and land
    # in the healthy denominator of clean_solvable / recovery_fraction.
    fault = validate_manifest(make_manifest())
    with pytest.raises(ResultError, match="inconsistent with manifest fault_mode"):
        validate_result(make_result(fault, clean_or_fault="clean"), fault)

    healthy = validate_manifest(make_manifest(**healthy_overrides()))
    with pytest.raises(ResultError, match="inconsistent with manifest fault_mode"):
        validate_result(
            make_result(healthy, clean_or_fault="fault"),
            healthy,
        )
    assert validate_result(
        make_result(healthy, clean_or_fault="clean"), healthy
    ).fault_cell_valid is True


def test_result_schema_version_is_separate_from_the_protocol_version():
    from mail_bench.manifest import PROTOCOL_VERSION, RESULT_SCHEMA_VERSION

    # The version this build speaks, asserted so a bump is a deliberate edit
    # rather than something that slips through with the fixtures.
    assert PROTOCOL_VERSION == "mail_bench_perturbation_v1"
    m = validate_manifest(make_manifest())
    assert validate_result(make_result(m), m).result_schema_version == RESULT_SCHEMA_VERSION
    # An observability-only bump must not orphan cells written by an older build.
    older = validate_result(make_result(m, result_schema_version=1), m)
    assert older.result_schema_version == 1
    # A result written by a newer build must fail loudly rather than be misread.
    with pytest.raises(ResultError, match="not readable by this build"):
        validate_result(make_result(m, result_schema_version=RESULT_SCHEMA_VERSION + 1), m)
    with pytest.raises(ResultError, match="must be an integer"):
        validate_result(make_result(m, result_schema_version="2"), m)
