import subprocess

import pytest

yaml = pytest.importorskip("yaml")

from pathlib import Path

from mail_bench.cohort import (
    CohortError,
    admission_gate,
    audit_cohort_seeds,
    cohort_units,
    load_model_entry,
    preflight,
    require_clean_worktree,
)
from mail_bench.manifest import PROTOCOL_VERSION


ROOT = Path(__file__).resolve().parents[1]
REAL_ROSTER = ROOT / "configs" / "mail_bench_roster.yaml"
ROBOCASA = ["robot0_agentview_left", "robot0_agentview_right", "robot0_eye_in_hand"]


def write_roster(tmp_path, **overrides):
    entry = dict(
        id="tiny_vla",
        display_name="Tiny VLA",
        family="tiny",
        category="multi_view_vla",
        main_table=True,
        adaptation="clean_finetune",
        training_regime="clean_trained",
        checkpoint_scope="all_suites",
        checkpoint_provenance="official",
        policy_visible_cameras=list(ROBOCASA),
        native_action_chunk=8,
        model_revision="rev-1",
        checkpoint_sha256="a" * 64,
        pin_status="pinned",
    )
    entry.update(overrides.pop("entry", {}))
    roster = {"protocol_version": overrides.pop("protocol_version", PROTOCOL_VERSION),
              "models": [entry]}
    path = tmp_path / "roster.yaml"
    path.write_text(yaml.safe_dump(roster), encoding="utf-8")
    return path


def test_published_roster_admits_the_pinned_model():
    entry = load_model_entry(REAL_ROSTER, "pi05_robocasa", suite="atomic_seen")
    assert entry.main_table is True
    assert entry.suite == "atomic_seen"
    assert entry.checkpoint_scope == "all_suites"
    assert entry.training_regime == "clean_trained"
    assert entry.native_action_chunk == 50
    assert entry.native_execution_horizon == 5
    assert entry.policy_visible_cameras == tuple(ROBOCASA)


def test_roster_must_target_this_protocol(tmp_path):
    path = write_roster(tmp_path, protocol_version="mail_bench_perturbation_v1_2")
    with pytest.raises(CohortError, match="this build is"):
        load_model_entry(path, "tiny_vla", suite="atomic_seen")


def test_unpinned_or_unknown_models_are_refused(tmp_path):
    with pytest.raises(CohortError, match="not in the frozen roster"):
        load_model_entry(write_roster(tmp_path), "nobody", suite="atomic_seen")
    path = write_roster(tmp_path, entry={"pin_status": "unpinned"})
    with pytest.raises(CohortError, match="requires a pinned model"):
        load_model_entry(path, "tiny_vla", suite="atomic_seen")
    path = write_roster(tmp_path, entry={"checkpoint_sha256": None})
    with pytest.raises(CohortError, match="pinned checkpoint_sha256"):
        load_model_entry(path, "tiny_vla", suite="atomic_seen")
    path = write_roster(tmp_path, entry={"model_revision": None})
    with pytest.raises(CohortError, match="pinned model_revision"):
        load_model_entry(path, "tiny_vla", suite="atomic_seen")


def test_a_training_regime_is_declared_not_approved(tmp_path):
    # A fault-aware submission runs in the ordinary cohort. There is no
    # enum of permitted regimes and no leaderboard sorting by regime --
    # the declaration is carried through to the result and shown, not judged.
    path = write_roster(tmp_path, entry={"training_regime": "fault_aware"})
    assert load_model_entry(path, "tiny_vla", suite="atomic_seen").training_regime == (
        "fault_aware"
    )
    invented = write_roster(tmp_path, entry={"training_regime": "camera dropout p=0.15"})
    assert load_model_entry(invented, "tiny_vla", suite="atomic_seen").training_regime == (
        "camera dropout p=0.15"
    )
    # An architecture the benchmark has never heard of is admissible.
    custom = write_roster(
        tmp_path, entry={"category": "hybrid_router_policy", "family": "somebody_else"}
    )
    assert load_model_entry(custom, "tiny_vla", suite="atomic_seen").category == (
        "hybrid_router_policy"
    )
    # Saying nothing at all still is not a declaration.
    silent = write_roster(tmp_path, entry={"training_regime": "  "})
    with pytest.raises(CohortError, match="declares no training_regime"):
        load_model_entry(silent, "tiny_vla", suite="atomic_seen")


def test_a_cohort_may_still_ask_for_one_regime(tmp_path):
    # Filtering a roster is an analysis convenience, not an admission rule: the
    # default asks for nothing and runs whatever the entry declares.
    path = write_roster(tmp_path, entry={"training_regime": "fault_aware"})
    assert load_model_entry(path, "tiny_vla", suite="atomic_seen").training_regime == (
        "fault_aware"
    )
    with pytest.raises(CohortError, match="was asked for"):
        load_model_entry(path, "tiny_vla", suite="atomic_seen", track="clean_trained")


def test_non_main_table_entries_need_an_explicit_opt_in(tmp_path):
    path = write_roster(tmp_path, entry={"main_table": False})
    with pytest.raises(CohortError, match="not a main-table entry"):
        load_model_entry(path, "tiny_vla", suite="atomic_seen")
    assert load_model_entry(path, "tiny_vla", suite="atomic_seen", require_main_table=False).main_table is False


def git_repository(tmp_path):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "config", "user.email", "t@example.com"], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "config", "user.name", "t"], check=True)
    (tmp_path / "a.txt").write_text("one\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(tmp_path), "add", "a.txt"], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "commit", "-qm", "init"], check=True)
    return tmp_path


def test_clean_worktree_is_required_and_returns_the_commit(tmp_path):
    repository = git_repository(tmp_path / "repo")
    commit = require_clean_worktree(repository)
    assert len(commit) == 40
    (repository / "a.txt").write_text("two\n", encoding="utf-8")
    with pytest.raises(CohortError, match="worktree is dirty"):
        require_clean_worktree(repository)


def test_seed_projection_collisions_block_a_cohort():
    units = cohort_units([f"task_{i}" for i in range(8)], [0, 1])
    assert audit_cohort_seeds("robocasa", units, target_bits=32).has_collisions is False
    with pytest.raises(CohortError, match="collide at 4 bits"):
        audit_cohort_seeds("robocasa", units, target_bits=4)


def test_cohort_units_reject_duplicates():
    assert cohort_units(["a", "b"], [0, 1]) == (("a", 0), ("a", 1), ("b", 0), ("b", 1))
    with pytest.raises(CohortError, match="task list contains duplicates"):
        cohort_units(["a", "a"], [0])
    with pytest.raises(CohortError, match="episode list contains duplicates"):
        cohort_units(["a"], [0, 0])


PLATFORM_KWARGS = dict(
    benchmark="robocasa",
    benchmark_root=ROOT,
    registry_path=ROOT / "configs" / "dataset_registry.yaml",
    dataset_id="robocasa",
    units=cohort_units(["CloseBlenderLid"], [0]),
    platform_camera_ids=ROBOCASA,
)


def test_preflight_refuses_the_dry_run_stage():
    with pytest.raises(CohortError, match="stage_0 is the mock and dry-run stage"):
        preflight(
            roster_path=REAL_ROSTER, model_id="pi05", suite="atomic_seen",
            stage="stage_0", **PLATFORM_KWARGS,
        )


def test_a_third_party_needs_no_roster_to_pass_the_static_preflight():
    from mail_bench.cohort import platform_preflight

    # The platform layer must be reachable without naming a model at all: being
    # absent from the reference roster is not a reason to refuse an evaluation.
    for call in (
        lambda: platform_preflight(stage="stage_0", **PLATFORM_KWARGS),
        lambda: preflight(stage="stage_0", **PLATFORM_KWARGS),
    ):
        with pytest.raises(CohortError, match="mock and dry-run stage"):
            call()
    with pytest.raises(CohortError, match="camera inventory"):
        platform_preflight(
            stage="stage_1",
            **{**PLATFORM_KWARGS, "platform_camera_ids": []},
        )
    # Naming half a reference baseline is a mistake, not a third-party run.
    with pytest.raises(CohortError, match="roster_path, model_id and suite together"):
        preflight(model_id="pi05", stage="stage_1", **PLATFORM_KWARGS)


def test_admission_gate_reports_per_unit_and_cohort_outcome():
    report = admission_gate(
        [(("t0", 0), True), (("t0", 1), True), (("t1", 0), True), (("t1", 1), False)],
        cohort_target=0.80,
    )
    assert report.admitted_units == (("t0", 0), ("t0", 1), ("t1", 0))
    assert report.rejected_units == (("t1", 1),)
    assert report.clean_success_rate == 0.75
    assert report.meets_cohort_target is False
    assert admission_gate([(("t0", 0), True)], cohort_target=0.80).meets_cohort_target is True
    with pytest.raises(CohortError, match="at least one evaluated unit"):
        admission_gate([])


def test_checkpoint_identity_is_resolved_per_suite(tmp_path):
    path = write_roster(
        tmp_path,
        entry={
            "checkpoint_scope": "per_suite",
            "checkpoint_sha256": None,
            "suite_checkpoints": {
                "suite_a": {"path": "w/spatial.pt", "sha256": "b" * 64,
                                   "pin_status": "pinned"},
                "suite_b": {"path": "w/goal.pt", "sha256": "c" * 64,
                                "pin_status": "unpinned"},
            },
        },
    )
    spatial = load_model_entry(path, "tiny_vla", suite="suite_a")
    assert spatial.checkpoint_sha256 == "b" * 64
    assert spatial.checkpoint_path == "w/spatial.pt"
    # A suite whose weights are not locally verified may not run.
    with pytest.raises(CohortError, match="requires a locally verified pin"):
        load_model_entry(path, "tiny_vla", suite="suite_b")
    # A suite with no published checkpoint at all may not run either.
    with pytest.raises(CohortError, match="has none pinned for suite"):
        load_model_entry(path, "tiny_vla", suite="suite_c")


def test_unresolved_checkpoint_scope_is_refused(tmp_path):
    path = write_roster(tmp_path, entry={"checkpoint_scope": "unknown_pending_verification"})
    with pytest.raises(CohortError, match="checkpoint_scope"):
        load_model_entry(path, "tiny_vla", suite="atomic_seen")


def test_declared_upstream_dependencies_must_be_pinned_too(tmp_path):
    base = {"repo": "vendor/backbone", "revision": "r1", "pin_status": "unpinned"}
    path = write_roster(tmp_path, entry={"base_model": dict(base)})
    with pytest.raises(CohortError, match="identity is incomplete without it"):
        load_model_entry(path, "tiny_vla", suite="atomic_seen")
    base["pin_status"] = "pinned"
    path = write_roster(tmp_path, entry={"base_model": base})
    entry = load_model_entry(path, "tiny_vla", suite="atomic_seen")
    assert entry.dependency_revisions == (("vendor/backbone", "r1"),)


def test_execution_profiles_are_distinct_identities(tmp_path):
    from mail_bench.cohort import ExecutionProfile
    from mail_bench.manifest import semantic_sha256
    from mail_bench.runner import RunnerConfig

    entry = load_model_entry(REAL_ROSTER, "pi05_robocasa", suite="atomic_seen")
    native = ExecutionProfile.native(entry)
    assert native.name == "native" and native.max_actions_per_query == 5
    matched = ExecutionProfile.matched(8)
    assert matched.name == "matched_replan_8"
    assert ExecutionProfile.parse("matched_replan_8", entry) == matched
    assert ExecutionProfile.parse("native", entry) == native
    with pytest.raises(CohortError, match="unknown execution profile"):
        ExecutionProfile.parse("fastest", entry)
    with pytest.raises(CohortError, match=">= 1"):
        ExecutionProfile.matched(0)

    # The profile must reach the execution identity, or a native run and a
    # matched-replanning run would be recorded as the same cell.
    configs = [
        RunnerConfig(max_environment_steps=220, max_actions_per_query=profile.max_actions_per_query)
        for profile in (native, matched)
    ]
    assert semantic_sha256(configs[0]) != semantic_sha256(configs[1])


def test_native_profile_requires_a_declared_horizon(tmp_path):
    from mail_bench.cohort import ExecutionProfile

    path = write_roster(tmp_path, entry={"native_execution_horizon": None})
    entry = load_model_entry(path, "tiny_vla", suite="atomic_seen")
    with pytest.raises(CohortError, match="native_execution_horizon"):
        ExecutionProfile.native(entry)


def test_fault_grid_is_expanded_from_the_frozen_subset_rules():
    from mail_bench.cohort import fault_cell_specs

    units = cohort_units(["t0", "t1"], [0, 1])
    specs = fault_cell_specs(
        units,
        platform_camera_ids=ROBOCASA,
        onset_fractions=[0.30, 0.45, 0.60],
        semantic_platform="robocasa365",
    )
    # 4 units x 3 missing states x 3 onsets: the nine ranking conditions per unit.
    assert len(specs) == 36
    assert len({spec.faulted_cameras for spec in specs}) == 3
    total_loss = [spec for spec in specs if len(spec.faulted_cameras) == 3]
    assert len(total_loss) == 12
    # Without a resolved policy scope every cell is noted as such; the grid
    # enumerates every cell regardless, and every one of them is ranked.
    assert {spec.scope_note for spec in specs} == {"policy_scope_unresolved"}
    with pytest.raises(CohortError, match="onset fraction"):
        fault_cell_specs(units, platform_camera_ids=ROBOCASA, onset_fractions=[1.0],
                         semantic_platform="robocasa365")
    with pytest.raises(CohortError, match="semantic platform"):
        fault_cell_specs(units, platform_camera_ids=ROBOCASA, onset_fractions=[0.45])


def test_worker_shards_partition_the_units_exactly():
    from mail_bench.cohort import shard_units

    units = cohort_units([f"t{i}" for i in range(5)], [0, 1, 2])
    shards = [shard_units(units, workers=3, worker_index=i) for i in range(3)]
    flattened = [unit for shard in shards for unit in shard]
    assert sorted(flattened) == sorted(units)          # complete
    assert len(flattened) == len(set(flattened))       # non-overlapping
    assert max(len(shard) for shard in shards) - min(len(shard) for shard in shards) <= 1
    # One worker sees every unit; sharding is scheduling, not identity.
    assert shard_units(units, workers=1, worker_index=0) == units
    with pytest.raises(CohortError, match="worker_index"):
        shard_units(units, workers=3, worker_index=3)
    with pytest.raises(CohortError, match="duplicates"):
        shard_units((("t0", 0), ("t0", 0)), workers=1, worker_index=0)


def test_the_fault_grid_is_a_platform_property_not_a_policy_property():
    from mail_bench.cohort import fault_cell_specs

    from mail_bench.semantic_states import state_cameras

    units = cohort_units(["t0"], [0])
    platform = ROBOCASA
    wrist, agentview, everything = (state_cameras("robocasa365", s) for s in
                                    ("wrist_missing", "agentview_missing", "all_vision_missing"))
    everyone = fault_cell_specs(units, platform_camera_ids=platform, onset_fractions=[0.45],
                                semantic_platform="robocasa365")
    wrist_only = fault_cell_specs(
        units,
        platform_camera_ids=platform,
        policy_visible_cameras=["robot0_eye_in_hand"],
        onset_fractions=[0.45],
        semantic_platform="robocasa365",
    )
    # A single-view policy faces exactly the same schedule as everyone else.
    assert [spec.faulted_cameras for spec in everyone] == [
        spec.faulted_cameras for spec in wrist_only
    ]

    def note(specs, cameras):
        return next(s for s in specs if s.faulted_cameras == cameras).scope_note

    # Losing cameras it never reads is a no-op for this design: noted as
    # provenance, executed, reported and ranked like every other cell.
    assert note(wrist_only, agentview) == "no_op_for_declared_scope"
    assert note(wrist_only, wrist) is None
    # Total visual loss takes the one camera this policy reads.
    assert note(wrist_only, everything) is None
    # Without a declared scope the note says so: unresolved is not "all cameras".
    assert note(everyone, agentview) == "policy_scope_unresolved"
    resolved = fault_cell_specs(
        units,
        platform_camera_ids=platform,
        policy_visible_cameras=platform,
        onset_fractions=[0.45],
        semantic_platform="robocasa365",
    )
    # A router declares both cameras as its upper bound, so both are rankable
    # even if a healthy rollout happened to lean on one of them.
    assert note(resolved, agentview) is None
    assert note(resolved, wrist) is None
    assert note(resolved, everything) is None


def test_policy_scope_must_be_a_subset_of_the_platform():
    from mail_bench.cohort import fault_cell_specs

    units = cohort_units(["t0"], [0])
    with pytest.raises(CohortError, match="does not provide"):
        fault_cell_specs(
            units,
            platform_camera_ids=ROBOCASA,
            policy_visible_cameras=ROBOCASA + ["lidar"],
            semantic_platform="robocasa365",
            onset_fractions=[0.45],
        )


def test_matched_profile_cannot_exceed_the_predicted_chunk():
    from mail_bench.cohort import ExecutionProfile

    entry = load_model_entry(REAL_ROSTER, "pi05_robocasa", suite="atomic_seen")
    assert entry.native_action_chunk == 50
    assert ExecutionProfile.matched(5, predicted_action_chunk=50).max_actions_per_query == 5
    with pytest.raises(CohortError, match="exceeds the 50 actions"):
        ExecutionProfile.parse("matched_replan_100", entry)
    # Native must also be consistent with the chunk the model produces.
    assert ExecutionProfile.native(entry).name == "native"


def server_identity(**overrides):
    base = dict(
        protocol="mcr-anything-v1",
        model_id="somebody/their-own-router",
        suite="atomic_seen",
        checkpoint_sha256="d" * 64,
        policy_visible_cameras=list(ROBOCASA),
        availability_consumed_by_policy=True,
        predicted_action_chunk=16,
        native_execution_horizon=4,
        action_dimension=7,
        stateful_policy=True,
        reset_semantics="clears_all_state",
        training_regime="clean_trained",
    )
    base.update(overrides)
    return base


def test_a_third_party_policy_resolves_without_appearing_in_any_roster():
    from mail_bench.cohort import execution_profile_for, resolve_runtime_contract

    # No roster entry: the reference roster is this project's baseline registry,
    # not a membership requirement for evaluation.
    declaration = resolve_runtime_contract(
        server_identity(),
        platform_camera_ids=ROBOCASA,
        suite="atomic_seen",
    )
    assert declaration.model_id == "somebody/their-own-router"
    assert execution_profile_for("native", declaration).max_actions_per_query == 4
    with pytest.raises(CohortError, match="exceeds the 16 actions"):
        execution_profile_for("matched_replan_20", declaration)


def test_the_server_and_the_roster_must_agree_before_any_rollout():
    from mail_bench.cohort import resolve_runtime_contract

    entry = load_model_entry(REAL_ROSTER, "pi05_robocasa", suite="atomic_seen")
    served = server_identity(
        model_id="pi05_robocasa",
        checkpoint_sha256=entry.checkpoint_sha256,
        predicted_action_chunk=50,
        native_execution_horizon=5,
        availability_consumed_by_policy=False,
        stateful_policy=False,
        reset_semantics="stateless",
    )
    resolved = resolve_runtime_contract(
        served,
        platform_camera_ids=ROBOCASA,
        suite="atomic_seen",
        entry=entry,
    )
    assert resolved.predicted_action_chunk == 50
    with pytest.raises(CohortError, match="the roster declares"):
        resolve_runtime_contract(
            dict(served, predicted_action_chunk=20),
            platform_camera_ids=ROBOCASA,
            entry=entry,
        )
    with pytest.raises(CohortError, match="frozen entry pins"):
        resolve_runtime_contract(
            dict(served, checkpoint_sha256="e" * 64),
            platform_camera_ids=ROBOCASA,
            entry=entry,
        )
    with pytest.raises(CohortError, match="serves suite"):
        resolve_runtime_contract(
            dict(served, suite="suite_b"),
            platform_camera_ids=ROBOCASA,
            suite="atomic_seen",
            entry=entry,
        )


def test_an_incomplete_declaration_cannot_start_a_cohort():
    from mail_bench.cohort import resolve_runtime_contract

    with pytest.raises(CohortError, match="not admissible"):
        resolve_runtime_contract(
            server_identity(policy_visible_cameras=None),
            platform_camera_ids=ROBOCASA,
        )


def test_unresolved_policy_scope_makes_every_cell_unrankable():
    from mail_bench.cohort import fault_cell_specs

    specs = fault_cell_specs(
        cohort_units(["t0"], [0]),
        platform_camera_ids=ROBOCASA,
        onset_fractions=[0.45],
        semantic_platform="robocasa365",
    )
    # None means unresolved, never "all cameras": the note records that the
    # cohort was planned before anyone established what the policy reads.
    assert {spec.scope_note for spec in specs} == {"policy_scope_unresolved"}


def test_the_regime_a_server_declares_must_match_the_frozen_entry():
    from mail_bench.cohort import resolve_runtime_contract

    entry = load_model_entry(REAL_ROSTER, "pi05_robocasa", suite="atomic_seen")
    served = server_identity(
        model_id="pi05_robocasa",
        checkpoint_sha256=entry.checkpoint_sha256,
        predicted_action_chunk=50,
        native_execution_horizon=5,
        availability_consumed_by_policy=False,
        stateful_policy=False,
        reset_semantics="stateless",
        training_regime="fault_aware",
    )
    # Neither answer is refused, but a result must be published under the regime
    # that actually ran: the roster and the running server are two halves of one
    # identity, and a mismatch means one of them is wrong.
    with pytest.raises(CohortError, match="must be published under the regime that ran"):
        resolve_runtime_contract(
            served,
            platform_camera_ids=ROBOCASA,
            suite="atomic_seen",
            entry=entry,
        )


def test_identity_fields_must_be_well_formed():
    from mail_bench.cohort import resolve_runtime_contract

    for bad, pattern in (
        ({"checkpoint_sha256": "not-a-digest"}, "64 hexadecimal"),
        ({"model_id": "   "}, "model_id must be a non-empty"),
        ({"protocol": ""}, "protocol must be a non-empty"),
    ):
        with pytest.raises(CohortError, match=pattern):
            resolve_runtime_contract(
                server_identity(**bad), platform_camera_ids=ROBOCASA
            )
    # Hex case is canonicalised so a roster comparison cannot fail on case alone.
    resolved = resolve_runtime_contract(
        server_identity(checkpoint_sha256="A" * 64),
        platform_camera_ids=ROBOCASA,
    )
    assert resolved.checkpoint_sha256 == "a" * 64


def test_the_platforms_action_dimension_is_checked_at_admission():
    """A server declaring seven-dimensional actions is refused at admission rather
    than failing at the first step of the first healthy cell inside convert_action."""
    from mail_bench.cohort import resolve_runtime_contract

    served = server_identity(action_dimension=7)
    resolve_runtime_contract(served, platform_camera_ids=ROBOCASA, platform_action_dimension=7)
    with pytest.raises(CohortError, match="the platform executes 12-dimensional actions"):
        resolve_runtime_contract(served, platform_camera_ids=ROBOCASA,
                                 platform_action_dimension=12)

