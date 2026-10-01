"""End-to-end orchestration checks with no simulator and no GPU.

These assert the *order* the protocol requires and the identity rules that keep
results attributable, using fake environments and fake policies so the whole
cohort path can be exercised cheaply.
"""

import dataclasses

import pytest

from mail_bench.cohort import (
    CohortError,
    ExecutionProfile,
    PlatformPreflight,
    cohort_units,
    shard_units,
)
from mail_bench.driver import CohortRunner, CohortSpecification, cohort_admission_status
from mail_bench.experiment import MethodArmExecution, PairedCellRunner
from mail_bench.interfaces import (
    CanonicalObservation,
    EnvironmentAdapter,
    EnvironmentStep,
    PolicyAdapter,
    PolicyOutput,
)
from mail_bench.io import AtomicResultWriter
from mail_bench.manifest import PROTOCOL_VERSION, FaultManifest
from mail_bench.policy_api.contract import validate_declaration
from mail_bench.registry import validate_dataset
from mail_bench.runner import RunnerConfig
from mail_bench.seeds import audit_seed_projection


CAMERAS = ("robot0_agentview_left", "robot0_agentview_right", "robot0_eye_in_hand")
WRIST = ("robot0_eye_in_hand",)
AGENTVIEW = ("robot0_agentview_left", "robot0_agentview_right")
EVERYTHING = CAMERAS
HORIZON = 20


class FakeEnvironment(EnvironmentAdapter):
    def __init__(self, success_at=8):
        self.success_at = success_at
        self.step_index = 0

    def _observation(self):
        return CanonicalObservation(
            step=self.step_index,
            cameras={camera: (camera, self.step_index) for camera in CAMERAS},
            availability={camera: True for camera in CAMERAS},
            robot_state=(self.step_index,),
        )

    def reset(self, environment_seed):
        self.step_index = 0
        return self._observation()

    def step(self, action):
        self.step_index += 1
        return EnvironmentStep(self._observation(), terminated=self.success())

    def camera_inventory(self):
        return CAMERAS

    def official_metrics(self):
        return {"success": self.success(), "control_steps": self.step_index}

    def success(self):
        return self.success_at is not None and self.step_index >= self.success_at


class DeterministicPolicy(PolicyAdapter):
    """Same action chunk whatever the policy seed, like a single-forward-pass WAM."""

    chunk = 4

    def reset(self, policy_seed):
        self.seed = policy_seed

    def act(self, observation):
        return PolicyOutput(tuple(("act", index) for index in range(self.chunk)))


class StochasticPolicy(DeterministicPolicy):
    """Action chunk depends on the seed, like a sampling VLA."""

    def act(self, observation):
        return PolicyOutput(tuple(("act", self.seed % 7, index) for index in range(self.chunk)))


def platform_preflight_stub():
    return PlatformPreflight(
        protocol_version=PROTOCOL_VERSION,
        dataset=validate_dataset({}, None, stage="stage_0", mock=True),
        benchmark_commit="0" * 40,
        seed_projection=audit_seed_projection([1, 2, 3], 32),
        unit_count=2,
        platform_camera_ids=CAMERAS,
    )


def declaration(visible=CAMERAS, track="clean_trained"):
    return validate_declaration(
        {
            "protocol": "mcr-fake-v1",
            "model_id": "somebody/fake",
            "checkpoint_sha256": "a" * 64,
            "policy_visible_cameras": list(visible),
            "availability_consumed_by_policy": False,
            "predicted_action_chunk": 4,
            "native_execution_horizon": 2,
            "action_dimension": 7,
            "stateful_policy": False,
            "reset_semantics": "stateless",
            "training_regime": track,
        },
        environment_cameras=CAMERAS,
    )


def specification(profile="native", onsets=(0.5,)):
    return CohortSpecification(
        benchmark="fake",
        suite="fake_suite",
        dataset_id="fake",
        dataset_revision="rev",
        tasks=("t0", "t1"),
        episode_indices=(0,),
        platform_camera_ids=CAMERAS,
        onset_fractions=onsets,
        official_horizon=HORIZON,
        execution_profile=profile,
    )


def make_runner(
    tmp_path,
    *,
    policy_class=DeterministicPolicy,
    visible=CAMERAS,
    profile_spec="native",
    success_at=8,
    official_profile=None,
    enforce_admission=None,
    order_log=None,
    healthy_replicates=1,
):
    spec = specification(profile=profile_spec)
    decl = declaration(visible=visible)
    profile = ExecutionProfile.matched(
        decl.native_execution_horizon, predicted_action_chunk=decl.predicted_action_chunk
    ).renamed("native")
    if profile_spec != "native":
        profile = ExecutionProfile.matched(
            int(profile_spec.rsplit("_", 1)[1]),
            predicted_action_chunk=decl.predicted_action_chunk,
        )

    def template(task, episode_index, faulted, onset_fraction):
        return FaultManifest(
            dataset_version="rev",
            split="stage_0",
            scene_or_task=task,
            episode_or_sequence=episode_index,
            camera_ids=list(CAMERAS),
            faulted_camera_ids=list(faulted),
            perturbation_family="availability",
            fault_mode="hard_missing",
            onset_fraction=onset_fraction,
            duration_fraction=1.0,
            environment_seed=1,
            policy_seed=2,
            fault_seed=3,
            official_horizon=HORIZON,
            adapter_version="fake-v1",
            protocol_version=PROTOCOL_VERSION,
            onset_basis="not_applicable",
            healthy_reference_success=False,
            onset_reference_hash="b" * 64,
            onset_fallback=False,
            subset_ranking_eligible=True,
        )

    def environment_factory(manifest):
        if order_log is not None:
            order_log.append(("clean" if manifest.fault_mode == "healthy" else "fault",
                              tuple(manifest.faulted_camera_ids or ())))
        return FakeEnvironment(success_at=success_at)

    paired = PairedCellRunner(
        environment_factory=environment_factory,
        policy_factory=lambda arm, manifest: policy_class(),
        config=RunnerConfig(
            max_environment_steps=HORIZON,
            max_actions_per_query=profile.max_actions_per_query,
        ),
        execution=validate_dataset({}, None, stage="stage_0", mock=True),
        writer=AtomicResultWriter(tmp_path),
    )
    return CohortRunner(
        specification=spec,
        healthy_replicates=healthy_replicates,
        platform=platform_preflight_stub(),
        declaration=decl,
        profile=profile,
        paired=paired,
        arm=MethodArmExecution.from_config("fake", "c" * 64, {"policy": policy_class.__name__}),
        manifest_template=template,
        official_profile=official_profile,
        enforce_admission=enforce_admission,
        semantic_platform="robocasa365",
    )


UNITS = (("t0", 0), ("t1", 0))


def test_healthy_always_runs_before_any_fault_cell_of_that_unit(tmp_path):
    order = []
    report = make_runner(tmp_path, order_log=order, healthy_replicates=3).run(UNITS)
    assert len(report.units) == 2
    # Three healthy cells (a replicate probe), then the fault cells, unit by unit.
    kinds = [kind for kind, _ in order]
    assert kinds[:3] == ["clean"] * 3
    first_fault = kinds.index("fault")
    assert set(kinds[:first_fault]) == {"clean"}


def unit_success(tmp_path, solvable_units, total_units, **kwargs):
    """A cohort where the first ``solvable_units`` units have a solvable scene."""
    import dataclasses

    units = tuple((f"t{i}", 0) for i in range(total_units))
    solvable = {units[i] for i in range(solvable_units)}
    runner = make_runner(tmp_path, **kwargs)
    # The specification must describe what is actually being run, or every
    # cohort would look like a deviation from its own task list.
    runner.specification = dataclasses.replace(
        runner.specification, tasks=tuple(task for task, _ in units)
    )
    state = {"unit": None}
    outer_run_unit = runner._healthy_for

    def tracked(task, episode_index):
        state["unit"] = (task, episode_index)
        return outer_run_unit(task, episode_index)

    runner._healthy_for = tracked
    runner.paired.environment_factory = lambda manifest: FakeEnvironment(
        success_at=8 if state["unit"] in solvable else None
    )
    return runner, units


def test_an_unsolvable_unit_gets_no_fault_cells_at_all(tmp_path):
    # 4 of 5 solvable. The unsolvable unit has no successful healthy trajectory,
    # so 30, 45 and 60 percent of it do not exist: the benchmark runs none of its fault
    # cells rather than anchoring them to the horizon. That is not the same as
    # scoring them as failures, and being unsolvable is still not an admission
    # failure for the cohort.
    runner, units = unit_success(tmp_path, 4, 5, official_profile=official_profile())
    report = runner.run(units)
    assert report.admission.clean_success_rate == pytest.approx(0.8)
    assert cohort_admission_status(report.admission) == "admitted"
    unsolvable = [outcome for outcome in report.units if not outcome.clean_solvable]
    assert len(unsolvable) == 1
    outcome = unsolvable[0]
    assert outcome.fault_cells == []
    assert outcome.skipped_fault_cells == 3
    assert outcome.fault_skip_reason == "healthy_unsolved"
    # The solvable units are untouched by this.
    solvable = [o for o in report.units if o.clean_solvable]
    assert all(len(o.fault_cells) == 3 for o in solvable)

def official_profile():
    from mail_bench.driver import OfficialProfile

    return OfficialProfile(
        suite="fake_suite",
        tasks=tuple(f"t{i}" for i in range(5)),
        episode_indices=(0,),
        onset_fractions=(0.5,),
        platform_camera_ids=CAMERAS,
        official_horizon=HORIZON,
        dataset_id="fake",
        dataset_revision="rev",
    )


def test_a_custom_experiment_runs_and_records_what_it_changed(tmp_path):
    # Two tasks and one onset is a perfectly good experiment; the benchmark runs
    # it and simply says it is not comparable to the frozen leaderboard.
    runner, units = unit_success(tmp_path, 2, 2, official_profile=official_profile())
    report = runner.run(units)
    certificate = report.certificate
    assert certificate["result_valid"] is True
    assert certificate["official_protocol_conformant"] is False
    assert certificate["leaderboard_eligible"] is False
    assert "custom_task_subset" in certificate["deviations"]
    assert all(len(outcome.fault_cells) == 3 for outcome in report.units)


def test_an_official_run_declares_itself_comparable(tmp_path):
    runner, units = unit_success(tmp_path, 5, 5, official_profile=official_profile())
    report = runner.run(units)
    assert report.certificate["deviations"] == []
    assert report.certificate["official_protocol_conformant"] is True
    assert report.certificate["leaderboard_eligible"] is True


def test_a_small_exploration_is_not_blocked_by_the_cohort_gate(tmp_path):
    # Someone validating an idea on a couple of units must not be stopped by a
    # cohort-level clean-success target they never opted into.
    runner, units = unit_success(tmp_path, 1, 3)
    assert runner.enforce_admission is False        # open by default
    report = runner.run(units)
    assert cohort_admission_status(report.admission) == "insufficient_clean_success"
    assert any(outcome.fault_cells for outcome in report.units)
    assert "admission_gate_not_enforced" in report.certificate["deviations"]
    assert report.certificate["result_valid"] is True


def test_low_clean_coverage_still_runs_the_official_fault_phase(tmp_path):
    """Low clean coverage is a result, not an admission failure: the solvable
    scenes still define their task phases and run their fault cells."""
    # 3 of 5 solvable is 0.60, well under the target, and an official run.
    runner, units = unit_success(tmp_path, 3, 5, official_profile=official_profile())
    report = runner.run(units)
    assert report.admission.clean_success_rate == pytest.approx(0.6)
    # The status is reported, not acted on.
    assert cohort_admission_status(report.admission) == "insufficient_clean_success"
    assert report.certificate["admission"]["scope"] == "whole_cohort"
    # The three solvable units produced fault cells; the two unsolvable ones did
    # not, because there is no trajectory of theirs to take a fraction of.
    solvable = [o for o in report.units if o.clean_solvable]
    unsolvable = [o for o in report.units if not o.clean_solvable]
    assert len(solvable) == 3 and len(unsolvable) == 2
    assert all(outcome.fault_cells for outcome in solvable)
    assert all(outcome.fault_cells == [] for outcome in unsolvable)
    assert all(o.fault_skip_reason == "healthy_unsolved" for o in unsolvable)
    # And calling the fault phase directly is not refused either.
    runner.run_fault_phase(units, report.admission)


def test_a_cohort_that_solves_nothing_runs_no_fault_cell(tmp_path):
    # Different in kind from a low score: with no healthy success anywhere there
    # is no trajectory to take 30% of, so there is no fault cell to run rather
    # than a decision to withhold one.
    runner, units = unit_success(tmp_path, 0, 5, official_profile=official_profile())
    report = runner.run(units)
    assert cohort_admission_status(report.admission) == "checkpoint_incompatible"
    assert all(outcome.fault_cells == [] for outcome in report.units)
    with pytest.raises(CohortError, match="no unit has a healthy success"):
        runner.run_fault_phase(units, report.admission)


def test_admission_is_decided_over_the_cohort_not_over_one_shard(tmp_path):
    # Shard 0 would see 1/2 solvable and shard 1 would see 3/3; neither number is
    # the model's. Only the complete list gives the true 4/5 = 0.80.
    runner, units = unit_success(tmp_path, 4, 5)
    shard_a = shard_units(units, workers=2, worker_index=0)
    shard_b = shard_units(units, workers=2, worker_index=1)
    runner.run_healthy_phase(shard_a)
    with pytest.raises(CohortError, match="still missing"):
        runner.resolve_admission(units)      # the barrier is not satisfied yet
    runner.run_healthy_phase(shard_b)
    admission = runner.resolve_admission(units)
    assert admission.clean_success_rate == pytest.approx(0.8)
    assert cohort_admission_status(admission) == "admitted"


def test_skipping_an_unsolvable_unit_is_protocol_not_a_deviation(tmp_path):
    # Skipping is what the protocol says to do, so it is recorded and costs nothing.
    runner, units = unit_success(tmp_path, 4, 5, official_profile=official_profile())
    report = runner.run(units)
    assert report.certificate["result_valid"] is True
    assert report.certificate["matrix_complete_for_requested_experiment"] is True
    assert report.certificate["leaderboard_eligible"] is True
    assert "incomplete_unsolvable_matrix" not in report.certificate["deviations"]
    assert report.certificate["fault_cells_skipped_healthy_unsolved"] == 3

def test_the_onset_is_frozen_from_the_healthy_completion_step(tmp_path):
    report = make_runner(tmp_path, success_at=8).run(UNITS)
    unit = report.units[0]
    assert unit.clean_solvable is True
    assert unit.healthy_completion_step == 8
    for cell in unit.fault_cells:
        # half_up(0.5 * 8) = 4, not half_up(0.5 * official_horizon) = 10.
        assert cell.execution.result.onset_basis == "healthy_reference"
        assert "onset=0.5" in cell.execution.result.cell_id


def test_the_fault_grid_matches_the_platform_grid(tmp_path):
    report = make_runner(tmp_path).run(UNITS)
    for unit in report.units:
        faulted = {cell.spec.faulted_cameras for cell in unit.fault_cells}
        assert faulted == {WRIST, AGENTVIEW, EVERYTHING}


def test_a_single_view_policy_runs_the_no_op_cell_and_it_is_ranked_like_any_other(tmp_path):
    report = make_runner(tmp_path, visible=WRIST).run(UNITS)
    unit = report.units[0]
    assert len(unit.fault_cells) == 3           # the schedule is unchanged
    cells = {cell.spec.faulted_cameras: cell for cell in unit.fault_cells}
    assert cells[WRIST].scope_note is None
    assert cells[AGENTVIEW].scope_note == "no_op_for_declared_scope"
    # Total visual loss takes the one camera this policy reads.
    assert cells[EVERYTHING].scope_note is None
    # The manifest keeps the PLATFORM's grid and every cell of it is ranked:
    # the scope note is provenance of the (policy, cell) pair, never a rule.
    assert cells[AGENTVIEW].execution.result.subset_ranking_eligible is True
    assert cells[AGENTVIEW].execution.result.ranking_eligible is True
    assert report.certificate["fault_cells_by_scope_note"] == {
        "no_op_for_declared_scope": 2, "policy_scope_unresolved": 0}


def test_native_and_matched_replanning_never_share_a_cell_identity(tmp_path):
    native = make_runner(tmp_path / "native").run(UNITS)
    matched = make_runner(tmp_path / "matched", profile_spec="matched_replan_1").run(UNITS)
    assert native.profile.max_actions_per_query == 2
    assert matched.profile.max_actions_per_query == 1
    native_keys = {cell.execution.path.name for cell in native.units[0].fault_cells}
    matched_keys = {cell.execution.path.name for cell in matched.units[0].fault_cells}
    assert native_keys.isdisjoint(matched_keys)


def test_the_second_run_resumes_every_cell(tmp_path):
    first = make_runner(tmp_path).run(UNITS)
    second = make_runner(tmp_path).run(UNITS)
    def executions(report):
        return [e for u in report.units for e in u.healthy_cells] + [
            c.execution for u in report.units for c in u.fault_cells
        ]

    assert all(execution.resumed is False for execution in executions(first))
    assert all(execution.resumed is True for execution in executions(second))
    assert first.specification_hash == second.specification_hash
    assert first.certificate["cohort_unit_list_hash"] == second.certificate["cohort_unit_list_hash"]


def test_determinism_is_detected_from_the_healthy_replicates(tmp_path):
    # Determinism is a property only two or more replicates can show; the
    # official run has one, so this is a diagnostic cohort with three.
    deterministic = make_runner(tmp_path / "d", policy_class=DeterministicPolicy,
                                healthy_replicates=3).run(UNITS)
    stochastic = make_runner(tmp_path / "s", policy_class=StochasticPolicy,
                             healthy_replicates=3).run(UNITS)
    assert all(unit.policy_determinism == "deterministic" for unit in deterministic.units)
    assert all(unit.policy_determinism == "stochastic" for unit in stochastic.units)


def test_placement_stays_out_of_the_cell_identity(tmp_path):
    a = make_runner(tmp_path)
    a.worker_index, a.workers, a.runtime_metadata = 0, 3, {"gpu": "cuda:0"}
    first = a.run(shard_units(cohort_units(["t0", "t1"], [0]), workers=1, worker_index=0))
    b = make_runner(tmp_path)
    b.worker_index, b.workers, b.runtime_metadata = 2, 3, {"gpu": "cuda:2"}
    second = b.run(shard_units(cohort_units(["t0", "t1"], [0]), workers=1, worker_index=0))
    # A different worker and device produce the very same cells.
    assert [c.execution.path for u in first.units for c in u.fault_cells] == [
        c.execution.path for u in second.units for c in u.fault_cells
    ]
    assert all(c.execution.resumed for u in second.units for c in u.fault_cells)
    assert first.certificate["runtime"]["gpu"] != second.certificate["runtime"]["gpu"]
    assert first.specification_hash == second.specification_hash


def test_a_complete_custom_run_is_not_a_complete_official_matrix(tmp_path):
    # Two of five official tasks, run without a single gap: complete as the
    # experiment it is, incomplete as the official matrix.
    runner, units = unit_success(tmp_path, 2, 2, official_profile=official_profile())
    certificate = runner.run(units).certificate
    assert certificate["matrix_complete_for_requested_experiment"] is True
    assert certificate["official_matrix_complete"] is False
    assert certificate["deviations"] == ["custom_task_subset"]


def test_a_different_evaluation_distribution_is_never_official_comparable(tmp_path):
    import dataclasses

    from mail_bench.driver import profile_deviations

    profile = official_profile()
    base = CohortSpecification(
        benchmark="fake", suite="fake_suite", dataset_id="fake", dataset_revision="rev",
        tasks=profile.tasks, episode_indices=profile.episode_indices,
        platform_camera_ids=CAMERAS, onset_fractions=profile.onset_fractions,
        official_horizon=HORIZON, execution_profile="native",
    )
    common = dict(
        healthy_replicates=1, cohort_target=0.80,
        enforce_admission=True,
    )
    assert profile_deviations(base, profile, **common) == ()
    for field, value, tag in (
        ("suite", "other_suite", "different_suite"),
        ("dataset_id", "other", "different_dataset"),
        ("dataset_revision", "other-rev", "different_dataset_revision"),
        ("protocol_version", "mail_bench_perturbation_v1_2", "different_protocol"),
    ):
        changed = dataclasses.replace(base, **{field: value})
        assert tag in profile_deviations(changed, profile, **common)


def test_the_official_profile_takes_the_protocols_replicate_count(tmp_path):
    """A run doing exactly what the protocol says must not be marked custom:
    the profile's default replicate count is the protocol's, and this pins
    both directions."""
    from mail_bench.driver import profile_deviations

    profile = dataclasses.replace(official_profile(), healthy_replicates=1)
    deviations = profile_deviations(
        specification(), profile, healthy_replicates=1,
        cohort_target=profile.cohort_target, enforce_admission=True,
    )
    assert "custom_healthy_replicates" not in deviations
    # And a run that really does differ is still recorded as differing.
    assert "custom_healthy_replicates" in profile_deviations(
        specification(), profile, healthy_replicates=3,
        cohort_target=profile.cohort_target, enforce_admission=True,
    )


def test_completeness_is_counted_from_the_outcomes_not_declared(tmp_path):
    """A certificate's completion flags are counted from the outcomes: every
    requested unit must have an outcome and every clean-solvable unit all the
    fault cells it owes (three states at every onset), and the official
    matrix is complete only when the whole cohort was requested."""
    runner = make_runner(tmp_path)
    report = runner.run(UNITS)
    certificate = report.certificate
    assert certificate["fault_cells_owed_per_solvable_unit"] == 3 * len(specification().onset_fractions)
    assert certificate["matrix_complete_for_requested_experiment"] is True
    assert certificate["result_valid"] is True
    # A custom experiment is complete for what it asked and is not the
    # official matrix; the two flags are different claims.
    assert certificate["official_matrix_complete"] is False
    assert "no_official_profile_declared" in certificate["deviations"]


def test_the_healthy_certificate_carries_the_runs_identity(tmp_path):
    """The onset manifest is built from the healthy phase's certificate alone,
    so that certificate must say which protocol, weights and declaration the
    cells belong to, and whether this worker finished its shard."""
    runner = make_runner(tmp_path)
    outcomes = runner.run_healthy_phase(UNITS)
    certificate = runner.healthy_certificate(UNITS, outcomes)
    assert certificate["phase"] == "healthy"
    assert certificate["protocol_version"] == PROTOCOL_VERSION
    assert certificate["cohort_specification"]["protocol_version"] == PROTOCOL_VERSION
    assert certificate["measurement_identity"]["checkpoint_hash"] == "c" * 64
    assert certificate["policy_declaration"]["model_id"] == "somebody/fake"
    assert certificate["healthy_phase_complete_for_worker"] is True
    assert {u["task"] for u in certificate["units"]} == {"t0", "t1"}
    fewer = runner.healthy_certificate(UNITS, outcomes[:1])
    assert fewer["healthy_phase_complete_for_worker"] is False

