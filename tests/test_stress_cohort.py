"""A 246-cell synthetic cohort, end to end, with no simulator and no GPU.

Thirty units, one healthy rollout each. Only the twenty-four clean-solvable
units produce fault cells -- no onset is placed in a task phase the policy
never reached -- so twenty-four units x three missing states x three onsets
gives two hundred and sixteen fault cells, and two hundred and forty-six in
total.  The point is not the size but the properties that only appear at
size: sharding, order independence, the admission barrier, resume after a
crash, and whether the official scorer reproduces a ground truth that can be
worked out by hand.

The ground truth is chosen so every statistic is exact:

    healthy solvable                24 / 30
    agentview_missing .30/.45/.60   12 / 18 / 21 successes   M_c = 0.50 0.75 0.875
    wrist_missing     .30/.45/.60    6 / 12 / 18 successes   M_c = 0.25 0.50 0.75
    all_vision_missing               0 successes             M_c = 0, and ranked

Successes are assigned to the lowest-indexed units, which are exactly the
solvable ones, so every fault cell has a healthy success behind it and the
fault denominator is 24 rather than 30.
"""

from pathlib import Path

import pytest

from mail_bench.aggregate import mail_bench_report
from mail_bench.cohort import CohortError, cohort_units, shard_units
from mail_bench.driver import CohortRunner, CohortSpecification
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
from mail_bench.semantic_states import state_of
from mail_bench.cohort import PlatformPreflight

PLATFORM = "robocasa365"
CAMERAS = ("robot0_agentview_left", "robot0_agentview_right", "robot0_eye_in_hand")
HORIZON = 12
ONSETS = (0.30, 0.45, 0.60)
UNITS = 30
SOLVABLE = 24
GROUND_TRUTH = {
    ("agentview_missing", 0.30): 12, ("agentview_missing", 0.45): 18, ("agentview_missing", 0.60): 21,
    ("wrist_missing", 0.30): 6, ("wrist_missing", 0.45): 12, ("wrist_missing", 0.60): 18,
    ("all_vision_missing", 0.30): 0, ("all_vision_missing", 0.45): 0, ("all_vision_missing", 0.60): 0,
}
TASKS = tuple(f"task_{i}" for i in range(UNITS))


def unit_index(task):
    return int(task.split("_")[-1])


class ScriptedEnvironment(EnvironmentAdapter):
    """Succeeds or fails exactly as the ground-truth table prescribes."""

    def __init__(self, *, succeed_at):
        self.succeed_at = succeed_at
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
        return {"success": self.success()}

    def success(self):
        return self.succeed_at is not None and self.step_index >= self.succeed_at


class ScriptedPolicy(PolicyAdapter):
    def reset(self, policy_seed):
        self.seed = policy_seed

    def act(self, observation):
        return PolicyOutput(tuple(("a", self.seed % 5, index) for index in range(2)))


DECLARATION = validate_declaration(
    {
        "protocol": "mcr-stress-v1",
        "model_id": "stress/policy",
        "checkpoint_sha256": "a" * 64,
        "policy_visible_cameras": list(CAMERAS),
        "availability_consumed_by_policy": False,
        "predicted_action_chunk": 2,
        "native_execution_horizon": 2,
        "action_dimension": 7,
        "stateful_policy": False,
        "reset_semantics": "stateless",
        "training_regime": "clean_trained",
    },
    environment_cameras=CAMERAS,
)


def template(task, episode_index, faulted, onset_fraction):
    return FaultManifest(
        dataset_version="rev", split="stage_1", scene_or_task=task,
        episode_or_sequence=episode_index, camera_ids=list(CAMERAS),
        faulted_camera_ids=list(faulted), perturbation_family="availability",
        fault_mode="hard_missing", onset_fraction=onset_fraction, duration_fraction=1.0,
        environment_seed=1, policy_seed=2, fault_seed=3, official_horizon=HORIZON,
        adapter_version="stress-v1", protocol_version=PROTOCOL_VERSION,
        onset_basis="not_applicable", healthy_reference_success=False,
        onset_reference_hash="b" * 64, onset_fallback=False,
        subset_ranking_eligible=True,
    )


def make_runner(root, *, worker_index=0, workers=1, method="stress/policy"):
    def environment_factory(manifest):
        index = unit_index(manifest.scene_or_task)
        if manifest.fault_mode == "healthy":
            return ScriptedEnvironment(succeed_at=2 if index < SOLVABLE else None)
        state = state_of(PLATFORM, manifest.faulted_camera_ids)
        successes = GROUND_TRUTH[(state, round(manifest.onset_fraction, 2))]
        # Successes go to the lowest-indexed units, which are exactly the
        # clean-solvable ones, so every condition's M_c is exact.
        return ScriptedEnvironment(succeed_at=2 if index < successes else None)

    paired = PairedCellRunner(
        environment_factory=environment_factory,
        policy_factory=lambda arm, manifest: ScriptedPolicy(),
        config=RunnerConfig(max_environment_steps=HORIZON, max_actions_per_query=2),
        execution=validate_dataset({}, None, stage="stage_0", mock=True),
        writer=AtomicResultWriter(root / "cells"),
    )
    specification = CohortSpecification(
        benchmark="stress", suite="stress_suite", dataset_id="stress", dataset_revision="rev",
        tasks=TASKS, episode_indices=(0,),
        platform_camera_ids=CAMERAS, onset_fractions=ONSETS,
        official_horizon=HORIZON, execution_profile="native",
    )
    arm = MethodArmExecution.from_config(method, "a" * 64, {"policy": "scripted"})
    return CohortRunner(
        specification=specification,
        platform=PlatformPreflight(
            protocol_version=PROTOCOL_VERSION,
            dataset=validate_dataset({}, None, stage="stage_0", mock=True),
            benchmark_commit="0" * 40,
            seed_projection=audit_seed_projection([1, 2], 32),
            unit_count=UNITS,
            platform_camera_ids=CAMERAS,
        ),
        declaration=DECLARATION,
        profile=type("P", (), {"name": "native", "max_actions_per_query": 2})(),
        paired=paired, arm=arm, manifest_template=template,
        worker_index=worker_index, workers=workers,
        healthy_replicates=1,
        semantic_platform=PLATFORM,
    )


ALL_UNITS = cohort_units(list(TASKS), [0])


def run_cohort(root, *, workers=4, order=None):
    """Run the whole cohort through ``workers`` shards, healthy phase first."""
    shards = [shard_units(ALL_UNITS, workers=workers, worker_index=i) for i in range(workers)]
    indices = list(range(workers)) if order is None else list(order)
    runners = {i: make_runner(root, worker_index=i, workers=workers) for i in range(workers)}
    for i in indices:
        runners[i].run_healthy_phase(shards[i])
    admissions = {}
    for i in indices:
        admission = runners[i].resolve_admission(ALL_UNITS)
        admissions[i] = admission
        runners[i].run_fault_phase(shards[i], admission)
    return shards, admissions


def summarise(root):
    """The official scorer on a synthetic suite: neutral shape, exact numbers."""
    return mail_bench_report(root / "cells", platform=PLATFORM, expected_tasks=TASKS,
                             scenes_per_task=1, require_official=False)


def condition_totals(report):
    """Successes and owed denominators per condition, summed over tasks."""
    totals = {}
    for task in report["tasks"]:
        for key, condition in task["conditions"].items():
            state, onset = key.split("@")
            entry = totals.setdefault((state, round(float(onset), 2)), {"successes": 0, "owed": 0, "written": 0})
            entry["successes"] += condition["successes"]
            entry["owed"] += condition["owed"]
            entry["written"] += condition["written"]
    return totals


@pytest.fixture(scope="module")
def cohort(tmp_path_factory):
    root = tmp_path_factory.mktemp("stress")
    shards, admissions = run_cohort(root, workers=4)
    report = summarise(root)
    return {"root": root, "shards": shards, "admissions": admissions, "report": report}


def test_the_cohort_produced_exactly_246_cells(cohort):
    cells = sorted(p.name for p in (cohort["root"] / "cells").glob("*.json"))
    assert len(cells) == UNITS + SOLVABLE * 3 * 3               # 30 + 216
    report = cohort["report"]
    assert len(report["tasks"]) == UNITS
    assert sum(t["healthy_successes"] for t in report["tasks"]) == SOLVABLE
    totals = condition_totals(report)
    assert sum(v["written"] for v in totals.values()) == SOLVABLE * 3 * 3
    assert report["audit_failures"] == []
    assert report["zero_exposure_fault_cells"] == 0


def test_four_shards_partition_the_cohort(cohort):
    shards = cohort["shards"]
    flattened = [unit for shard in shards for unit in shard]
    assert sorted(flattened) == sorted(ALL_UNITS)
    assert len(flattened) == len(set(flattened))


def test_every_worker_reaches_the_same_global_admission(cohort):
    rates = {admission.clean_success_rate for admission in cohort["admissions"].values()}
    assert rates == {SOLVABLE / UNITS}                      # 0.80 for all four


def test_the_barrier_refuses_until_every_shard_has_run(tmp_path):
    shards = [shard_units(ALL_UNITS, workers=4, worker_index=i) for i in range(4)]
    runner = make_runner(tmp_path, worker_index=0, workers=4)
    for shard in shards[:3]:
        runner.run_healthy_phase(shard)
    with pytest.raises(CohortError, match="still missing"):
        runner.resolve_admission(ALL_UNITS)
    runner.run_healthy_phase(shards[3])
    assert runner.resolve_admission(ALL_UNITS).clean_success_rate == pytest.approx(0.8)


def test_worker_order_does_not_change_the_result_set(tmp_path):
    forward = tmp_path / "forward"
    reverse = tmp_path / "reverse"
    forward.mkdir()
    reverse.mkdir()
    run_cohort(forward, workers=4, order=[0, 1, 2, 3])
    run_cohort(reverse, workers=4, order=[3, 1, 0, 2])
    names = {
        root.name: sorted(path.name for path in (root / "cells").glob("*.json"))
        for root in (forward, reverse)
    }
    assert names["forward"] == names["reverse"]


def test_resume_after_a_crash_matches_a_fresh_run(tmp_path):
    partial = tmp_path / "partial"
    fresh = tmp_path / "fresh"
    partial.mkdir()
    fresh.mkdir()
    # "Crash" after two of four shards, then resume the whole cohort.
    shards = [shard_units(ALL_UNITS, workers=4, worker_index=i) for i in range(4)]
    crashed = make_runner(partial, worker_index=0, workers=4)
    for shard in shards[:2]:
        crashed.run_healthy_phase(shard)
    run_cohort(partial, workers=4)
    run_cohort(fresh, workers=4)
    assert sorted(p.name for p in (partial / "cells").glob("*.json")) == sorted(
        p.name for p in (fresh / "cells").glob("*.json")
    )
    assert condition_totals(summarise(partial)) == condition_totals(summarise(fresh))
    assert summarise(partial)["score"] == pytest.approx(summarise(fresh)["score"])


def test_the_scorer_reproduces_the_ground_truth(cohort):
    totals = condition_totals(cohort["report"])
    for (state, onset), expected in GROUND_TRUTH.items():
        assert totals[(state, onset)]["successes"] == expected
        # Every fault cell has a healthy success behind it: the denominator is
        # the twenty-four solvable units, never the thirty.
        assert totals[(state, onset)]["owed"] == SOLVABLE
    # Total visual loss is a ranked condition, scored zero rather than dropped.
    for task in cohort["report"]["tasks"]:
        for key, condition in task["conditions"].items():
            assert condition["complete"] is True
            if key.startswith("all_vision_missing") and task["healthy_successes"]:
                assert condition["score"] == 0.0


def test_the_task_score_is_the_ten_equal_conditions(cohort):
    # A solvable unit: H = 1 and its nine M_c are the unit's own fault outcomes.
    report = cohort["report"]
    by_task = {t["task"]: t for t in report["tasks"]}
    solved = by_task["task_0"]                      # index 0 succeeds under every condition
    assert solved["healthy_score"] == 1.0
    assert solved["task_score"] == pytest.approx((1 + 6 * 1.0 + 3 * 0.0) / 10)
    unsolved = by_task[f"task_{UNITS - 1}"]         # never solved healthily: H = 0, no fault cells
    assert unsolved["healthy_successes"] == 0
    assert unsolved["task_score"] == pytest.approx(0.0)


def test_one_healthy_cell_per_scene_or_the_scorer_refuses(tmp_path):
    root = tmp_path / "dup"
    root.mkdir()
    run_cohort(root, workers=1)
    from mail_bench.aggregate import AggregationError, load_cells
    rows, _ = load_cells(root / "cells")
    healthy = Path(next(row.source for row in rows if row.is_healthy))
    # A second replicate of the same scene: same cell, another policy seed.
    # (An identical copy is a resume artefact and is deduplicated on load.)
    import json
    payload = json.loads(healthy.read_text())
    payload["result"]["policy_seed"] = int(payload["result"]["policy_seed"]) + 1
    healthy.with_name(healthy.stem + "_rep1.json").write_text(json.dumps(payload))
    with pytest.raises(AggregationError, match="more than one healthy cell"):
        summarise(root)
