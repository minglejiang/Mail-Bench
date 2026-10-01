import pytest

from mail_bench.interfaces import (
    CanonicalObservation,
    EnvironmentAdapter,
    EnvironmentStep,
    PolicyAdapter,
    PolicyOutput,
)
from mail_bench.experiment import (
    HealthyReferenceRunner,
    HealthyReplicate,
    HealthyReplicateOutcome,
    MethodArmExecution,
    PairedCellRunner,
    aggregate_healthy_references,
    freeze_fault_manifest,
    healthy_replicate_manifests,
    resolve_reference_onset,
)
from mail_bench.io import AtomicResultWriter
from mail_bench.manifest import PROTOCOL_VERSION
from mail_bench.onset import round_half_up
from mail_bench.manifest import FaultManifest
from mail_bench.registry import DatasetValidation, validate_dataset
from mail_bench.runner import RunnerConfig
from mail_bench.seeds import episode_seed, policy_seed


def replicate(index, success, step, policy_seed=None):
    return HealthyReplicate(
        replicate_index=index,
        environment_seed=10,
        policy_seed=20 + index if policy_seed is None else policy_seed,
        success=success,
        completion_step=step,
    )


def test_healthy_reference_uses_success_probability_and_median_half_up():
    summary = aggregate_healthy_references(
        [replicate(0, True, 120), replicate(1, True, 137), replicate(2, False, None)],
        official_horizon=200,
    )
    assert summary.success_probability == pytest.approx(2 / 3)
    assert summary.clean_solvable is True
    assert summary.healthy_completion_step == 129
    assert len(summary.onset_reference_hash) == 64
    onset = resolve_reference_onset(summary, fraction=0.5, official_horizon=200)
    assert onset["onset_step"] == 65
    assert onset["basis"] == "healthy_reference"


def test_an_unsolvable_reference_yields_no_onset_at_all():
    summary = aggregate_healthy_references(
        [replicate(0, True, 80), replicate(1, False, None), replicate(2, False, None)],
        official_horizon=200,
    )
    assert summary.clean_solvable is False
    assert summary.healthy_completion_step is None
    onset = resolve_reference_onset(summary, fraction=0.5, official_horizon=200)
    assert onset["onset_step"] is None
    assert onset["fault_evaluable"] is False
    assert onset["skip_reason"] == "healthy_unsolved"


def test_healthy_reference_rejects_non_frozen_or_ambiguous_replicates():
    with pytest.raises(ValueError, match="frozen environment"):
        aggregate_healthy_references(
            [replicate(0, True, 10), HealthyReplicate(1, 11, 21, True, 12)],
            official_horizon=20,
        )
    with pytest.raises(ValueError, match="distinct policy seeds"):
        aggregate_healthy_references(
            [replicate(0, True, 10, 20), replicate(1, True, 12, 20)],
            official_horizon=20,
        )
    with pytest.raises(ValueError, match="failed healthy"):
        aggregate_healthy_references(
            [replicate(0, False, 10)],
            official_horizon=20,
        )


def test_healthy_reference_runner_derives_and_aggregates_three_requests():
    requests = []
    outcomes = [(True, 8), (False, None), (True, 11)]

    def execute(request):
        requests.append(request)
        success, step = outcomes[request.replicate_index]
        return HealthyReplicateOutcome(success, step)

    summary = HealthyReferenceRunner(
        benchmark="fake",
        task="task_1",
        episode_index=0,
        protocol_version=PROTOCOL_VERSION,
        official_horizon=20,
        replicates=3,
        run_replicate=execute,
    ).run()
    assert len(requests) == 3
    assert len({request.environment_seed for request in requests}) == 1
    assert len({request.policy_seed for request in requests}) == 3
    assert summary.healthy_completion_step == 10


class TinyEnvironment(EnvironmentAdapter):
    def __init__(self):
        self.step_index = 0
        self.closed = False

    def observation(self):
        return CanonicalObservation(
            step=self.step_index,
            cameras={"agentview": ("frame", self.step_index)},
            availability={"agentview": True},
        )

    def reset(self, environment_seed):
        self.step_index = 0
        return self.observation()

    def step(self, action):
        self.step_index += 1
        return EnvironmentStep(self.observation(), terminated=self.step_index >= 2)

    def camera_inventory(self):
        return ("agentview",)

    def official_metrics(self):
        return {"success": self.success()}

    def success(self):
        return self.step_index >= 2

    def close(self):
        self.closed = True


class TinyPolicy(PolicyAdapter):
    def __init__(self):
        self.closed = False

    def reset(self, policy_seed):
        pass

    def act(self, observation):
        return PolicyOutput((observation.step,))

    def close(self):
        self.closed = True


def fault_template():
    return FaultManifest(
        dataset_version="mock-v1",
        split="stage_0",
        scene_or_task="tiny_task",
        episode_or_sequence=0,
        camera_ids=["agentview"],
        faulted_camera_ids=["agentview"],
        perturbation_family="availability",
        fault_mode="hard_missing",
        onset_fraction=0.5,
        duration_fraction=1.0,
        environment_seed=1,
        policy_seed=2,
        fault_seed=3,
        official_horizon=2,
        adapter_version="tiny-v1",
        protocol_version=PROTOCOL_VERSION,
        onset_step=1,
        onset_basis="healthy_reference",
        healthy_reference_success=True,
        onset_reference_hash="a" * 64,
        onset_fallback=False,
        subset_ranking_eligible=False,
    )


def test_paired_cell_runner_freezes_manifest_and_resumes_without_rerun(tmp_path):
    summary = aggregate_healthy_references(
        [replicate(0, True, 2), replicate(1, True, 2), replicate(2, True, 2)],
        official_horizon=2,
    )
    manifest = freeze_fault_manifest(fault_template(), summary, onset_fraction=0.5)
    arm = MethodArmExecution.from_config("tiny-policy", "b" * 64, {"mask": True})
    factory_calls = []
    environments = []
    policies = []

    def environment_factory(frozen_manifest):
        factory_calls.append(frozen_manifest.semantic_hash())
        environment = TinyEnvironment()
        environments.append(environment)
        return environment

    def policy_factory(arm_spec, frozen_manifest):
        policy = TinyPolicy()
        policies.append(policy)
        return policy

    paired = PairedCellRunner(
        environment_factory=environment_factory,
        policy_factory=policy_factory,
        config=RunnerConfig(max_environment_steps=2),
        execution=validate_dataset({}, None, stage="stage_0", mock=True),
        writer=AtomicResultWriter(tmp_path),
    )
    first = paired.run_cell(manifest, arm)
    second = paired.run_cell(manifest, arm)
    assert first.resumed is False and second.resumed is True
    assert first.path == second.path
    assert len(factory_calls) == 1
    assert environments[0].closed is True
    assert policies[0].closed is True
    assert second.result.policy_input_trace_hash == first.result.policy_input_trace_hash

    unauthorized = PairedCellRunner(
        environment_factory=environment_factory,
        policy_factory=lambda arm_spec, frozen_manifest: TinyPolicy(),
        config=RunnerConfig(max_environment_steps=2),
        execution=DatasetValidation(None, "stage_1", True, False),
        writer=AtomicResultWriter(tmp_path),
    )
    with pytest.raises(ValueError, match="mock execution authorization"):
        unauthorized.run_cell(manifest, arm)


def test_paired_cell_runner_closes_environment_when_policy_factory_fails(tmp_path):
    summary = aggregate_healthy_references(
        [replicate(0, True, 2), replicate(1, True, 2), replicate(2, True, 2)],
        official_horizon=2,
    )
    manifest = freeze_fault_manifest(fault_template(), summary, onset_fraction=0.5)
    environment = TinyEnvironment()

    def fail_policy_factory(arm_spec, frozen_manifest):
        raise RuntimeError("policy setup failed")

    paired = PairedCellRunner(
        environment_factory=lambda frozen_manifest: environment,
        policy_factory=fail_policy_factory,
        config=RunnerConfig(max_environment_steps=2),
        execution=validate_dataset({}, None, stage="stage_0", mock=True),
        writer=AtomicResultWriter(tmp_path),
    )
    arm = MethodArmExecution.from_config("tiny-policy", "b" * 64, {"mask": True})
    with pytest.raises(RuntimeError, match="policy setup failed"):
        paired.run_cell(manifest, arm)
    assert environment.closed is True


UNIT = ("tiny", "tiny_task", 0, PROTOCOL_VERSION)


def healthy_runner(tmp_path):
    return PairedCellRunner(
        environment_factory=lambda manifest: TinyEnvironment(),
        policy_factory=lambda arm_spec, manifest: TinyPolicy(),
        config=RunnerConfig(max_environment_steps=2),
        execution=validate_dataset({}, None, stage="stage_0", mock=True),
        writer=AtomicResultWriter(tmp_path),
    )


def test_healthy_replicate_manifests_freeze_one_scene_and_separate_policy_streams():
    manifests = healthy_replicate_manifests(
        fault_template(),
        benchmark=UNIT[0],
        task=UNIT[1],
        episode_index=UNIT[2],
        protocol_version=UNIT[3],
        replicates=3,
    )
    assert len(manifests) == 3
    assert {m.environment_seed for m in manifests} == {episode_seed(*UNIT)}
    assert manifests[0].policy_seed == policy_seed(*UNIT)
    assert len({m.policy_seed for m in manifests}) == 3
    assert len({m.semantic_hash() for m in manifests}) == 3
    for manifest in manifests:
        assert manifest.fault_mode == "healthy"
        assert manifest.faulted_camera_ids == []
        assert manifest.onset_step is None and manifest.end_step is None
        assert manifest.onset_basis == "not_applicable"
        assert manifest.subset_ranking_eligible is False


def test_healthy_reference_runs_as_atomic_clean_cells_and_resumes(tmp_path):
    arm = MethodArmExecution.from_config("tiny-policy", "c" * 64, {"policy": "tiny"})
    kwargs = dict(
        benchmark=UNIT[0],
        task=UNIT[1],
        episode_index=UNIT[2],
        protocol_version=UNIT[3],
        official_horizon=2,
        replicates=3,
    )
    summary, executions = healthy_runner(tmp_path).run_healthy_reference(
        fault_template(), arm, **kwargs
    )
    assert summary.success_probability == 1.0
    assert summary.clean_solvable is True
    assert summary.healthy_completion_step == 2
    assert [execution.resumed for execution in executions] == [False, False, False]
    for execution in executions:
        result = execution.result
        assert result.clean_or_fault == "clean"
        assert result.fault_cell_valid is True
        assert result.faulted_observations_produced == 0
        assert result.ranking_eligible is False
        assert result.subset_ranking_eligible is False
        assert result.recovery_eligible is False
        assert result.availability_trace_hash and result.policy_input_trace_hash

    resumed_summary, resumed = healthy_runner(tmp_path).run_healthy_reference(
        fault_template(), arm, **kwargs
    )
    assert [execution.resumed for execution in resumed] == [True, True, True]
    assert resumed_summary.onset_reference_hash == summary.onset_reference_hash
    assert [e.path for e in resumed] == [e.path for e in executions]


def test_visual_recovery_is_expressible_and_both_ends_are_phases_of_one_trajectory():
    """Visual Recovery cannot be written as a duration_fraction.

    Both of its ends are anchored to the policy's own trajectory -- lost at 30%,
    returned at 60% -- so the equivalent fraction of anything else differs per
    scene. duration_fraction and end_step are not cross-checked, so a submitter
    who guessed a fraction would run a different condition and nothing would
    say so. The parameter names the thing the specification names.
    """
    summary = aggregate_healthy_references([replicate(0, True, 188)], official_horizon=450)
    assert summary.healthy_completion_step == 188
    template = fault_template()
    template.official_horizon = 450
    manifest = freeze_fault_manifest(template, summary,
                                     onset_fraction=0.30, recovery_fraction=0.60)
    assert manifest.onset_step == round_half_up(0.30 * 188)          # 56
    assert manifest.end_step == round_half_up(0.60 * 188)            # 113
    # The interruption is 30% of the task, as the specification states, and the
    # return lands exactly where Failure Form's 0.60 onset lands.
    assert manifest.end_step - manifest.onset_step == pytest.approx(0.30 * 188, abs=1)
    assert manifest.recovery_mode == "single_recovery"
    # The recovery end is part of the cell's identity: its id and hash differ
    # from the ranking cell's, and a ranking cell's hash does not include the field.
    ranking = freeze_fault_manifest(template, summary, onset_fraction=0.30)
    assert manifest.cell_id.endswith("|onset=0.3|rec=0.6") and ranking.cell_id.endswith("|onset=0.3|dur=end")
    assert manifest.semantic_hash() != ranking.semantic_hash()
    assert "recovery_fraction" not in ranking.to_dict() or ranking.to_dict()["recovery_fraction"] is None
    from mail_bench.aggregate import parse_cell_id
    assert parse_cell_id(manifest.cell_id)["duration"] == "rec=0.6"

    with pytest.raises(ValueError, match="one end"):
        freeze_fault_manifest(template, summary, onset_fraction=0.30,
                              duration_fraction=0.5, recovery_fraction=0.60)
    with pytest.raises(ValueError, match="must return after"):
        freeze_fault_manifest(template, summary, onset_fraction=0.60,
                              recovery_fraction=0.30)
