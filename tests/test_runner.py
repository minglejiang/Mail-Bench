from dataclasses import replace
from concurrent.futures import ThreadPoolExecutor

import pytest

from mail_bench.interfaces import (
    CanonicalObservation,
    EnvironmentAdapter,
    EnvironmentStep,
    PolicyAdapter,
    PolicyOutput,
)
from mail_bench.io import AtomicResultWriter, ResultConflictError, semantic_run_key
from mail_bench.manifest import PROTOCOL_VERSION
from mail_bench.manifest import FaultManifest, validate_manifest
from mail_bench.operators import FaultSchedule
from mail_bench.registry import DatasetValidation, validate_dataset
from mail_bench.runner import RolloutRunner, RunnerConfig


CAMS = ("head", "wrist")
MOCK_EXECUTION = validate_dataset({}, None, stage="stage_0", mock=True)


class FakeEnvironment(EnvironmentAdapter):
    def __init__(self, horizon=5, hidden_variant="a", inventory=CAMS):
        self.horizon = horizon
        self.hidden_variant = hidden_variant
        self.inventory = tuple(inventory)
        self.current_step = 0
        self.actions = []

    def _observation(self):
        cameras = {
            "head": (
                "head",
                self.current_step,
                self.hidden_variant if self.current_step >= 2 else "common",
            ),
            "wrist": ("wrist", self.current_step),
        }
        available = {camera: True for camera in cameras}
        return CanonicalObservation(
            step=self.current_step,
            cameras=cameras,
            availability=available,
            capture_time_ms={camera: self.current_step * 50.0 for camera in cameras},
            arrival_time_ms={camera: self.current_step * 50.0 for camera in cameras},
            sequence_id={camera: self.current_step for camera in cameras},
            new_frame={camera: True for camera in cameras},
        )

    def reset(self, environment_seed):
        self.current_step = 0
        self.actions = []
        self.environment_seed = environment_seed
        return self._observation()

    def step(self, action):
        self.actions.append(action)
        self.current_step += 1
        return EnvironmentStep(
            observation=self._observation(),
            terminated=self.current_step >= self.horizon,
        )

    def camera_inventory(self):
        return self.inventory

    def official_metrics(self):
        return {"success": self.success(), "actions": len(self.actions)}

    def success(self):
        return self.current_step >= self.horizon


class RecordingPolicy(PolicyAdapter):
    def __init__(self, chunk_size=1):
        self.chunk_size = chunk_size
        self.queries = []
        self.invalidations = []
        self.memory_resets = []

    def reset(self, policy_seed):
        self.policy_seed = policy_seed
        self.queries = []
        self.invalidations = []
        self.memory_resets = []

    def act(self, observation):
        self.queries.append(observation)
        signature = tuple((camera, observation.cameras[camera]) for camera in observation.camera_ids)
        return PolicyOutput(
            actions=tuple((observation.step, signature, index) for index in range(self.chunk_size))
        )

    def invalidate_action_chunk(self, reason, step):
        self.invalidations.append((reason, step))

    def reset_visual_memory(self, reason, step):
        self.memory_resets.append((reason, step))


def manifest(**overrides):
    values = dict(
        dataset_version="mock-v1",
        split="stage_0",
        scene_or_task="mock_task",
        episode_or_sequence=0,
        camera_ids=list(CAMS),
        faulted_camera_ids=["head"],
        perturbation_family="availability",
        fault_mode="hard_missing",
        onset_fraction=0.5,
        duration_fraction=1.0,
        environment_seed=11,
        policy_seed=22,
        fault_seed=0,
        official_horizon=4,
        adapter_version="fake-v1",
        protocol_version=PROTOCOL_VERSION,
        onset_step=2,
        onset_basis="healthy_reference",
        healthy_reference_success=True,
        onset_reference_hash="b" * 64,
        onset_fallback=False,
        subset_ranking_eligible=True,
    )
    values.update(overrides)
    return validate_manifest(FaultManifest(**values))


def healthy_manifest(**overrides):
    values = dict(
        faulted_camera_ids=[],
        fault_mode="healthy",
        onset_fraction=0.0,
        duration_fraction=1.0,
        onset_step=None,
        end_step=None,
        recovery_mode="none",
        onset_basis="not_applicable",
        healthy_reference_success=False,
        onset_fallback=False,
        subset_ranking_eligible=False,
    )
    values.update(overrides)
    return manifest(**values)


def run_healthy(*, horizon=4, chunk_size=1, invalidate=True, variant="a", reset_memory=False, fault_seed=0):
    environment = FakeEnvironment(horizon=horizon, hidden_variant=variant)
    policy = RecordingPolicy(chunk_size=chunk_size)
    runner = RolloutRunner(
        environment,
        policy,
        healthy_manifest(fault_seed=fault_seed),
        RunnerConfig(
            max_environment_steps=horizon,
            invalidate_chunk_on_availability_change=invalidate,
            reset_visual_memory_on_recovery=reset_memory,
        ),
        MOCK_EXECUTION,
    )
    return environment, policy, runner.run()


def run(schedule, *, horizon=4, chunk_size=1, invalidate=True, variant="a", reset_memory=False,
        max_actions_per_query=None, invalidate_on_recovery=False):
    environment = FakeEnvironment(horizon=horizon, hidden_variant=variant)
    policy = RecordingPolicy(chunk_size=chunk_size)
    if not schedule.faulted_cameras:
        # A schedule with no faulted camera is the healthy arm, which the
        # protocol executes as a first-class availability cell.
        return run_healthy(
            horizon=horizon,
            chunk_size=chunk_size,
            invalidate=invalidate,
            variant=variant,
            reset_memory=reset_memory,
            fault_seed=schedule.fault_seed,
        )
    schedule_fields = dict(
        faulted_camera_ids=sorted(schedule.faulted_cameras),
        fault_mode=schedule.mode,
        onset_step=schedule.onset_step,
        end_step=schedule.end_step,
        fault_seed=schedule.fault_seed,
    )
    if schedule.stale_k is not None:
        schedule_fields["stale_k"] = schedule.stale_k
    if schedule.mode == "burst_dropout":
        schedule_fields["burst_frames"] = schedule.burst_frames
        schedule_fields["available_rate"] = schedule.available_rate
    if schedule.recovery != "none":
        schedule_fields["recovery_mode"] = schedule.recovery
    frozen_manifest = manifest(**schedule_fields)
    runner = RolloutRunner(
        environment,
        policy,
        frozen_manifest,
        RunnerConfig(
            max_environment_steps=horizon,
            invalidate_chunk_on_availability_change=invalidate,
            invalidate_chunk_on_recovery=invalidate_on_recovery,
            reset_visual_memory_on_recovery=reset_memory,
            max_actions_per_query=max_actions_per_query,
        ),
        MOCK_EXECUTION,
    )
    outcome = runner.run()
    return environment, policy, outcome


def test_canonical_schema_and_camera_slot_sentinel():
    with pytest.raises(ValueError, match="identical camera ids"):
        CanonicalObservation(step=0, cameras={"head": 1}, availability={"wrist": True})
    environment = FakeEnvironment(inventory=("head", "wrong_slot"))
    runner = RolloutRunner(
        environment,
        RecordingPolicy(),
        manifest(),
        RunnerConfig(max_environment_steps=2),
        MOCK_EXECUTION,
    )
    with pytest.raises(ValueError, match="inventory mismatch"):
        runner.run()


def test_runner_requires_validated_execution_authorization():
    with pytest.raises(ValueError, match="pinned dataset"):
        RolloutRunner(
            FakeEnvironment(),
            RecordingPolicy(),
            manifest(),
            RunnerConfig(max_environment_steps=2),
            DatasetValidation("robocasa", "stage_1", False, False),
        )
    with pytest.raises(ValueError, match="revision differs"):
        RolloutRunner(
            FakeEnvironment(),
            RecordingPolicy(),
            manifest(),
            RunnerConfig(max_environment_steps=2),
            DatasetValidation("robocasa", "stage_1", False, True, revision="other"),
        )


def test_fault_reached_can_precede_policy_consumption():
    schedule = FaultSchedule({"head"}, onset_step=2)
    _, _, outcome = run(schedule, chunk_size=5, invalidate=False)
    assert outcome.audit.query_steps == [0]
    assert outcome.audit.fault_reached is True
    assert outcome.audit.faulted_observations_produced == 2
    assert outcome.audit.faulted_observations_consumed == 0

    result = outcome.to_result(manifest(), method_id="chunk-policy", checkpoint_hash="c" * 64)
    assert result.recovery_eligible is True
    assert result.fault_cell_valid is False
    assert result.ranking_eligible is False


def test_availability_change_invalidates_chunk_on_query_clock():
    schedule = FaultSchedule({"head"}, onset_step=2)
    _, policy, outcome = run(schedule, chunk_size=5, invalidate=True)
    assert outcome.audit.query_steps == [0, 2]
    assert outcome.audit.policy_query_count == 2
    assert outcome.audit.action_execution_count == 4
    assert outcome.audit.faulted_observations_consumed == 1
    assert policy.invalidations == [("availability_change", 2)]
    assert outcome.audit.invalidations[0].discarded_actions == 3
    assert len(outcome.audit.policy_query_trace_hash) == 64
    assert len(outcome.audit.policy_input_trace_hash) == 64
    assert len(outcome.audit.action_chunk_invalidation_trace_hash) == 64


def test_result_rejects_a_manifest_not_used_for_execution():
    _, _, outcome = run(FaultSchedule({"head"}, onset_step=2), chunk_size=1)
    different = manifest(onset_fraction=0.6, onset_step=2)
    with pytest.raises(ValueError, match="differs from the manifest used"):
        outcome.to_result(different, method_id="policy", checkpoint_hash="f" * 64)


def test_manifest_is_the_only_schedule_and_seed_source():
    frozen_manifest = manifest(
        environment_seed=101,
        policy_seed=202,
        fault_seed=303,
        onset_step=1,
        end_step=3,
        recovery_mode="single_recovery",
    )
    environment = FakeEnvironment(horizon=4)
    policy = RecordingPolicy()
    runner = RolloutRunner(
        environment,
        policy,
        frozen_manifest,
        RunnerConfig(max_environment_steps=4),
        MOCK_EXECUTION,
    )
    outcome = runner.run()
    assert environment.environment_seed == 101
    assert policy.policy_seed == 202
    assert runner.schedule.fault_seed == 303
    assert runner.schedule.onset_step == 1 and runner.schedule.end_step == 3
    assert outcome.binding.manifest_hash == frozen_manifest.semantic_hash()
    assert len(outcome.binding.schedule_hash) == 64
    assert len(outcome.binding.runner_config_hash) == 64


def test_recovery_is_measured_not_imposed_by_default():
    """Losing a camera drops the queued chunk so the fault reaches the policy
    at its scheduled step; vision returning does not. What a policy does with
    a chunk it planned blind is model design, and the protocol measures it.
    The memory-reset hook is separate and still fires when asked for."""
    schedule = FaultSchedule(
        {"head"}, onset_step=1, end_step=3, recovery="single_recovery"
    )
    _, policy, outcome = run(
        schedule, horizon=5, chunk_size=5, invalidate=True, reset_memory=True
    )
    # Re-queried at the loss (1), not at the recovery (3): the blind chunk runs on.
    assert outcome.audit.query_steps == [0, 1]
    assert policy.queries[1].cameras["head"] is None
    assert [(inv.step, inv.reason) for inv in outcome.audit.invalidations] == \
        [(1, "availability_change")]
    assert policy.memory_resets == [("availability_recovery", 3)]


def test_recovery_invalidation_is_an_explicit_hashed_choice():
    """Switched on, the recovery query receives the live frame at once; the
    switch is a RunnerConfig field and so part of the runner config hash."""
    schedule = FaultSchedule(
        {"head"}, onset_step=1, end_step=3, recovery="single_recovery"
    )
    _, policy, outcome = run(
        schedule, horizon=5, chunk_size=5, invalidate=True, invalidate_on_recovery=True
    )
    assert outcome.audit.query_steps == [0, 1, 3]
    recovery_query = policy.queries[2]
    assert recovery_query.cameras["head"] == ("head", 3, "a")
    assert recovery_query.source_age_steps["head"] == 0
    from mail_bench.manifest import semantic_sha256
    assert semantic_sha256(RunnerConfig(max_environment_steps=5)) != \
        semantic_sha256(RunnerConfig(max_environment_steps=5, invalidate_chunk_on_recovery=True))


def test_a_failed_rollout_says_why_it_ended():
    """The platform ending an episode at its own horizon must name the reason;
    the kernel's own cap is not the only way an episode ends."""

    class ExhaustedEnvironment(FakeEnvironment):
        def step(self, action):
            self.actions.append(action)
            self.current_step += 1
            return EnvironmentStep(observation=self._observation(),
                                   terminated=self.current_step >= 3)

        def success(self):
            return False

        def official_metrics(self):
            return {"success": False, "horizon_exhausted": self.current_step >= 3}

    environment = ExhaustedEnvironment(horizon=3)
    runner = RolloutRunner(environment, RecordingPolicy(chunk_size=1), manifest(),
                           RunnerConfig(max_environment_steps=4), MOCK_EXECUTION)
    outcome = runner.run()
    assert outcome.success is False
    assert outcome.failure_reason == "horizon_exhausted"
    # And the kernel's own cap is still named when it is the one that ended it.
    capped = RolloutRunner(FakeEnvironment(horizon=9), RecordingPolicy(chunk_size=1), manifest(),
                           RunnerConfig(max_environment_steps=2), MOCK_EXECUTION).run()
    assert capped.failure_reason == "max_environment_steps"


def test_pre_fault_equivalence_and_hidden_frame_state_contamination():
    healthy = FaultSchedule(frozenset(), onset_step=2)
    fault = FaultSchedule({"head"}, onset_step=2)
    _, _, clean_outcome = run(healthy, chunk_size=1, variant="clean")
    _, _, fault_a = run(fault, chunk_size=1, variant="poison-a")
    _, _, fault_b = run(fault, chunk_size=1, variant="poison-b")

    assert clean_outcome.audit.actions[:2] == fault_a.audit.actions[:2]
    assert clean_outcome.audit.fault_reached is False
    assert clean_outcome.audit.actions[2:] != fault_a.audit.actions[2:]
    assert fault_a.audit.actions == fault_b.audit.actions
    assert fault_a.audit.action_trace_hash == fault_b.audit.action_trace_hash
    assert fault_a.audit.availability_trace_hash == fault_b.audit.availability_trace_hash
    assert fault_a.audit.policy_query_trace_hash == fault_b.audit.policy_query_trace_hash
    assert (
        fault_a.audit.action_chunk_invalidation_trace_hash
        == fault_b.audit.action_chunk_invalidation_trace_hash
    )


def test_atomic_writer_is_idempotent_and_rejects_conflicts(tmp_path):
    _, _, outcome = run(FaultSchedule({"head"}, onset_step=2), chunk_size=1)
    frozen_manifest = manifest()
    result = outcome.to_result(
        frozen_manifest,
        method_id="recording-policy",
        checkpoint_hash="d" * 64,
    )
    writer = AtomicResultWriter(tmp_path)
    path = writer.write("run-001", result, frozen_manifest)
    assert path.exists() and writer.write("run-001", result, frozen_manifest) == path
    assert not list(tmp_path.glob("*.tmp"))

    conflicting = replace(result, method_id="different-policy")
    with pytest.raises(ResultConflictError, match="different content"):
        writer.write("run-001", conflicting, frozen_manifest)
    with pytest.raises(ValueError, match="run_id"):
        writer.write("../escape", result, frozen_manifest)


def test_atomic_writer_never_overwrites_under_concurrent_writers(tmp_path):
    _, _, outcome = run(FaultSchedule({"head"}, onset_step=2), chunk_size=1)
    frozen_manifest = manifest()
    first = outcome.to_result(
        frozen_manifest, method_id="first", checkpoint_hash="e" * 64
    )
    second = replace(first, method_id="second")
    writer = AtomicResultWriter(tmp_path)

    def attempt(result):
        try:
            writer.write("shared-run", result, frozen_manifest)
            return "written"
        except ResultConflictError:
            return "conflict"

    with ThreadPoolExecutor(max_workers=2) as pool:
        outcomes = list(pool.map(attempt, (first, second)))
    assert sorted(outcomes) == ["conflict", "written"]
    assert not list(tmp_path.glob("*.tmp"))


def test_semantic_writer_prevents_renamed_duplicate_cells(tmp_path):
    _, _, outcome = run(FaultSchedule({"head"}, onset_step=2), chunk_size=1)
    frozen_manifest = manifest()
    result = outcome.to_result(
        frozen_manifest,
        method_id="recording-policy",
        checkpoint_hash="f" * 64,
        method_config_hash="a" * 64,
    )
    writer = AtomicResultWriter(tmp_path)
    key = semantic_run_key(result, frozen_manifest)
    path = writer.write_semantic(result, frozen_manifest)
    assert path.stem == key

    recomputed = replace(result, runtime_ms=result.runtime_ms + 1.0)
    assert semantic_run_key(recomputed, frozen_manifest) == key
    with pytest.raises(ResultConflictError, match="different content"):
        writer.write_semantic(recomputed, frozen_manifest)


def test_atomic_reader_rejects_corrupt_or_mismatched_results(tmp_path):
    writer = AtomicResultWriter(tmp_path)
    frozen_manifest = manifest()
    corrupt = writer.result_path("corrupt")
    corrupt.write_text("not-json", encoding="utf-8")
    with pytest.raises(ResultConflictError, match="not a readable result"):
        writer.read("corrupt", frozen_manifest)

    corrupt.write_text('{"manifest_hash":"wrong","result":{}}', encoding="utf-8")
    with pytest.raises(ResultConflictError, match="different manifest"):
        writer.read("corrupt", frozen_manifest)


def test_prefix_hash_chain_proves_pre_fault_equivalence():
    # A whole-trace hash cannot show that two arms shared a prefix; the chain can.
    onset = 2
    _, _, clean_outcome = run(FaultSchedule(frozenset(), onset_step=onset), variant="clean")
    _, _, fault_outcome = run(FaultSchedule({"head"}, onset_step=onset), variant="poison-a")

    assert (
        fault_outcome.audit.pre_fault_action_hash
        == clean_outcome.audit.action_prefix_hashes[onset - 1]
    )
    assert (
        fault_outcome.audit.pre_fault_policy_input_hash
        == clean_outcome.audit.policy_input_prefix_hashes[onset - 1]
    )
    assert fault_outcome.audit.action_trace_hash != clean_outcome.audit.action_trace_hash
    assert fault_outcome.audit.action_prefix_hashes[onset:] != (
        clean_outcome.audit.action_prefix_hashes[onset:]
    )
    # The healthy cell has no onset of its own and freezes no onset digests.
    assert clean_outcome.audit.pre_fault_action_hash is None
    assert clean_outcome.audit.pre_fault_policy_input_hash is None


def test_fault_exposure_separates_visual_and_control_exposure():
    _, _, outcome = run(FaultSchedule({"head"}, onset_step=1), horizon=5, chunk_size=5)
    assert outcome.audit.faulted_observations_consumed == 1     # visual exposure
    assert outcome.audit.post_fault_action_steps == 4           # control exposure
    assert outcome.audit.max_actions_after_fault_query == 4
    assert outcome.audit.native_action_chunk_length == 5
    assert outcome.audit.mean_query_interval == 2.5


def test_matched_replanning_cap_requeries_without_hiding_the_native_chunk():
    schedule = FaultSchedule({"head"}, onset_step=0)
    _, _, native = run(schedule, chunk_size=4)
    _, _, capped = run(schedule, chunk_size=4, max_actions_per_query=2)

    assert native.audit.policy_query_count == 1
    assert capped.audit.policy_query_count == 2
    assert native.audit.native_action_chunk_length == 4
    assert capped.audit.native_action_chunk_length == 4
    assert any(
        event.reason == "matched_replanning_cap" for event in capped.audit.invalidations
    )
    # Native and matched-replanning runs can never be mistaken for one another.
    assert native.binding.runner_config_hash != capped.binding.runner_config_hash


def test_matched_replanning_cap_rejects_non_positive_values():
    with pytest.raises(ValueError, match="max_actions_per_query"):
        RunnerConfig(max_environment_steps=4, max_actions_per_query=0)
