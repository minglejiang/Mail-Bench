"""Healthy-reference aggregation primitives for paired benchmark execution."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Callable, Mapping, Optional, Sequence

from .interfaces import EnvironmentAdapter, PolicyAdapter
from .io import (
    AtomicResultWriter,
    ResultConflictError,
    semantic_run_key,
    semantic_run_key_fields,
)
from .manifest import (
    HEALTHY_MODE,
    FaultManifest,
    ResultRecord,
    semantic_sha256,
    validate_manifest,
)
from .onset import resolve_duration, resolve_onset, round_half_up
from .registry import DatasetValidation
from .runner import (
    RolloutRunner,
    RunnerConfig,
    schedule_from_manifest,
    validate_execution_request,
)
from .seeds import fault_seed, healthy_replicate_seeds


@dataclass(frozen=True)
class HealthyReplicate:
    replicate_index: int
    environment_seed: int
    policy_seed: int
    success: bool
    completion_step: Optional[int]
    result_hash: Optional[str] = None


@dataclass(frozen=True)
class HealthyReferenceSummary:
    replicates: tuple[HealthyReplicate, ...]
    success_probability: float
    clean_solvable: bool
    healthy_completion_step: Optional[int]
    onset_reference_hash: str

    @property
    def healthy_reference_success(self) -> bool:
        return self.clean_solvable


@dataclass(frozen=True)
class HealthyReplicateRequest:
    replicate_index: int
    environment_seed: int
    policy_seed: int


@dataclass(frozen=True)
class HealthyReplicateOutcome:
    success: bool
    completion_step: Optional[int]
    result_hash: Optional[str] = None


class HealthyReferenceRunner:
    """Run and aggregate the healthy replicates for one unit."""

    def __init__(
        self,
        *,
        benchmark: str,
        task: str,
        episode_index: int,
        protocol_version: str,
        official_horizon: int,
        run_replicate: Callable[[HealthyReplicateRequest], HealthyReplicateOutcome],
        replicates: int = 1,
        threshold: float = 0.5,
    ) -> None:
        self.benchmark = benchmark
        self.task = task
        self.episode_index = episode_index
        self.protocol_version = protocol_version
        self.official_horizon = official_horizon
        self.run_replicate = run_replicate
        self.replicates = replicates
        self.threshold = threshold

    def run(self) -> HealthyReferenceSummary:
        records = []
        seeds = healthy_replicate_seeds(
            self.benchmark,
            self.task,
            self.episode_index,
            self.protocol_version,
            self.replicates,
        )
        for index, (environment_seed, policy_seed) in enumerate(seeds):
            request = HealthyReplicateRequest(index, environment_seed, policy_seed)
            outcome = self.run_replicate(request)
            if not isinstance(outcome, HealthyReplicateOutcome):
                raise TypeError("run_replicate must return HealthyReplicateOutcome")
            records.append(
                HealthyReplicate(
                    replicate_index=index,
                    environment_seed=environment_seed,
                    policy_seed=policy_seed,
                    success=outcome.success,
                    completion_step=outcome.completion_step,
                    result_hash=outcome.result_hash,
                )
            )
        return aggregate_healthy_references(
            records,
            official_horizon=self.official_horizon,
            threshold=self.threshold,
        )


def aggregate_healthy_references(
    replicates: Sequence[HealthyReplicate],
    *,
    official_horizon: int,
    threshold: float = 0.5,
) -> HealthyReferenceSummary:
    """Freeze clean solvability and a deterministic completion-step reference.

    Successful completion steps are aggregated by median. When their count is
    even, the fractional median is rounded half-up to an integer control step.
    """
    reps = tuple(replicates)
    if not reps:
        raise ValueError("at least one healthy replicate is required")
    if official_horizon < 1:
        raise ValueError("official_horizon must be >= 1")
    if not 0.0 < threshold <= 1.0:
        raise ValueError("threshold must be in (0, 1]")
    indices = [rep.replicate_index for rep in reps]
    if indices != list(range(len(reps))):
        raise ValueError("healthy replicate indices must be contiguous from zero")
    environment_seeds = {rep.environment_seed for rep in reps}
    if len(environment_seeds) != 1:
        raise ValueError("healthy replicates must use one frozen environment seed")
    policy_seeds = [rep.policy_seed for rep in reps]
    if len(set(policy_seeds)) != len(policy_seeds):
        raise ValueError("healthy replicates must use distinct policy seeds")

    successful_steps: list[int] = []
    for rep in reps:
        if rep.success:
            if rep.completion_step is None or not 1 <= rep.completion_step <= official_horizon:
                raise ValueError("successful healthy replicate requires a valid completion_step")
            successful_steps.append(int(rep.completion_step))
        elif rep.completion_step is not None:
            raise ValueError("failed healthy replicate must not carry a completion_step")

    probability = len(successful_steps) / len(reps)
    clean_solvable = probability >= threshold
    completion_step = None
    if clean_solvable:
        completion_step = round_half_up(float(median(successful_steps)))
    payload = {
        "aggregation": "successful_completion_median_half_up",
        "threshold": threshold,
        "official_horizon": official_horizon,
        "success_probability": probability,
        "clean_solvable": clean_solvable,
        "healthy_completion_step": completion_step,
        "replicates": reps,
    }
    return HealthyReferenceSummary(
        replicates=reps,
        success_probability=probability,
        clean_solvable=clean_solvable,
        healthy_completion_step=completion_step,
        onset_reference_hash=semantic_sha256(payload),
    )


def resolve_reference_onset(
    summary: HealthyReferenceSummary,
    *,
    fraction: float,
    official_horizon: int,
) -> dict[str, object]:
    """Resolve one onset directly from a frozen healthy-reference summary."""
    return resolve_onset(
        fraction,
        official_horizon,
        healthy_completion_step=summary.healthy_completion_step,
        healthy_success=summary.clean_solvable,
    )


def healthy_replicate_manifests(
    template: FaultManifest,
    *,
    benchmark: str,
    task: str,
    episode_index: int,
    protocol_version: str,
    replicates: int = 1,
) -> tuple[FaultManifest, ...]:
    """Freeze one validated healthy cell per replicate of a paired unit.

    The healthy arm is executed as an availability cell with no faulted camera
    so that it shares the rollout kernel, the atomic result schema and the
    pre-onset trajectory audit with every fault arm.  Replicate zero carries
    the canonical paired policy stream; the environment seed is frozen across
    replicates.
    """
    if replicates < 1:
        raise ValueError("replicates must be >= 1")
    seeds = healthy_replicate_seeds(
        benchmark, task, episode_index, protocol_version, replicates
    )
    manifests = []
    for replicate_index, (environment_seed, replicate_policy_seed) in enumerate(seeds):
        manifest = copy.deepcopy(template)
        manifest.cell_id = None
        manifest.perturbation_family = "availability"
        manifest.fault_mode = HEALTHY_MODE
        manifest.faulted_camera_ids = []
        manifest.onset_fraction = 0.0
        manifest.duration_fraction = 1.0
        manifest.onset_step = None
        manifest.end_step = None
        manifest.recovery_mode = "none"
        manifest.onset_basis = "not_applicable"
        manifest.onset_fallback = False
        manifest.healthy_reference_success = False
        manifest.subset_ranking_eligible = False
        manifest.environment_seed = environment_seed
        manifest.policy_seed = replicate_policy_seed
        manifest.fault_seed = fault_seed(
            benchmark,
            task,
            episode_index,
            protocol_version,
            HEALTHY_MODE,
            replicate_index,
        )
        manifest.onset_reference_hash = semantic_sha256(
            {
                "healthy_replicate": {
                    "benchmark": benchmark,
                    "task": task,
                    "episode_index": int(episode_index),
                    "protocol_version": protocol_version,
                    "replicates": int(replicates),
                    "replicate_index": replicate_index,
                }
            }
        )
        manifests.append(validate_manifest(manifest))
    return tuple(manifests)


def freeze_fault_manifest(
    template: FaultManifest,
    summary: HealthyReferenceSummary,
    *,
    onset_fraction: float,
    duration_fraction: float = 1.0,
    recovery_fraction: Optional[float] = None,
) -> FaultManifest:
    """Resolve onset provenance and duration into a new validated manifest.

    ``recovery_fraction`` is Visual Recovery's other end: vision returns at that
    fraction of the same healthy trajectory the onset is taken from. It cannot
    be written as a ``duration_fraction`` -- both of its ends are anchored to
    the policy's trajectory, so the equivalent fraction differs per scene -- and
    a submitter who guessed one would silently run a different condition, since
    ``duration_fraction`` and ``end_step`` are not cross-checked.
    """
    onset = resolve_reference_onset(
        summary,
        fraction=onset_fraction,
        official_horizon=int(template.official_horizon),
    )
    if not onset.get("fault_evaluable", False):
        raise ValueError(
            f"{template.scene_or_task} #{template.episode_or_sequence} has no successful healthy "
            "rollout, so there is no task phase to place an onset in; the benchmark skips "
            "its fault cells rather than anchoring them to the official horizon"
        )
    onset_step = int(onset["onset_step"])
    completion = int(summary.healthy_completion_step)
    if recovery_fraction is not None:
        if duration_fraction != 1.0:
            raise ValueError(
                "a cell has one end: give recovery_fraction or duration_fraction, not both"
            )
        if not onset_fraction < recovery_fraction <= 1.0:
            raise ValueError(
                f"vision must return after it was lost: recovery_fraction "
                f"{recovery_fraction} against onset_fraction {onset_fraction}"
            )
        # The same arithmetic the onset uses, at the fraction vision returns at,
        # so both ends of the interruption are phases of one trajectory.
        end_step = max(onset_step + 1, min(int(template.official_horizon),
                                           round_half_up(recovery_fraction * completion)))
    else:
        end_step = resolve_duration(
            duration_fraction,
            onset_step,
            int(template.official_horizon),
            completion,
        )
    manifest = copy.deepcopy(template)
    manifest.cell_id = None
    manifest.onset_fraction = onset_fraction
    manifest.duration_fraction = duration_fraction
    manifest.onset_step = onset_step
    manifest.end_step = end_step
    manifest.recovery_mode = "single_recovery" if end_step is not None else "none"
    manifest.recovery_fraction = recovery_fraction
    if manifest.fault_mode == "burst_dropout":
        # The availability sequence is a function of the fault seed and the
        # block length; its hash is frozen into the manifest so a run can be
        # checked against the schedule it was supposed to apply.
        from .operators import burst_hash, burst_schedule
        from .runner import _burst_frames_for

        frames = _burst_frames_for(manifest)
        last = end_step if end_step is not None else int(manifest.official_horizon)
        manifest.burst_sequence_hash = burst_hash(burst_schedule(
            int(manifest.fault_seed), max(last - onset_step, 0), int(frames),
            float(manifest.available_rate if manifest.available_rate is not None else 0.5),
        ))
    manifest.onset_basis = str(onset["basis"])
    manifest.healthy_reference_success = summary.clean_solvable
    manifest.onset_reference_hash = summary.onset_reference_hash
    # There is no fallback: a scene reaches here only with a successful healthy
    # rollout behind it, so the field is always false.
    manifest.onset_fallback = False
    return validate_manifest(manifest)


@dataclass(frozen=True)
class MethodArmExecution:
    method_id: str
    checkpoint_hash: str
    method_config_hash: str

    @classmethod
    def from_config(
        cls,
        method_id: str,
        checkpoint_hash: str,
        config: Mapping[str, object],
    ) -> "MethodArmExecution":
        return cls(method_id, checkpoint_hash, semantic_sha256(dict(config)))


@dataclass(frozen=True)
class CellExecution:
    result: ResultRecord
    path: Path
    resumed: bool


class PairedCellRunner:
    """Execute and atomically resume a frozen manifest-by-method matrix."""

    def __init__(
        self,
        *,
        environment_factory: Callable[[FaultManifest], EnvironmentAdapter],
        policy_factory: Callable[[MethodArmExecution, FaultManifest], PolicyAdapter],
        config: RunnerConfig,
        execution: DatasetValidation,
        writer: AtomicResultWriter,
    ) -> None:
        self.environment_factory = environment_factory
        self.policy_factory = policy_factory
        self.config = config
        self.execution = execution
        self.writer = writer

    def _planned_key(self, manifest: FaultManifest, arm: MethodArmExecution) -> str:
        return semantic_run_key_fields(
            manifest,
            method_id=arm.method_id,
            checkpoint_hash=arm.checkpoint_hash,
            method_config_hash=arm.method_config_hash,
            schedule_hash=semantic_sha256(schedule_from_manifest(manifest)),
            runner_config_hash=semantic_sha256(self.config),
            dataset_authorization_hash=semantic_sha256(self.execution),
        )

    def peek_cell(
        self,
        manifest: FaultManifest,
        arm: MethodArmExecution,
    ) -> Optional[CellExecution]:
        """Return an already-written cell, or ``None``; never execute anything.

        A multi-worker cohort needs to know whether another worker has finished a
        cell without running it itself, so the global admission barrier can be
        evaluated over the complete unit list rather than one worker's shard.
        """
        manifest = validate_execution_request(manifest, self.execution)
        key = self._planned_key(manifest, arm)
        target = self.writer.result_path(key)
        if not target.exists():
            return None
        result = self.writer.read(key, manifest)
        if semantic_run_key(result, manifest) != key:
            raise ResultConflictError(f"semantic result {key!r} has a different identity")
        return CellExecution(result, target, True)

    def run_cell(self, manifest: FaultManifest, arm: MethodArmExecution) -> CellExecution:
        manifest = validate_execution_request(manifest, self.execution)
        key = self._planned_key(manifest, arm)
        target = self.writer.result_path(key)
        if target.exists():
            result = self.writer.read(key, manifest)
            if semantic_run_key(result, manifest) != key:
                raise ResultConflictError(f"semantic result {key!r} has a different identity")
            return CellExecution(result, target, True)

        environment = self.environment_factory(manifest)
        policy: Optional[PolicyAdapter] = None
        try:
            policy = self.policy_factory(arm, manifest)
            runner = RolloutRunner(
                environment,
                policy,
                manifest,
                self.config,
                self.execution,
            )
            outcome = runner.run()
        finally:
            try:
                if policy is not None:
                    policy.close()
            finally:
                environment.close()
        result = outcome.to_result(
            manifest,
            method_id=arm.method_id,
            checkpoint_hash=arm.checkpoint_hash,
            method_config_hash=arm.method_config_hash,
            clean_or_fault="clean" if manifest.fault_mode == HEALTHY_MODE else "fault",
        )
        path = self.writer.write_semantic(result, manifest)
        return CellExecution(result, path, False)

    def run_healthy_reference(
        self,
        template: FaultManifest,
        arm: MethodArmExecution,
        *,
        benchmark: str,
        task: str,
        episode_index: int,
        protocol_version: str,
        official_horizon: int,
        replicates: int = 1,
        threshold: float = 0.5,
    ) -> tuple[HealthyReferenceSummary, list[CellExecution]]:
        """Execute the healthy replicates as atomic cells and freeze the reference.

        Every replicate is resumed from its atomic result when it already
        exists, so the healthy arm has the same resume semantics as the fault
        arms instead of a bespoke cache.
        """
        manifests = healthy_replicate_manifests(
            template,
            benchmark=benchmark,
            task=task,
            episode_index=episode_index,
            protocol_version=protocol_version,
            replicates=replicates,
        )
        executions = [self.run_cell(manifest, arm) for manifest in manifests]
        return self._summarise_healthy(manifests, executions, official_horizon, threshold)

    def peek_healthy_reference(
        self,
        template: FaultManifest,
        arm: MethodArmExecution,
        *,
        benchmark: str,
        task: str,
        episode_index: int,
        protocol_version: str,
        official_horizon: int,
        replicates: int = 1,
        threshold: float = 0.5,
    ) -> Optional[tuple[HealthyReferenceSummary, list[CellExecution]]]:
        """Summarise a unit's healthy reference only if every replicate exists."""
        manifests = healthy_replicate_manifests(
            template,
            benchmark=benchmark,
            task=task,
            episode_index=episode_index,
            protocol_version=protocol_version,
            replicates=replicates,
        )
        executions = [self.peek_cell(manifest, arm) for manifest in manifests]
        if any(execution is None for execution in executions):
            return None
        return self._summarise_healthy(manifests, executions, official_horizon, threshold)

    def _summarise_healthy(
        self,
        manifests: Sequence[FaultManifest],
        executions: Sequence[CellExecution],
        official_horizon: int,
        threshold: float,
    ) -> tuple[HealthyReferenceSummary, list[CellExecution]]:
        records = []
        for index, (manifest, execution) in enumerate(zip(manifests, executions)):
            result = execution.result
            success = bool(result.success_or_valid)
            records.append(
                HealthyReplicate(
                    replicate_index=index,
                    environment_seed=int(manifest.environment_seed),
                    policy_seed=int(manifest.policy_seed),
                    success=success,
                    completion_step=int(result.action_execution_count) if success else None,
                    result_hash=result.action_or_prediction_hash,
                )
            )
        summary = aggregate_healthy_references(
            records,
            official_horizon=official_horizon,
            threshold=threshold,
        )
        return summary, list(executions)
