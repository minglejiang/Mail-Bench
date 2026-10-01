"""Simulator-neutral rollout execution with query-clock fault auditing."""

from __future__ import annotations

import copy
import hashlib
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Optional

from .interfaces import EpisodeContext, CanonicalObservation, EnvironmentAdapter, PolicyAdapter, PolicyOutput
from .manifest import (
    HEALTHY_MODE,
    FaultManifest,
    ResultRecord,
    canonical_json,
    semantic_sha256,
    validate_manifest,
    validate_result,
)
from .operators import FaultInjector, FaultSchedule, TraceRecord
from .registry import EXECUTION_STAGES, DatasetValidation


# Digest of the empty prefix.  ``_extend_prefix`` chains one value at a time so
# that ``chain[i]`` is a commitment to elements ``0..i``; a whole-trace hash
# cannot prove that two method arms shared a trajectory prefix, a chain can.
# Frozen identifier: changing it would change every prefix chain.
EMPTY_PREFIX_HASH = hashlib.sha256(b"mcr_bench.prefix.v1").hexdigest()


def _extend_prefix(previous: str, value: Any) -> str:
    return hashlib.sha256((previous + canonical_json(value)).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ChunkInvalidation:
    step: int
    reason: str
    discarded_actions: int



def _digest(value: Any) -> Optional[str]:
    """SHA256 of an array's bytes (or of a small value's canonical JSON)."""
    if value is None:
        return None
    import hashlib

    tobytes = getattr(value, "tobytes", None)
    if tobytes is not None:
        try:
            import numpy as np

            array = np.ascontiguousarray(value)
            return hashlib.sha256(
                str(array.shape).encode() + str(array.dtype).encode() + array.tobytes()
            ).hexdigest()
        except Exception:                                   # noqa: BLE001
            pass
    return semantic_sha256(value)

class ActionChunkManager:
    """Own queued actions and an auditable invalidation trace."""

    def __init__(self) -> None:
        self._pending: deque[Any] = deque()
        self.invalidations: list[ChunkInvalidation] = []

    @property
    def pending_count(self) -> int:
        return len(self._pending)

    def load(self, output: PolicyOutput) -> None:
        if self._pending:
            raise RuntimeError("cannot load a policy output while actions remain queued")
        self._pending.extend(output.actions)

    def next_action(self) -> Any:
        if not self._pending:
            raise RuntimeError("no queued action is available")
        return self._pending.popleft()

    def invalidate(self, step: int, reason: str) -> ChunkInvalidation:
        event = ChunkInvalidation(step, reason, len(self._pending))
        self._pending.clear()
        self.invalidations.append(event)
        return event


@dataclass(frozen=True)
class RunnerConfig:
    max_environment_steps: int
    invalidate_chunk_on_availability_change: bool = True
    # Off by default: how a policy treats queued actions when vision returns
    # is model design, measured rather than imposed. Losing a camera is
    # different -- the kernel drops the queue so the fault reaches the policy
    # at the step it was scheduled for.
    invalidate_chunk_on_recovery: bool = False
    reset_visual_memory_on_recovery: bool = False
    # Matched-replanning diagnostic: cap how many actions of a chunk are executed
    # before the policy must be queried again.  ``None`` keeps the model's native
    # deployment semantics, which is what the main ranking uses; a fixed cap
    # separates "robust because it models the world" from "robust because it
    # stopped looking for 50 steps".  The value enters ``runner_config_hash``, so
    # native and matched runs can never be confused for one another.
    max_actions_per_query: Optional[int] = None

    def __post_init__(self) -> None:
        if self.max_environment_steps < 1:
            raise ValueError("max_environment_steps must be >= 1")
        if self.max_actions_per_query is not None and self.max_actions_per_query < 1:
            raise ValueError("max_actions_per_query must be >= 1 when set")


@dataclass
class RolloutAudit:
    environment_step_count: int = 0
    policy_query_count: int = 0
    action_execution_count: int = 0
    fault_reached: bool = False
    faulted_observations_produced: int = 0
    faulted_observations_consumed: int = 0
    #: The step of the first policy query that actually carried a faulted
    #: observation. Observed, not predicted: a policy queries when its action
    #: buffer empties, which need not land on a multiple of its nominal
    #: replanning interval, so computing this from the interval would report a
    #: boundary the rollout may never have used.
    realized_onset_step: Optional[int] = None
    query_steps: list[int] = field(default_factory=list)
    actions: list[Any] = field(default_factory=list)
    invalidations: list[ChunkInvalidation] = field(default_factory=list)
    availability_trace_hash: Optional[str] = None
    action_trace_hash: Optional[str] = None
    policy_query_trace_hash: Optional[str] = None
    policy_input_trace_hash: Optional[str] = None
    action_chunk_invalidation_trace_hash: Optional[str] = None
    action_prefix_hashes: list[str] = field(default_factory=list)
    policy_input_prefix_hashes: list[str] = field(default_factory=list)
    pre_fault_action_hash: Optional[str] = None
    pre_fault_policy_input_hash: Optional[str] = None
    post_fault_action_steps: int = 0
    max_actions_after_fault_query: int = 0
    native_action_chunk_length: int = 0
    mean_query_interval: Optional[float] = None


@dataclass(frozen=True)
class ExecutionBinding:
    """Hashes and seeds that prove which frozen execution produced an outcome."""

    manifest_hash: str
    schedule_hash: str
    runner_config_hash: str
    dataset_authorization_hash: str
    environment_seed: int
    policy_seed: int
    fault_seed: int


def _stale_k_for(manifest: FaultManifest) -> Optional[int]:
    """The stale delay in control steps: the protocol's 500 ms at the platform's
    control rate, or the manifest's own when it names no rate. A manifest that
    gives both must agree with itself, or a hand-filled ``stale_k=15`` on a
    20 Hz platform would quietly mean 750 ms and validate."""
    from .operators import CANONICAL_STALE_MS, control_steps_for

    if manifest.fault_mode != "stale_k":
        return manifest.stale_k
    derived = None
    if manifest.nominal_control_hz:
        derived = control_steps_for(CANONICAL_STALE_MS, float(manifest.nominal_control_hz))
    if manifest.stale_k is None:
        if derived is None:
            raise ValueError(
                "a stale manifest needs stale_k or nominal_control_hz; the protocol's "
                f"{CANONICAL_STALE_MS} ms delay cannot be turned into steps otherwise"
            )
        return derived
    if derived is not None and int(manifest.stale_k) != derived:
        raise ValueError(
            f"stale_k={manifest.stale_k} is not the protocol's {CANONICAL_STALE_MS} ms at "
            f"{manifest.nominal_control_hz} Hz ({derived} steps)"
        )
    return int(manifest.stale_k)


def _burst_frames_for(manifest: FaultManifest) -> Optional[int]:
    """The burst block in control steps: the manifest's own, or 500 ms at the
    platform's control rate. A literal step default is not the protocol's 500 ms
    and is never used."""
    from .operators import CANONICAL_BURST_BLOCK_MS, control_steps_for

    if manifest.fault_mode != "burst_dropout":
        return manifest.burst_frames
    if manifest.burst_frames is not None:
        return int(manifest.burst_frames)
    if manifest.nominal_control_hz:
        return control_steps_for(CANONICAL_BURST_BLOCK_MS, float(manifest.nominal_control_hz))
    raise ValueError(
        "a burst_dropout manifest needs burst_frames or nominal_control_hz; the "
        f"protocol's {CANONICAL_BURST_BLOCK_MS} ms block cannot be turned into steps otherwise"
    )


def schedule_from_manifest(manifest: FaultManifest) -> FaultSchedule:
    """Build the only executable availability schedule for a frozen manifest.

    A healthy cell (``fault_mode='healthy'``, no faulted camera) yields a
    schedule with an empty camera set, so the injector is a pass-through and
    the healthy arm runs through the same kernel as every fault arm.  The
    schedule ``mode`` is inert in that case because no camera is ever faulted.
    """
    manifest = validate_manifest(copy.deepcopy(manifest))
    if manifest.perturbation_family != "availability":
        raise ValueError("the availability rollout kernel requires an availability manifest")
    if manifest.fault_mode == HEALTHY_MODE:
        return FaultSchedule(
            frozenset(),
            mode="hard_missing",
            onset_step=0,
            end_step=None,
            fault_seed=int(manifest.fault_seed),
            recovery="none",
        )
    if manifest.onset_step is None:
        raise ValueError("execution requires a resolved manifest onset_step")
    recovery = manifest.recovery_mode
    if recovery is None:
        recovery = "single_recovery" if manifest.end_step is not None else "none"
    return FaultSchedule(
        frozenset(manifest.faulted_camera_ids or ()),
        mode=str(manifest.fault_mode),
        onset_step=int(manifest.onset_step),
        end_step=manifest.end_step,
        stale_k=_stale_k_for(manifest),
        burst_frames=_burst_frames_for(manifest),
        available_rate=manifest.available_rate if manifest.available_rate is not None else 0.5,
        fault_seed=int(manifest.fault_seed),
        recovery=recovery,
    )


def validate_execution_request(
    manifest: FaultManifest,
    execution: DatasetValidation,
) -> FaultManifest:
    """Fail closed before execution or resume lookup and return a frozen copy."""
    if execution.stage not in EXECUTION_STAGES:
        raise ValueError(f"unknown execution stage {execution.stage!r}")
    if execution.mock and execution.stage != "stage_0":
        raise ValueError("mock execution authorization is valid only for stage_0")
    if not execution.mock and (not execution.pinned or not execution.dataset_id):
        raise ValueError("real rollout execution requires a pinned dataset authorization")
    manifest = validate_manifest(copy.deepcopy(manifest))
    if not execution.mock and execution.revision != manifest.dataset_version:
        raise ValueError("dataset authorization revision differs from manifest.dataset_version")
    schedule_from_manifest(manifest)
    return manifest


@dataclass
class RolloutOutcome:
    audit: RolloutAudit
    success: bool
    official_metrics: dict[str, Any]
    completed: bool
    failure_reason: Optional[str]
    runtime_ms: float
    binding: ExecutionBinding

    def to_result(
        self,
        manifest: FaultManifest,
        *,
        method_id: str,
        checkpoint_hash: str,
        method_config_hash: Optional[str] = None,
        clean_or_fault: str = "fault",
    ) -> ResultRecord:
        """Create and validate the minimum atomic result for this rollout."""
        manifest = validate_manifest(copy.deepcopy(manifest))
        if manifest.semantic_hash() != self.binding.manifest_hash:
            raise ValueError("result manifest differs from the manifest used for execution")
        record = ResultRecord(
            cell_id=manifest.cell_id,
            method_id=method_id,
            checkpoint_hash=checkpoint_hash,
            clean_or_fault=clean_or_fault,
            official_metrics=dict(self.official_metrics),
            success_or_valid=self.success,
            availability_trace_hash=self.audit.availability_trace_hash,
            action_or_prediction_hash=self.audit.action_trace_hash,
            policy_query_trace_hash=self.audit.policy_query_trace_hash,
            policy_input_trace_hash=self.audit.policy_input_trace_hash,
            action_chunk_invalidation_trace_hash=(
                self.audit.action_chunk_invalidation_trace_hash
            ),
            runtime_ms=self.runtime_ms,
            failure_reason=self.failure_reason,
            completed=self.completed,
            environment_step_count=self.audit.environment_step_count,
            policy_query_count=self.audit.policy_query_count,
            action_execution_count=self.audit.action_execution_count,
            action_prefix_hash_chain=list(self.audit.action_prefix_hashes),
            policy_input_prefix_hash_chain=list(self.audit.policy_input_prefix_hashes),
            pre_fault_action_hash=self.audit.pre_fault_action_hash,
            pre_fault_policy_input_hash=self.audit.pre_fault_policy_input_hash,
            post_fault_action_steps=self.audit.post_fault_action_steps,
            max_actions_after_fault_query=self.audit.max_actions_after_fault_query,
            native_action_chunk_length=self.audit.native_action_chunk_length,
            mean_query_interval=self.audit.mean_query_interval,
            faulted_observations_produced=self.audit.faulted_observations_produced,
            faulted_observations_consumed=self.audit.faulted_observations_consumed,
            realized_onset_step=self.audit.realized_onset_step,
            onset_step=(int(manifest.onset_step) if manifest.faulted_camera_ids
                        and manifest.onset_step is not None else None),
            environment_seed=self.binding.environment_seed,
            policy_seed=self.binding.policy_seed,
            fault_seed=self.binding.fault_seed,
            manifest_hash=self.binding.manifest_hash,
            schedule_hash=self.binding.schedule_hash,
            runner_config_hash=self.binding.runner_config_hash,
            dataset_authorization_hash=self.binding.dataset_authorization_hash,
            method_config_hash=method_config_hash or semantic_sha256({}),
            onset_basis=manifest.onset_basis,
            healthy_reference_success=manifest.healthy_reference_success,
            fault_reached=self.audit.fault_reached,
            onset_reference_hash=manifest.onset_reference_hash,
            onset_fallback=manifest.onset_fallback,
        )
        return validate_result(record, manifest)


class RolloutRunner:
    """Execute one paired rollout without embedding platform-specific keys."""

    def __init__(
        self,
        environment: EnvironmentAdapter,
        policy: PolicyAdapter,
        manifest: FaultManifest,
        config: RunnerConfig,
        execution: DatasetValidation,
    ) -> None:
        manifest = validate_execution_request(manifest, execution)
        if not execution.mock:
            identity = environment.dataset_identity()
            expected_identity = (execution.dataset_id, execution.revision)
            if identity != expected_identity:
                raise ValueError(
                    f"environment dataset identity {identity!r} differs from "
                    f"authorization {expected_identity!r}"
                )
            manifest_identity = environment.manifest_identity()
            expected_manifest_identity = (
                manifest.adapter_version,
                manifest.scene_or_task,
                manifest.episode_or_sequence,
            )
            if manifest_identity != expected_manifest_identity:
                raise ValueError(
                    f"environment manifest identity {manifest_identity!r} differs from "
                    f"manifest {expected_manifest_identity!r}"
                )
        schedule = schedule_from_manifest(manifest)
        self.environment = environment
        self.policy = policy
        self.manifest = manifest
        self.schedule = schedule
        self.config = config
        self.execution = execution
        self.binding = ExecutionBinding(
            manifest_hash=manifest.semantic_hash(),
            schedule_hash=semantic_sha256(schedule),
            runner_config_hash=semantic_sha256(config),
            dataset_authorization_hash=semantic_sha256(execution),
            environment_seed=int(manifest.environment_seed),
            policy_seed=int(manifest.policy_seed),
            fault_seed=int(manifest.fault_seed),
        )

    @staticmethod
    def _apply_fault(
        observation: CanonicalObservation,
        injector: FaultInjector,
    ) -> tuple[CanonicalObservation, TraceRecord]:
        cameras, availability, trace = injector.apply(observation.step, observation.cameras)
        return observation.with_fault(
            cameras,
            availability,
            source_step=trace.source_step,
            source_age_steps=trace.source_age_steps,
        ), trace

    @staticmethod
    def _recovered(previous: dict[str, bool], current: dict[str, bool]) -> bool:
        return any(not previous[camera] and current[camera] for camera in previous)

    @staticmethod
    def _policy_input_provenance(observation: CanonicalObservation) -> dict[str, Any]:
        """What the policy saw at this query, as digests: never the tensors.

        The frame and state digests are what let a pre-onset divergence be
        attributed. Without them the chain would say only which cameras were
        available, and a pair that drew different actions from identical
        inputs would be indistinguishable from a pair whose renderer produced
        different pixels.
        """
        return {
            "step": observation.step,
            "robot_state": _digest(observation.robot_state),
            "cameras": {
                camera: {
                    "available": observation.availability[camera],
                    "source_step": observation.source_step.get(camera),
                    "source_age_steps": observation.source_age_steps.get(camera),
                    "frame": _digest(observation.cameras.get(camera)),
                }
                for camera in observation.camera_ids
            },
        }

    def run(self) -> RolloutOutcome:
        started = time.perf_counter()
        environment_seed = self.binding.environment_seed
        policy_seed = self.binding.policy_seed
        observation = self.environment.reset(environment_seed)
        if observation.step != 0:
            raise ValueError(f"reset observation must have step=0, got {observation.step}")
        inventory = tuple(sorted(str(c) for c in self.environment.camera_inventory()))
        if observation.camera_ids != inventory:
            raise ValueError(
                f"canonical camera inventory mismatch: observation={observation.camera_ids}, "
                f"adapter={inventory}"
            )
        self.policy.announce_episode(EpisodeContext(
            task=self.manifest.scene_or_task,
            episode_index=self.manifest.episode_or_sequence,
            official_horizon=self.manifest.official_horizon,
            # The scene's instruction is part of what a pair shares and is
            # known once the environment has reset; a server that builds a
            # prompt cache per episode gets it here, not first at step 0.
            instruction=observation.language or None,
        ))
        self.policy.reset(policy_seed)
        injector = FaultInjector(self.schedule)
        chunks = ActionChunkManager()
        audit = RolloutAudit()
        previous_availability: Optional[dict[str, bool]] = None
        policy_input_trace: list[dict[str, Any]] = []
        terminated = False
        truncated = False
        # A healthy cell has no onset, so it records the chains only; a fault cell
        # additionally freezes the digests at its own onset.
        onset_step = self.schedule.onset_step if self.schedule.faulted_cameras else None
        action_prefix = EMPTY_PREFIX_HASH
        policy_input_prefix = EMPTY_PREFIX_HASH
        actions_since_query = 0
        current_query_faulted = False

        while audit.action_execution_count < self.config.max_environment_steps:
            visible, trace = self._apply_fault(observation, injector)
            current_availability = dict(visible.availability)
            audit.fault_reached = audit.fault_reached or bool(
                self.schedule.faulted_cameras and trace.active
            )

            if previous_availability is not None and current_availability != previous_availability:
                recovered = self._recovered(previous_availability, current_availability)
                reason = "availability_recovery" if recovered else "availability_change"
                invalidate = (
                    self.config.invalidate_chunk_on_recovery if recovered
                    else self.config.invalidate_chunk_on_availability_change
                )
                if invalidate:
                    chunks.invalidate(observation.step, reason)
                    self.policy.invalidate_action_chunk(reason, observation.step)
                if recovered and self.config.reset_visual_memory_on_recovery:
                    self.policy.reset_visual_memory(reason, observation.step)
            previous_availability = current_availability

            if chunks.pending_count == 0:
                provenance = self._policy_input_provenance(visible)
                policy_input_trace.append(provenance)
                output = self.policy.act(visible)
                if not isinstance(output, PolicyOutput):
                    raise TypeError("PolicyAdapter.act() must return PolicyOutput")
                if current_query_faulted:
                    audit.max_actions_after_fault_query = max(
                        audit.max_actions_after_fault_query, actions_since_query
                    )
                actions_since_query = 0
                current_query_faulted = bool(trace.faulted_observation_produced)
                audit.policy_query_count += 1
                audit.query_steps.append(observation.step)
                policy_input_prefix = _extend_prefix(policy_input_prefix, provenance)
                audit.policy_input_prefix_hashes.append(policy_input_prefix)
                audit.native_action_chunk_length = max(
                    audit.native_action_chunk_length, len(output.actions)
                )
                if trace.faulted_observation_produced:
                    audit.faulted_observations_consumed += 1
                    if audit.realized_onset_step is None:
                        audit.realized_onset_step = observation.step
                chunks.load(output)

            action = chunks.next_action()
            audit.actions.append(action)
            step_result = self.environment.step(action)
            audit.action_execution_count += 1
            audit.environment_step_count = audit.action_execution_count
            action_prefix = _extend_prefix(action_prefix, action)
            audit.action_prefix_hashes.append(action_prefix)
            actions_since_query += 1
            if onset_step is not None and observation.step >= onset_step:
                audit.post_fault_action_steps += 1
            if (
                self.config.max_actions_per_query is not None
                and actions_since_query >= self.config.max_actions_per_query
                and chunks.pending_count
            ):
                chunks.invalidate(observation.step, "matched_replanning_cap")
                self.policy.invalidate_action_chunk("matched_replanning_cap", observation.step)
            if step_result.observation.step != observation.step + 1:
                raise ValueError(
                    "environment observation steps must advance by exactly one per action "
                    f"(got {step_result.observation.step} after {observation.step})"
                )
            if step_result.observation.camera_ids != inventory:
                raise ValueError(
                    "camera inventory changed during an episode: "
                    f"{step_result.observation.camera_ids} != {inventory}"
                )
            terminated = bool(step_result.terminated)
            truncated = bool(step_result.truncated)
            if terminated or truncated:
                break
            observation = step_result.observation

        if current_query_faulted:
            audit.max_actions_after_fault_query = max(
                audit.max_actions_after_fault_query, actions_since_query
            )
        if onset_step is not None:
            # Truncated when the episode ended before the onset; the auditor only
            # compares these against the healthy chain when the fault was reached.
            executed_before_onset = min(onset_step, len(audit.action_prefix_hashes))
            audit.pre_fault_action_hash = (
                audit.action_prefix_hashes[executed_before_onset - 1]
                if executed_before_onset
                else EMPTY_PREFIX_HASH
            )
            queries_before_onset = sum(1 for step in audit.query_steps if step < onset_step)
            audit.pre_fault_policy_input_hash = (
                audit.policy_input_prefix_hashes[queries_before_onset - 1]
                if queries_before_onset
                else EMPTY_PREFIX_HASH
            )
        if audit.policy_query_count:
            audit.mean_query_interval = (
                audit.action_execution_count / audit.policy_query_count
            )
        audit.faulted_observations_produced = injector.faulted_observations_produced
        audit.invalidations = list(chunks.invalidations)
        audit.availability_trace_hash = injector.availability_trace_hash()
        audit.action_trace_hash = semantic_sha256(audit.actions)
        audit.policy_query_trace_hash = semantic_sha256(audit.query_steps)
        audit.policy_input_trace_hash = semantic_sha256(policy_input_trace)
        audit.action_chunk_invalidation_trace_hash = semantic_sha256(audit.invalidations)
        completed = terminated or truncated or (
            audit.action_execution_count >= self.config.max_environment_steps
        )
        success = bool(self.environment.success())
        official_metrics = dict(self.environment.official_metrics())
        failure_reason = None
        if not success:
            if official_metrics.get("horizon_exhausted"):
                # The platform ended the episode at its own horizon.
                failure_reason = "horizon_exhausted"
            elif not terminated and not truncated:
                failure_reason = "max_environment_steps"
            elif terminated or truncated:
                failure_reason = "terminated_without_success"
        return RolloutOutcome(
            audit=audit,
            success=success,
            official_metrics=official_metrics,
            completed=completed,
            failure_reason=failure_reason,
            runtime_ms=(time.perf_counter() - started) * 1000.0,
            binding=self.binding,
        )
