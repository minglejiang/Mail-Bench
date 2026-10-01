"""Cohort orchestration: the order things happen in, and nothing else.

This module re-states no scientific rule.  Horizons come from the platform
profile, onsets and subsets from the protocol config and
:mod:`mail_bench.semantic_states`, replicate counts and the solvability threshold from
:mod:`mail_bench.experiment`, the ranking rules from :mod:`mail_bench.manifest`.
What lives here is the sequence, which is itself part of the protocol:

    platform preflight
      -> reference preflight (only for this project's own baselines)
      -> runtime contract          (resolved from the live policy server)
      -> execution profile
      -> healthy replicates
      -> admission
      -> freeze onset from the healthy reference
      -> fault cells
      -> cohort certificate

The order matters: an onset may not be frozen before the healthy reference that
defines it exists, and a cell may not be ranked before the policy's camera scope
is resolved.  Worker and device placement appear only in the certificate, never
in a cell's identity, so the same cell keeps one identity wherever it runs.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Optional, Sequence

from .cohort import (
    AdmissionReport,
    CohortError,
    ExecutionProfile,
    FaultCellSpec,
    PlatformPreflight,
    admission_gate,
    fault_cell_specs,
)
from .experiment import (
    CellExecution,
    MethodArmExecution,
    PairedCellRunner,
    freeze_fault_manifest,
)
from .manifest import PROTOCOL_VERSION, FaultManifest, semantic_sha256
from .policy_api.contract import PolicyDeclaration


@dataclass(frozen=True)
class CohortSpecification:
    """Everything that defines a cohort, independent of where it runs."""

    benchmark: str
    suite: str
    dataset_id: str
    dataset_revision: str
    tasks: tuple[str, ...]
    episode_indices: tuple[int, ...]
    platform_camera_ids: tuple[str, ...]
    onset_fractions: tuple[float, ...]
    official_horizon: int
    execution_profile: str
    protocol_version: str = PROTOCOL_VERSION

    def specification_hash(self) -> str:
        """Identity of the cohort itself; every worker of a run shares it."""
        return semantic_sha256(self)


@dataclass(frozen=True)
class OfficialProfile:
    """One frozen evaluation profile: what "comparable to the official ranking" means.

    It is a profile, not a fence.  A run that differs from it is executed and
    reported exactly the same way; it simply records which choices differed and
    stops claiming comparability with the official ranking.
    """

    suite: str
    tasks: tuple[str, ...]
    episode_indices: tuple[int, ...]
    onset_fractions: tuple[float, ...]
    platform_camera_ids: tuple[str, ...]
    official_horizon: int
    # Comparability is about the evaluation distribution, so the profile pins the
    # data it was defined on: a different suite, dataset revision or protocol is
    # a different distribution, whatever the rest of the grid looks like.
    dataset_id: Optional[str] = None
    dataset_revision: Optional[str] = None
    protocol_version: str = PROTOCOL_VERSION
    execution_profile: str = "native"
    healthy_replicates: int = 1
    cohort_target: float = 0.80


#: Tags recorded when a run departs from the official profile. None of them is an
#: error: they say what was different, so a reader never has to guess.
DEVIATIONS = (
    "different_protocol",
    "different_suite",
    "different_dataset",
    "different_dataset_revision",
    "custom_task_subset",
    "custom_init_state_subset",
    "custom_onset_grid",
    "custom_camera_subset",
    "custom_horizon",
    "custom_execution_profile",
    "custom_healthy_replicates",
    "custom_admission_target",
    "admission_gate_not_enforced",
)


def profile_deviations(
    specification: CohortSpecification,
    profile: Optional[OfficialProfile],
    *,
    healthy_replicates: int,
    cohort_target: float,
    enforce_admission: bool,
) -> tuple[str, ...]:
    """List how a run differs from the official profile, in stable order."""
    found: list[str] = []
    if profile is None:
        # Some choices are recordable without a profile to compare against.
        found.append("no_official_profile_declared")
        if not enforce_admission:
            found.append("admission_gate_not_enforced")
        return tuple(found)
    # Identity of the evaluation distribution first: everything below only means
    # something if these already match.
    if specification.protocol_version != profile.protocol_version:
        found.append("different_protocol")
    if specification.suite != profile.suite:
        found.append("different_suite")
    if profile.dataset_id is not None and specification.dataset_id != profile.dataset_id:
        found.append("different_dataset")
    if (
        profile.dataset_revision is not None
        and specification.dataset_revision != profile.dataset_revision
    ):
        found.append("different_dataset_revision")
    if tuple(specification.tasks) != tuple(profile.tasks):
        found.append("custom_task_subset")
    if tuple(specification.episode_indices) != tuple(profile.episode_indices):
        found.append("custom_init_state_subset")
    if tuple(specification.onset_fractions) != tuple(profile.onset_fractions):
        found.append("custom_onset_grid")
    if tuple(specification.platform_camera_ids) != tuple(profile.platform_camera_ids):
        found.append("custom_camera_subset")
    if int(specification.official_horizon) != int(profile.official_horizon):
        found.append("custom_horizon")
    if specification.execution_profile != profile.execution_profile:
        found.append("custom_execution_profile")
    if int(healthy_replicates) != int(profile.healthy_replicates):
        found.append("custom_healthy_replicates")
    if float(cohort_target) != float(profile.cohort_target):
        found.append("custom_admission_target")
    if not enforce_admission:
        found.append("admission_gate_not_enforced")
    return tuple(found)


@dataclass(frozen=True)
class ExecutedFaultCell:
    """A fault cell together with its execution.

    The spec's ``scope_note`` is provenance about the (policy, cell) pair --
    whether the faulted cameras are ones this policy declared it reads -- and
    is kept here rather than in the manifest so one execution can be
    re-analysed under a corrected declaration without re-running a rollout. It
    never changes what is ranked: every cell of the grid is.
    """

    spec: FaultCellSpec
    execution: CellExecution

    @property
    def scope_note(self) -> Optional[str]:
        return self.spec.scope_note


@dataclass
class UnitOutcome:
    """One ``(task, episode_index)`` unit after healthy, admission and faults."""

    task: str
    episode_index: int
    clean_solvable: bool
    success_probability: float
    healthy_completion_step: Optional[int]
    policy_determinism: str
    healthy_cells: list[CellExecution] = field(default_factory=list)
    fault_cells: list[ExecutedFaultCell] = field(default_factory=list)
    skipped_fault_cells: int = 0
    fault_skip_reason: Optional[str] = None


@dataclass
class CohortReport:
    specification: CohortSpecification
    specification_hash: str
    declaration: PolicyDeclaration
    profile: ExecutionProfile
    units: list[UnitOutcome]
    admission: AdmissionReport
    worker_index: int
    workers: int
    certificate: dict[str, Any]


def _determinism(cells: Sequence[CellExecution]) -> str:
    """Distinct policy seeds that yield one action trace mean a deterministic policy."""
    if len(cells) < 2:
        return "unknown"
    traces = {execution.result.action_or_prediction_hash for execution in cells}
    return "deterministic" if len(traces) == 1 else "stochastic"


class CohortRunner:
    """Execute one cohort shard in the protocol's order."""

    def __init__(
        self,
        *,
        specification: CohortSpecification,
        platform: PlatformPreflight,
        declaration: PolicyDeclaration,
        profile: ExecutionProfile,
        paired: PairedCellRunner,
        arm: MethodArmExecution,
        # (task, episode_index, faulted_cameras, onset_fraction) -> template
        manifest_template: Callable[[str, int, Sequence[str], float], FaultManifest],
        healthy_replicates: int = 1,
        cohort_target: float = 0.80,
        official_profile: Optional[OfficialProfile] = None,
        enforce_admission: Optional[bool] = None,
        worker_index: int = 0,
        workers: int = 1,
        runtime_metadata: Optional[Mapping[str, Any]] = None,
        semantic_platform: Optional[str] = None,
        fault_duration_fraction: float = 1.0,
        fault_recovery_fraction: Optional[float] = None,
    ) -> None:
        self.specification = specification
        # The mechanism analyses (Missing Duration, Visual Recovery) end the fault
        # before the episode does; the main ranking leaves both at their defaults.
        self.fault_duration_fraction = float(fault_duration_fraction)
        self.fault_recovery_fraction = fault_recovery_fraction
        self.platform = platform
        self.declaration = declaration
        self.profile = profile
        self.paired = paired
        self.arm = arm
        self.manifest_template = manifest_template
        self.healthy_replicates = healthy_replicates
        self.cohort_target = cohort_target
        # A run that departs from the official profile is still a real experiment:
        # it executes and reports normally and simply records what differed. Only
        # comparability with the official ranking is withdrawn, never validity.
        self.official_profile = official_profile
        # The only admission stop is a cohort with no healthy success at all. It
        # is enforced by default for an official run; a custom run may switch it
        # off, and the certificate records that it did.
        self.enforce_admission = (
            enforce_admission if enforce_admission is not None else official_profile is not None
        )
        self._summaries: dict[tuple[str, int], Any] = {}
        self._admission_outcomes: dict[tuple[str, int], UnitOutcome] = {}
        self.worker_index = worker_index
        self.workers = workers
        self.runtime_metadata = dict(runtime_metadata or {})
        # The ranking's three functional roles, when the platform has them.
        # Left unset, the grid is the enumeration of camera subsets, which a
        # diagnostic run may want.
        self.semantic_platform = semantic_platform

    def _fault_specs(self, task: str, episode_index: int) -> tuple[FaultCellSpec, ...]:
        return fault_cell_specs(
            [(task, episode_index)],
            platform_camera_ids=self.specification.platform_camera_ids,
            policy_visible_cameras=self.declaration.policy_visible_cameras,
            onset_fractions=self.specification.onset_fractions,
            semantic_platform=self.semantic_platform,
        )

    # --- Phase A: healthy -------------------------------------------------

    def run_healthy_phase(self, units: Sequence[tuple[str, int]]) -> list[UnitOutcome]:
        """Run this worker's healthy replicates. No fault cell may exist yet."""
        return [self._healthy_for(task, episode_index) for task, episode_index in units]

    def _healthy_for(self, task: str, episode_index: int) -> UnitOutcome:
        summary, healthy_cells = self.paired.run_healthy_reference(
            self.manifest_template(task, int(episode_index), (), 0.0),
            self.arm,
            benchmark=self.specification.benchmark,
            task=task,
            episode_index=episode_index,
            protocol_version=self.specification.protocol_version,
            official_horizon=self.specification.official_horizon,
            replicates=self.healthy_replicates,
        )
        return self._outcome(task, episode_index, summary, healthy_cells)

    def _peek_healthy(self, task: str, episode_index: int) -> Optional[UnitOutcome]:
        peeked = self.paired.peek_healthy_reference(
            self.manifest_template(task, int(episode_index), (), 0.0),
            self.arm,
            benchmark=self.specification.benchmark,
            task=task,
            episode_index=episode_index,
            protocol_version=self.specification.protocol_version,
            official_horizon=self.specification.official_horizon,
            replicates=self.healthy_replicates,
        )
        if peeked is None:
            return None
        summary, healthy_cells = peeked
        return self._outcome(task, episode_index, summary, healthy_cells)

    def _outcome(self, task, episode_index, summary, healthy_cells) -> UnitOutcome:
        self._summaries[(task, int(episode_index))] = summary
        return UnitOutcome(
            task=task,
            episode_index=int(episode_index),
            clean_solvable=summary.clean_solvable,
            success_probability=summary.success_probability,
            healthy_completion_step=summary.healthy_completion_step,
            policy_determinism=_determinism(healthy_cells),
            healthy_cells=list(healthy_cells),
        )

    # --- Global barrier: admission over the WHOLE frozen unit list ---------

    def resolve_admission(self, all_units: Sequence[tuple[str, int]]) -> AdmissionReport:
        """Admission is a property of the model over the complete cohort.

        Computing it per shard would let the same model be admitted by one worker
        and rejected by another, so this reads every unit's healthy reference --
        including units another worker produced -- and refuses to decide until the
        whole frozen list is present.
        """
        outcomes: list[UnitOutcome] = []
        missing: list[tuple[str, int]] = []
        for task, episode_index in all_units:
            unit = (str(task), int(episode_index))
            outcome = self._peek_healthy(*unit)
            if outcome is None:
                missing.append(unit)
            else:
                outcomes.append(outcome)
        if missing:
            raise CohortError(
                f"the admission barrier needs every unit's healthy reference; {len(missing)} "
                f"are still missing (e.g. {missing[:3]}). Run the healthy phase on every "
                "shard before any fault cell."
            )
        self._admission_outcomes = {
            (outcome.task, outcome.episode_index): outcome for outcome in outcomes
        }
        return admission_gate(
            [((o.task, o.episode_index), o.clean_solvable) for o in outcomes],
            cohort_target=self.cohort_target,
        )

    # --- Phase B: faults, once every healthy reference exists ---------------

    def run_fault_phase(
        self,
        units: Sequence[tuple[str, int]],
        admission: AdmissionReport,
    ) -> list[UnitOutcome]:
        """Execute this worker's fault cells; refuses only when no unit has a healthy success."""
        status = cohort_admission_status(admission)
        # Only an incompatible checkpoint stops here. Low clean performance is
        # a result, not an admission failure: it is published as H, and the
        # scenes the policy did solve still define their own task phases and
        # still run their fault cells. Only a cohort with no healthy success
        # anywhere stops, because there is nothing to run.
        if status == "checkpoint_incompatible" and self.enforce_admission:
            raise CohortError(
                "no unit has a healthy success, so no scene can define a task phase "
                "and no fault cell exists to run. This is a compatibility problem, "
                "not a robustness result. Pass enforce_admission=False to execute "
                "anyway; the certificate will record that the gate was not enforced."
            )
        outcomes = []
        for task, episode_index in units:
            unit = (str(task), int(episode_index))
            outcome = self._admission_outcomes.get(unit) or self._peek_healthy(*unit)
            if outcome is None:
                raise CohortError(f"unit {unit} has no healthy reference; run phase A first")
            summary = self._summaries[unit]
            specs = self._fault_specs(*unit)
            if not summary.clean_solvable:
                # No successful healthy trajectory means no task phase, so
                # there is no 30, 45 or 60 percent of anything to fault. The unit
                # is a clean failure and keeps its place in the clean and overall
                # rates; its fault cells are not executed, which is different from
                # scoring them as failures.
                outcome.skipped_fault_cells = len(specs)
                outcome.fault_skip_reason = "healthy_unsolved"
                outcomes.append(outcome)
                continue
            outcome.fault_cells = []
            for spec in specs:
                manifest = freeze_fault_manifest(
                    self.manifest_template(
                        spec.task, spec.episode_index, spec.faulted_cameras, spec.onset_fraction
                    ),
                    summary,
                    onset_fraction=spec.onset_fraction,
                    duration_fraction=self.fault_duration_fraction,
                    recovery_fraction=self.fault_recovery_fraction,
                )
                outcome.fault_cells.append(
                    ExecutedFaultCell(spec, self.paired.run_cell(manifest, self.arm))
                )
            outcomes.append(outcome)
        return outcomes

    def run(
        self,
        units: Sequence[tuple[str, int]],
        all_units: Optional[Sequence[tuple[str, int]]] = None,
    ) -> CohortReport:
        """Healthy for this shard, admission over ``all_units``, then faults.

        ``all_units`` defaults to this shard, which is correct only for a
        single-worker cohort; a sharded run must pass the complete frozen list so
        the admission decision is the model's, not the worker's.
        """
        complete = tuple(all_units) if all_units is not None else tuple(units)
        executed = {
            (outcome.task, outcome.episode_index): outcome
            for outcome in self.run_healthy_phase(units)
        }
        admission = self.resolve_admission(complete)
        # The barrier reads every unit back from disk, which always reports
        # resumed=True. This worker's own outcomes carry the truthful flags, so
        # they take precedence over the peeked copies.
        self._admission_outcomes.update(executed)
        status = cohort_admission_status(admission)
        outcomes: list[UnitOutcome]
        # Same condition as run_fault_phase.
        if status != "checkpoint_incompatible" or not self.enforce_admission:
            outcomes = self.run_fault_phase(units, admission)
        else:
            outcomes = [
                self._admission_outcomes[(str(task), int(index))] for task, index in units
            ]
        return self._report(units, complete, outcomes, admission)

    def expected_fault_cells_per_unit(self) -> int:
        """The cells a clean-solvable unit owes: every missing state at every onset."""
        from .semantic_states import MISSING_STATES

        return len(MISSING_STATES) * len(self.specification.onset_fractions)

    def identity_block(self) -> dict[str, Any]:
        """What both certificates say about the run: protocol, weights, declaration.

        The healthy-phase certificate carries all of it, so an onset manifest can
        be built from a run that executes its phases separately, which is how
        every official run executes them.
        """
        return {
            "cohort_specification_hash": self.specification.specification_hash(),
            "protocol_version": self.specification.protocol_version,
            "cohort_specification": dataclasses.asdict(self.specification),
            "benchmark_commit": self.platform.benchmark_commit,
            "dataset": {
                "id": self.specification.dataset_id,
                "revision": self.specification.dataset_revision,
            },
            "suite": self.specification.suite,
            "platform_camera_ids": list(self.specification.platform_camera_ids),
            "measurement_identity": {
                "method_id": self.arm.method_id,
                "checkpoint_hash": self.arm.checkpoint_hash,
                "method_config_hash": self.arm.method_config_hash,
                "runner_config_hash": semantic_sha256(self.paired.config),
                "dataset_authorization_hash": semantic_sha256(self.paired.execution),
            },
            "policy_declaration": {
                "model_id": self.declaration.model_id,
                "checkpoint_sha256": self.declaration.checkpoint_sha256,
                "policy_visible_cameras": list(self.declaration.policy_visible_cameras),
                "availability_consumed_by_policy": (
                    self.declaration.availability_consumed_by_policy
                ),
                "predicted_action_chunk": self.declaration.predicted_action_chunk,
                "native_execution_horizon": self.declaration.native_execution_horizon,
                "stateful_policy": self.declaration.stateful_policy,
                "reset_semantics": self.declaration.reset_semantics,
                "training_regime": self.declaration.training_regime,
                "training_disclosure": dict(self.declaration.training_disclosure),
            },
            "execution_profile": {
                "name": self.profile.name,
                "max_actions_per_query": self.profile.max_actions_per_query,
            },
            "runtime": {
                "worker_index": self.worker_index,
                "workers": self.workers,
                **self.runtime_metadata,
            },
        }

    def healthy_certificate(self, units, outcomes) -> dict[str, Any]:
        """The healthy phase's own certificate: identity plus what each unit did."""
        requested = [(str(task), int(index)) for task, index in units]
        done = {(o.task, o.episode_index) for o in outcomes}
        return {
            **self.identity_block(),
            "phase": "healthy",
            "worker_unit_list_hash": semantic_sha256([list(unit) for unit in requested]),
            "units": [
                {
                    "task": outcome.task,
                    "episode_index": outcome.episode_index,
                    "clean_solvable": outcome.clean_solvable,
                    "success_probability": outcome.success_probability,
                    "healthy_completion_step": outcome.healthy_completion_step,
                    "policy_determinism": outcome.policy_determinism,
                }
                for outcome in outcomes
            ],
            # Earned, not asserted: every requested unit wrote its healthy cell.
            "healthy_phase_complete_for_worker": all(unit in done for unit in requested),
            "next": "run every worker's healthy phase, then --phase fault",
        }

    def _report(self, units, complete, outcomes, admission) -> CohortReport:
        deviations = profile_deviations(
            self.specification,
            self.official_profile,
            healthy_replicates=self.healthy_replicates,
            cohort_target=self.cohort_target,
            enforce_admission=self.enforce_admission,
        )
        requested = [(str(task), int(index)) for task, index in units]
        by_unit = {(o.task, o.episode_index): o for o in outcomes}
        owed = self.expected_fault_cells_per_unit()
        # Completeness is counted, never declared: every requested unit has an
        # outcome, and every clean-solvable one has all the fault cells it owes.
        # A worker that ran a shard is complete for its shard; the official
        # matrix is complete only when the profile is met and nothing is owed.
        requested_complete = all(unit in by_unit for unit in requested) and all(
            (not outcome.clean_solvable) or len(outcome.fault_cells) == owed
            for outcome in by_unit.values()
        )
        results_written = all(
            cell.execution is not None and cell.execution.result is not None
            for outcome in by_unit.values() for cell in outcome.fault_cells
        )
        certificate = {
            **self.identity_block(),
            "worker_unit_list_hash": semantic_sha256([list(unit) for unit in units]),
            "cohort_unit_list_hash": semantic_sha256([list(unit) for unit in complete]),
            "seed_projection": {
                "target_bits": self.platform.seed_projection.target_bits,
                "collisions": len(self.platform.seed_projection.collisions),
            },
            "admission": {
                "scope": "whole_cohort",
                "units_considered": len(complete),
                "clean_success_rate": admission.clean_success_rate,
                "cohort_target": admission.cohort_target,
                "meets_cohort_target": admission.meets_cohort_target,
                "admitted_units": len(admission.admitted_units),
                "rejected_units": len(admission.rejected_units),
                "status": cohort_admission_status(admission),
            },
            # Not a deviation: a scene without a successful healthy rollout has no
            # task phase, so the benchmark does not execute its fault cells at all.
            "fault_cells_skipped_healthy_unsolved": sum(
                outcome.skipped_fault_cells for outcome in outcomes),
            # Provenance about the policy's declared camera scope. Every cell is
            # ranked; these say how many of them removed only cameras the policy
            # declared it never reads, or ran without a declared scope at all.
            "fault_cells_by_scope_note": {
                note: sum(
                    1
                    for outcome in outcomes
                    for cell in outcome.fault_cells
                    if cell.scope_note == note
                )
                for note in ("no_op_for_declared_scope", "policy_scope_unresolved")
            },
            # Conformance, not validity: a custom experiment is a real result that
            # simply is not comparable to the official ranking.
            "result_valid": results_written,
            "official_protocol_conformant": not deviations,
            # Completeness of what was asked for, and completeness as the official
            # matrix, are different claims: running two of eighteen official tasks
            # with no gaps is a complete custom experiment and an incomplete
            # official one. Both are counted from the outcomes.
            "fault_cells_owed_per_solvable_unit": owed,
            "matrix_complete_for_requested_experiment": requested_complete,
            "official_matrix_complete": (
                not deviations and requested_complete and results_written
                and sorted(requested) == sorted((str(a), int(b)) for a, b in complete)
            ),
            "leaderboard_eligible": not deviations and requested_complete and results_written,
            "official_profile": self.official_profile.suite if self.official_profile else None,
            "deviations": list(deviations),
        }
        return CohortReport(
            specification=self.specification,
            specification_hash=self.specification.specification_hash(),
            declaration=self.declaration,
            profile=self.profile,
            units=outcomes,
            admission=admission,
            worker_index=self.worker_index,
            workers=self.workers,
            certificate=certificate,
        )


def cohort_admission_status(admission: AdmissionReport) -> str:
    """The cohort's admission status.

    A diagnostic, not an eligibility decision. ``insufficient_clean_success`` is
    reported and published; only ``checkpoint_incompatible`` stops a fault phase,
    and then because there is nothing to run rather than as a judgement.
    """
    if not admission.admitted_units:
        # Nothing was solvable at all: a compatibility problem, not a robustness
        # result, and never reported as one.
        return "checkpoint_incompatible"
    if not admission.meets_cohort_target:
        return "insufficient_clean_success"
    return "admitted"


def require_resolved_scope(declaration: PolicyDeclaration) -> None:
    """A cohort may not start on an unresolved camera scope."""
    if not declaration.policy_visible_cameras:
        raise CohortError(
            "the policy's camera scope is unresolved; a cohort cannot rank "
            "cells before it is known which cameras may influence the policy"
        )
