"""Read stored cells and build the MAIL-Bench report from them.

Two things live here. The cell reader turns every ``*.json`` under a run
root into a :class:`CellRow` -- identity decoded from the frozen cell id, the
result record kept whole -- and refuses, as an audit failure rather than a
crash, anything unreadable, unparsable, or claiming an identity another file
already holds. Nothing downstream sees a cell that did not come through it.

:func:`mail_bench_report` is the scorer. It takes exactly the main-ranking
cells (hard missing, until the episode ends, at the three onsets, in the
three semantic states), one healthy cell per scene, one policy configuration
across every task, and turns them into the ten-condition task scores and the
eighteen-task benchmark score of :mod:`mail_bench.scoring`. Everything it
refuses is a way a run could look official without being one; what it
cannot verify it reports as provenance (zero-exposure cells, pre-fault
divergence, onset realisation) rather than corrects.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence, Union

PathLike = Union[str, Path]

#: ``FaultManifest.derived_cell_id`` is a frozen format; this is its inverse.
CELL_ID = re.compile(
    r"^(?P<task>.+)\|ep(?P<episode>-?\d+)\|(?P<mode>[a-z_]+)\|"
    r"miss=(?P<cameras>[^|]+)\|onset=(?P<onset>[^|]+)\|"
    r"(?:dur=(?P<duration>[^|]+)|rec=(?P<recovery>[^|]+))$"
)


class AggregationError(ValueError):
    """Raised when results cannot be aggregated without inventing something."""


@dataclass(frozen=True)
class CellRow:
    """One stored result, with its cell identity decoded."""

    task: str
    episode_index: int
    fault_mode: str
    faulted_cameras: tuple[str, ...]
    onset_fraction: float
    duration: str
    result: dict[str, Any]
    source: str

    @property
    def unit(self) -> tuple[str, int]:
        return (self.task, self.episode_index)

    @property
    def is_healthy(self) -> bool:
        return self.fault_mode == "healthy"

    @property
    def condition(self) -> tuple[tuple[str, ...], float]:
        return (self.faulted_cameras, self.onset_fraction)


@dataclass
class AuditFailure:
    kind: str
    detail: str
    source: str = ""


@dataclass(frozen=True)
class MeasurementKey:
    """What makes two rollouts part of the same measurement.

    A healthy reference belongs to a measurement, not to a scene: the same
    ``(task, init_state)`` evaluated by two models -- or by one model under two
    replanning profiles -- has two different healthy references, because the
    replanning interval changes the healthy trajectory too. Collapsing healthy
    cells by unit alone would hand every group the same, wrong reference.
    """

    method_id: str
    checkpoint_hash: str
    method_config_hash: str
    runner_config_hash: str
    dataset_authorization_hash: str

    @classmethod
    def of(cls, result: dict[str, Any]) -> "MeasurementKey":
        return cls(
            method_id=str(result.get("method_id")),
            checkpoint_hash=str(result.get("checkpoint_hash")),
            method_config_hash=str(result.get("method_config_hash")),
            runner_config_hash=str(result.get("runner_config_hash")),
            dataset_authorization_hash=str(result.get("dataset_authorization_hash")),
        )


def astuple_key(key: MeasurementKey) -> tuple[str, ...]:
    return (
        key.method_id,
        key.checkpoint_hash,
        key.method_config_hash,
        key.runner_config_hash,
        key.dataset_authorization_hash,
    )


def parse_cell_id(cell_id: str) -> dict[str, Any]:
    match = CELL_ID.match(str(cell_id))
    if not match:
        raise AggregationError(f"cell_id does not match the frozen format: {cell_id!r}")
    cameras = match.group("cameras")
    return {
        "task": match.group("task"),
        "episode_index": int(match.group("episode")),
        "fault_mode": match.group("mode"),
        "faulted_cameras": () if cameras == "none" else tuple(sorted(cameras.split("+"))),
        "onset_fraction": float(match.group("onset")),
        # A Visual Recovery cell ends where vision returns; its id says so.
        "duration": (match.group("duration") if match.group("duration") is not None
                     else f"rec={match.group('recovery')}"),
    }


def load_cells(root: PathLike) -> tuple[list[CellRow], list[AuditFailure]]:
    """Read every stored cell under ``root``; unreadable ones become audit rows."""
    rows: list[CellRow] = []
    failures: list[AuditFailure] = []
    seen: dict[str, str] = {}
    for path in sorted(Path(root).rglob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            failures.append(AuditFailure("unreadable_result", str(exc), str(path)))
            continue
        result = payload.get("result") if isinstance(payload, dict) else None
        if not isinstance(result, dict) or result.get("cell_id") is None:
            continue  # not a cell file (a certificate, a report, ...)
        try:
            parsed = parse_cell_id(result["cell_id"])
        except AggregationError as exc:
            failures.append(AuditFailure("unparsable_cell_id", str(exc), str(path)))
            continue
        # Two files claiming one identity would silently double-count a unit. The
        # identity must span every field that separates measurements, or two
        # wrappers around one checkpoint would collapse into each other.
        key = json.dumps(
            [result["cell_id"], result.get("policy_seed"), astuple_key(MeasurementKey.of(result))],
            sort_keys=True,
        )
        if key in seen and seen[key] != str(path):
            failures.append(
                AuditFailure("duplicate_semantic_key", f"also at {seen[key]}", str(path))
            )
            continue
        seen[key] = str(path)
        rows.append(CellRow(result=result, source=str(path), **parsed))
    return rows, failures


def read_certificates(root: PathLike) -> dict[str, Any]:
    """What the run's own certificates say about its conformance.

    The cohort script writes one ``certificate_<phase>_worker<n>.json`` per
    worker per invocation, and its ``phase`` says which phases that worker
    covered: ``healthy`` and ``fault`` for the two-invocation pipeline, or
    ``all`` for a single invocation that ran both. A healthy certificate says
    whether its worker finished its shard; a certificate covering the fault
    phase carries the deviation list and the ``leaderboard_eligible`` verdict.

    Official needs every certificate to pass its own check and both phases to
    be covered by some certificate: a root with no certificate is a pile of
    cells whose provenance nobody wrote down, and one covering only the
    healthy phase is a run that was scored before it finished. An unknown
    phase is a problem rather than a guess, since guessing would let a
    certificate this reader does not understand pass silently.
    """
    files = sorted(Path(root).glob("certificate*.json"))
    healthy, fault, deviations, problems = 0, 0, set(), []
    for path in files:
        try:
            certificate = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            problems.append(f"{path.name}: unreadable ({exc})")
            continue
        if not isinstance(certificate, dict):
            problems.append(f"{path.name}: not an object")
            continue
        phase = certificate.get("phase")
        if phase not in ("healthy", "fault", "all"):
            problems.append(f"{path.name}: unknown phase {phase!r}")
            continue
        if phase in ("healthy", "all"):
            healthy += 1
            # A single-invocation run reports its healthy completeness through
            # the cohort verdict below, not through the per-worker healthy flag.
            if phase == "healthy" and not certificate.get("healthy_phase_complete_for_worker"):
                problems.append(f"{path.name}: healthy phase incomplete for its worker")
        if phase in ("fault", "all"):
            fault += 1
            for deviation in certificate.get("deviations") or []:
                deviations.add(str(deviation))
            if not certificate.get("leaderboard_eligible"):
                problems.append(f"{path.name}: not leaderboard-eligible")
    if not files:
        problems.append("no certificate under the run root")
    else:
        if not fault:
            problems.append("no certificate covering the fault phase under the run root")
        if not healthy:
            problems.append("no certificate covering the healthy phase under the run root")
    return {
        "files": [path.name for path in files],
        "healthy_certificates": healthy,
        "fault_certificates": fault,
        "deviations": sorted(deviations),
        "problems": problems,
        "leaderboard_eligible": not problems,
    }


def _efficiency(pairs: Sequence[tuple[int, float]], median) -> dict[str, Any]:
    """Paired completion efficiency over scenes where both rollouts succeeded."""
    if not pairs:
        return {"pairs": 0, "median_delta_steps": None, "median_relative_slowdown": None}
    return {
        "pairs": len(pairs),
        "median_delta_steps": float(median([delta for delta, _ in pairs])),
        "median_relative_slowdown": float(median([rel for _, rel in pairs])),
    }


# ---------------------------------------------------------------------------
# MAIL-Bench: cells to a benchmark score
# ---------------------------------------------------------------------------
def mail_bench_report(
    root: PathLike,
    *,
    platform: str = "robocasa365",
    expected_tasks: Optional[Sequence[str]] = None,
    scenes_per_task: int = 50,
    require_official: bool = True,
) -> dict[str, Any]:
    """The official score, built from the standard cells and nothing else.

    The chain the protocol defines, end to end: healthy cells give each task its
    healthy score and the scenes a fault may be placed in; fault cells are
    grouped by semantic state and onset into the nine missing conditions; the
    ten make a task score; the eighteen make the benchmark score.

    Everything this refuses is a way a run could look official without being it.

    *A cell that is not a main-ranking cell.* The ranking is hard missing, until
    the episode ends, at one of three onsets, in one of three semantic states.
    A Failure Form blackout at 60% shares a state and an onset with a ranking
    condition and is a different experiment; matching on state and onset alone
    would let a mechanism study raise or lower an official ranking score.

    *A success the fault never reached.* The runner records whether the policy
    consumed a faulted observation, and a scene finished before the
    intervention arrived says nothing about robustness. The denominator still
    does not shrink: such a cell counts as written, not as valid, and its task
    keeps its healthy-success denominator.

    *More than one measurement.* One fixed policy configuration means one
    identity across every task -- not merely one checkpoint, since the same
    weights under two replanning profiles are two evaluated configurations.

    *A task set that is not the suite.* Eighteen task names is not the eighteen
    Atomic-Seen tasks, and a run missing scenes or missing owed fault cells is
    an experiment whatever its task count.

    *A run whose own certificate says it deviated.* The driver writes a
    certificate per worker naming every departure from the official profile;
    a score is official only if every certificate under the root is
    leaderboard-eligible and none is missing. Cells alone cannot tell an
    official run from a custom one, since both write the same records.

    *An onset that is not the healthy reference's.* Every fault cell's target
    step is recomputed here from its paired healthy completion step and must
    equal what was executed; the published onset manifest and the driver must
    not be allowed to drift apart unseen.

    What it computes beyond the ranking is the protocol's secondary reading:
    completion efficiency, paired per scene where both rollouts succeeded, as
    the median of the per-scene slowdowns and never over failed episodes.
    """
    from statistics import median

    from .onset import resolve_onset
    from .runner import EMPTY_PREFIX_HASH
    from .scoring import BenchmarkScore, TaskScore, condition_scores
    from .semantic_states import MISSING_STATES, RANKING_ONSETS, state_of
    from .suite import scenes_per_task as frozen_scenes, task_ids as frozen_tasks

    rows, failures = load_cells(root)
    if not rows:
        raise AggregationError(f"no readable cells under {root}")

    ranked_onsets = {round(float(f), 2) for f in RANKING_ONSETS}
    healthy: dict[str, dict[int, dict[str, Any]]] = {}
    fault: dict[str, list[dict[str, Any]]] = {}
    measurements: set[tuple[str, ...]] = set()
    outside = 0
    zero_exposure = 0
    paired_total = 0
    paired_identical = 0
    onset_pairs = 0
    onset_late = 0
    onset_checked = 0
    onset_mismatched: list[str] = []
    paired_not_comparable = 0
    slowdowns: dict[tuple[str, str, float], list[tuple[int, float]]] = {}
    fault_results: list[tuple[str, int, dict[str, Any]]] = []

    for row in rows:
        result = row.result
        measurements.add(astuple_key(MeasurementKey.of(result)))
        if row.is_healthy:
            per_task = healthy.setdefault(row.task, {})
            if row.episode_index in per_task:
                # One canonical healthy rollout per scene is the protocol; a
                # second one would silently pick which replicate defines the
                # task's denominator and onset. Refuse rather than choose.
                raise AggregationError(
                    f"{row.task} scene {row.episode_index} has more than one healthy "
                    "cell; the official protocol runs exactly one per scene"
                )
            per_task[row.episode_index] = result
            continue
        state = state_of(platform, row.faulted_cameras)
        main_ranking = (
            row.fault_mode == "hard_missing"
            and row.duration == "end"
            and round(float(row.onset_fraction), 2) in ranked_onsets
            and state in MISSING_STATES
        )
        if not main_ranking:
            outside += 1
            continue
        if "fault_cell_valid" not in result:
            # Absent is not false: a cell that never went through the validator
            # has no exposure verdict, and scoring it as zero exposure would
            # dress missing evidence as negative evidence.
            failures.append(AuditFailure(
                "fault_cell_valid_missing", "the cell carries no exposure verdict", row.source))
            continue
        valid = bool(result["fault_cell_valid"])
        if not valid:
            zero_exposure += 1
        realized = result.get("realized_onset_step")
        target = result.get("onset_step")
        if realized is not None and target is not None:
            onset_pairs += 1
            if int(realized) != int(target):
                onset_late += 1
        result = {**result, "_state": state, "_onset_fraction": row.onset_fraction}
        fault_results.append((row.task, row.episode_index, result))
        fault.setdefault(row.task, []).append({
            "state": state,
            "onset_fraction": row.onset_fraction,
            "episode_index": row.episode_index,
            "success": bool(result.get("success_or_valid")),
            "valid": valid,
        })

    # Whether each fault rollout's actions matched its healthy reference before
    # the fault. A second pass: cells arrive in file order, so a fault cell may
    # be read before its healthy reference. The protocol pairs on (task,
    # init_state, seed) and tolerates a stochastic policy taking a different
    # path, so divergence is provenance rather than
    # invalidity; but a reader deciding how much "45% of the healthy
    # trajectory" means for a given policy needs the rate.
    for task, episode_index, result in fault_results:
        reference = healthy.get(task, {}).get(episode_index)
        if reference is None:
            continue
        target = result.get("onset_step")
        chain = reference.get("action_prefix_hash_chain")
        # The healthy cell has no onset of its own, so its pre-fault digest is
        # the chain at the fault cell's onset: the prefix of actions 0..onset-1,
        # or the empty prefix for an onset at step zero.
        if target is None or not isinstance(chain, list) or int(target) > len(chain):
            paired_not_comparable += 1
        else:
            paired_total += 1
            expected_prefix = chain[int(target) - 1] if int(target) > 0 else EMPTY_PREFIX_HASH
            if result.get("pre_fault_action_hash") == expected_prefix:
                paired_identical += 1
        # The onset arithmetic, recomputed from the healthy cell it pairs with:
        # half_up(f * T_healthy), clamped to the task's official horizon. The
        # driver and the published onset manifest each compute this; the cell
        # is where they must agree.
        t_healthy = reference.get("action_execution_count")
        horizon = (result.get("official_metrics") or {}).get("official_horizon") or (
            (reference.get("official_metrics") or {}).get("official_horizon"))
        if (target is not None and reference.get("success_or_valid")
                and t_healthy is not None and horizon):
            fraction = next((f for f in ranked_onsets
                             if f == round(float(result["_onset_fraction"]), 2)), None)
            if fraction is not None:
                expected_onset = resolve_onset(
                    fraction, int(horizon), healthy_completion_step=int(t_healthy),
                    healthy_success=True)["onset_step"]
                onset_checked += 1
                if int(target) != int(expected_onset):
                    onset_mismatched.append(str(result.get("cell_id")))
        # Completion efficiency, paired on scenes where both rollouts succeeded.
        # A failed rollout has no completion time and is never given the
        # horizon as one.
        if (reference.get("success_or_valid") and result.get("success_or_valid")
                and bool(result.get("fault_cell_valid"))
                and t_healthy and result.get("action_execution_count") is not None):
            t_fault = int(result["action_execution_count"])
            delta = t_fault - int(t_healthy)
            slowdowns.setdefault(
                (task, result["_state"], round(float(result["_onset_fraction"]), 2)), []
            ).append((delta, delta / int(t_healthy)))

    if len(measurements) != 1:
        raise AggregationError(
            f"{len(measurements)} distinct measurement identities are present under "
            f"{root}. An official run is one fixed policy configuration across every "
            "task: the same weights under two replanning profiles, or two dataset "
            "authorizations, are two evaluated configurations and may not be averaged "
            "into one score."
        )
    configuration = sorted(measurements)[0]

    tasks: list[Any] = []
    observed: dict[str, int] = {}
    for task in sorted(healthy):
        solved = healthy[task]
        solved_episodes = sorted(index for index, r in solved.items()
                                 if r.get("success_or_valid"))
        successes = len(solved_episodes)
        # The denominator is the protocol's scene count, never the number of
        # cells that happen to be present: padding a short healthy phase up to
        # the official figure would hide exactly the gap it should surface, and
        # dividing by what is present would let a truncated run score on it.
        observed[task] = len(solved)
        tasks.append(TaskScore(
            task=task,
            scenes=scenes_per_task,
            healthy_successes=successes,
            conditions=condition_scores(
                fault.get(task, []),
                healthy_successes=successes,
                # Exactly the scenes this policy solved, not merely as many.
                healthy_episodes=solved_episodes,
            ),
        ))

    # An official score is the frozen suite or it is not official. Letting a
    # caller name five tasks of six scenes and calling the result official would
    # let anyone redefine the benchmark by argument.
    expected = tuple(expected_tasks) if expected_tasks is not None else None
    official_shape = (
        platform == "robocasa365"
        and (expected is None or tuple(sorted(expected)) == tuple(sorted(frozen_tasks())))
        and scenes_per_task == frozen_scenes()
    )
    benchmark = BenchmarkScore(
        tuple(tasks),
        expected_tasks=len(expected) if expected else len(frozen_tasks()),
    )
    certificates = read_certificates(root)

    # Coverage of execution, task by task, before anything is called official.
    healthy_complete = all(count == scenes_per_task for count in observed.values())
    # None means the suite could not be checked, which is not the same as
    # checking it and finding it right.
    task_set_verified = expected is not None
    task_set_exact = task_set_verified and sorted(observed) == sorted(expected)
    faults_complete = all(
        condition.complete for task in tasks for condition in task.conditions
    )
    onsets_consistent = not onset_mismatched
    official = bool(
        official_shape and benchmark.official and healthy_complete
        and task_set_exact and faults_complete and not failures
        and certificates["leaderboard_eligible"] and onsets_consistent
    )

    record = benchmark.as_record()
    all_pairs: list[tuple[int, float]] = []
    for entry in record["tasks"]:
        entry["healthy_cells_observed"] = observed.get(entry["task"], 0)
        entry["healthy_phase_complete"] = observed.get(entry["task"], 0) == scenes_per_task
        for key, condition in entry["conditions"].items():
            state, onset = key.split("@")
            pairs = slowdowns.get((entry["task"], state, round(float(onset), 2)), [])
            all_pairs.extend(pairs)
            condition["completion_efficiency"] = _efficiency(pairs, median)
    record.update({
        "platform": platform,
        "policy_configuration": configuration[1],
        "measurement_identity": list(configuration),
        "scenes_per_task": scenes_per_task,
        "healthy_phase_complete": healthy_complete,
        "task_set_verified": task_set_verified,
        "task_set_exact": task_set_exact,
        "official_shape": official_shape,
        "fault_coverage_complete": faults_complete,
        "audit_failures": [failure.kind for failure in failures],
        "cells_outside_main_ranking": outside,
        # Executed, but the fault never reached the policy. Reported rather than
        # hidden: it is a property of the scene, not a defect in the run.
        "zero_exposure_fault_cells": zero_exposure,
        # The fault is scheduled for a step and is observed at the first
        # policy query that carried it; with chunk invalidation on at onset
        # the two coincide, and a difference is reported, never corrected.
        "onset_realization": {
            "fault_cells": onset_pairs,
            "realized_after_target": onset_late,
        },
        "pre_fault_trajectory": {
            "fault_cells_with_reference": paired_total,
            "identical_to_healthy": paired_identical,
            "diverged_before_fault": paired_total - paired_identical,
            "divergence_rate": (
                (paired_total - paired_identical) / paired_total if paired_total else None),
            # Paired, but with no chain long enough to compare: reported, not
            # folded into either count.
            "not_comparable": paired_not_comparable,
        },
        # The onset each fault cell executed, recomputed from its healthy
        # reference. A mismatch is a run whose manifest and driver disagree,
        # which no score may paper over.
        "onset_check": {
            "fault_cells_checked": onset_checked,
            "mismatched": len(onset_mismatched),
            "mismatched_cells": onset_mismatched[:20],
        },
        # Secondary, and the protocol's: the median over scenes of the paired
        # per-scene slowdown, over every ranked condition of every task.
        "completion_efficiency": _efficiency(all_pairs, median),
        "certificates": certificates,
        "official": official,
        # The arithmetic, whatever the run is. A synthetic suite or a partial
        # one still has a task-score average, and reporting it under a neutral
        # name keeps the number available for analysis while the benchmark's own
        # name stays reserved for a run that earned it.
        "score": benchmark.score if benchmark.official else None,
        "mail_bench_score": None,
    })
    if official:
        record["mail_bench_score"] = record["score"]
    elif require_official:
        reasons = [name for name, ok in (
            ("frozen suite shape", official_shape),
            ("task count", benchmark.official),
            ("verified task set", task_set_verified),
            ("exact task set", task_set_exact),
            ("healthy coverage", healthy_complete),
            ("fault coverage", faults_complete), ("clean audit", not failures),
            ("leaderboard-eligible certificates", certificates["leaderboard_eligible"]),
            ("onsets consistent with the healthy reference", onsets_consistent),
        ) if not ok]
        raise AggregationError(
            f"this run is an experiment rather than an official submission: "
            f"{', '.join(reasons)} did not pass. Pass require_official=False to "
            "build the record anyway."
        )
    return record
