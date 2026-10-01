"""Fail-closed preflight for a cohort run.

A cohort run may only start when every one of the following holds; each
is checked here and raises :class:`CohortError` rather than degrading silently:

1. the protocol is the frozen protocol version (:data:`PROTOCOL_VERSION`);
2. the model is an entry of the frozen roster (reference baselines by default);
3. the dataset is pinned in the registry (delegated to :mod:`mail_bench.registry`);
4. the checkpoint that this (model, suite) pair runs is pinned by revision and
   SHA256, together with every declared upstream dependency;
5. the benchmark worktree is clean, so the run certificate is unambiguous;
6. no two protocol seeds of the cohort collide after backend projection;
7. every unit's healthy reference exists and at least one scene has a healthy
   success (evaluated once the healthy arms have run).

This module adds no scientific rule of its own: the
onset/subset/validity semantics stay in :mod:`mail_bench.onset`,
:mod:`mail_bench.semantic_states` and :mod:`mail_bench.manifest`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence, Union

from .manifest import PROTOCOL_VERSION
from .registry import DatasetValidation, RegistryError, load_and_validate_dataset, load_yaml
from .policy_api.contract import PolicyDeclaration
from .seeds import SeedProjectionAudit, audit_seed_projection, episode_seed

PathLike = Union[str, Path]


class CohortError(ValueError):
    """Raised when a cohort may not start."""


@dataclass(frozen=True)
class ModelEntry:
    """One validated, pinned roster entry, resolved for a single suite.

    Checkpoint identity is per ``(model_id, suite)``.  A model may publish one
    checkpoint per suite, so a cell must bind the SHA256 of the checkpoint that
    suite actually used; a single blurred model-level hash
    would make two different weights share one method identity.
    """

    model_id: str
    suite: str
    display_name: str
    category: str
    family: str
    main_table: bool
    adaptation: str
    training_regime: str
    checkpoint_scope: str
    model_revision: str
    checkpoint_sha256: str
    checkpoint_path: Optional[str]
    checkpoint_provenance: str
    upstream_code_revision: Optional[str]
    dependency_revisions: tuple[tuple[str, str], ...]
    policy_visible_cameras: Optional[tuple[str, ...]]
    native_action_chunk: Optional[int]
    #: Actions executed before the policy is re-queried in its published evaluation.
    native_execution_horizon: Optional[int]
    #: 'deterministic' when replicates with distinct policy seeds share one action trace.
    policy_determinism: Optional[str]


@dataclass(frozen=True)
class PlatformPreflight:
    """What must hold before any rollout, whoever the policy is.

    Nothing here mentions a model: the same checks protect a reference baseline
    and a third party's own server equally.
    """

    protocol_version: str
    dataset: DatasetValidation
    benchmark_commit: str
    seed_projection: SeedProjectionAudit
    unit_count: int
    platform_camera_ids: tuple[str, ...]


@dataclass(frozen=True)
class CohortPreflight:
    """Platform preflight plus, for a reference baseline, its roster entry.

    ``model`` is ``None`` for a third-party submission: the reference roster is
    this project's baseline registry, not an admission registry, so a custom
    policy reaches the runtime contract without appearing in any config file.
    """

    platform: PlatformPreflight
    model: Optional[ModelEntry] = None

    @property
    def protocol_version(self) -> str:
        return self.platform.protocol_version

    @property
    def dataset(self) -> DatasetValidation:
        return self.platform.dataset

    @property
    def benchmark_commit(self) -> str:
        return self.platform.benchmark_commit

    @property
    def seed_projection(self) -> SeedProjectionAudit:
        return self.platform.seed_projection

    @property
    def unit_count(self) -> int:
        return self.platform.unit_count


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise CohortError(message)


def load_model_entry(
    roster_path: PathLike,
    model_id: str,
    *,
    suite: str,
    track: Optional[str] = None,
    require_main_table: bool = True,
) -> ModelEntry:
    """Validate one roster entry for one suite, refusing anything not ready."""
    roster = load_yaml(roster_path)
    declared = roster.get("protocol_version")
    _require(
        declared == PROTOCOL_VERSION,
        f"roster targets protocol {declared!r}, this build is {PROTOCOL_VERSION!r}",
    )
    models = {entry.get("id"): entry for entry in roster.get("models", [])}
    _require(model_id in models, f"model {model_id!r} is not in the frozen roster")
    entry = models[model_id]

    main_table = bool(entry.get("main_table"))
    _require(
        main_table or not require_main_table,
        f"model {model_id!r} is not a main-table entry (reference baseline); pass "
        "require_main_table=False to run it as a declared experiment",
    )
    # A per-suite model is admitted per suite. Its model-level pin_status says
    # whether every suite it names has been verified, which is a statement about
    # the roster rather than about the run: a cohort executes one suite, and the
    # gate below refuses that suite unless its own artifact was checked.
    if entry.get("checkpoint_scope") != "per_suite":
        _require(
            entry.get("pin_status") == "pinned",
            f"model {model_id!r} is {entry.get('pin_status')!r}; a cohort requires a "
            "pinned model",
        )
    checkpoint, checkpoint_path, suite_revision = _resolve_suite_checkpoint(
        model_id, entry, suite
    )
    revision = suite_revision or entry.get("model_revision")
    _require(bool(revision), f"model {model_id!r} has no pinned model_revision")
    dependencies = _validated_dependencies(model_id, entry)
    adaptation = entry.get("adaptation")
    # How a model was trained is declared, not validated against an enum, and
    # it does not sort the cohort into a separate ranking. What is checked, below
    # and against the running server, is that one identity is used throughout: a
    # roster entry, a running server and a result table must agree on what ran.
    declared_regime = entry.get("training_regime", entry.get("adaptation_track"))
    _require(
        bool(str(declared_regime or "").strip()),
        f"model {model_id!r} declares no training_regime; the protocol asks every "
        "entry to say how it was trained, in its own words",
    )
    if track is not None:
        _require(
            declared_regime == track,
            f"model {model_id!r} declares training_regime {declared_regime!r}, but this "
            f"cohort was asked for {track!r}",
        )
    cameras = entry.get("policy_visible_cameras")
    return ModelEntry(
        model_id=model_id,
        suite=str(suite),
        display_name=str(entry.get("display_name", model_id)),
        category=str(entry.get("category", "")),
        family=str(entry.get("family", "")),
        main_table=main_table,
        adaptation=str(adaptation),
        training_regime=str(declared_regime),
        checkpoint_scope=str(entry.get("checkpoint_scope")),
        model_revision=str(revision),
        checkpoint_sha256=str(checkpoint),
        checkpoint_path=checkpoint_path,
        checkpoint_provenance=str(entry.get("checkpoint_provenance", "")),
        upstream_code_revision=entry.get("upstream_code_revision"),
        dependency_revisions=dependencies,
        policy_visible_cameras=tuple(cameras) if cameras else None,
        # ``predicted_action_chunk`` is the declaration's name; ``native_action_chunk``
        # is accepted as an alias.
        native_action_chunk=entry.get("predicted_action_chunk", entry.get("native_action_chunk")),
        native_execution_horizon=entry.get("native_execution_horizon"),
        policy_determinism=entry.get("policy_determinism"),
    )


def _resolve_suite_checkpoint(
    model_id: str,
    entry: dict,
    suite: str,
) -> tuple[str, Optional[str], Optional[str]]:
    """Resolve the checkpoint that this (model, suite) pair actually runs.

    Checkpoints come in two shapes. A single file is identified by its SHA256. A
    checkpoint that is several shards plus auxiliary heads plus code loaded
    through trust_remote_code cannot be, so its identity is the digest of its
    whole snapshot inventory, and it is admitted only once those bytes have
    actually been recomputed on this machine.
    """
    scope = entry.get("checkpoint_scope")
    if scope == "all_suites":
        # One checkpoint for every suite. Its identity is a single file's
        # SHA256 or, for a multi-file snapshot, the digest of the verified
        # inventory -- the same two shapes the per-suite branch accepts.
        if entry.get("inventory_sha256") or entry.get("inventory_manifest"):
            _require(
                entry.get("artifact_verified") is True,
                f"model {model_id!r} pins an inventory that has not been verified on "
                "this machine; a cohort runs bytes that were recomputed, not a "
                "revision that was merely resolved",
            )
            inventory = entry.get("inventory_sha256")
            _require(
                bool(inventory),
                f"model {model_id!r} claims verification but declares no inventory_sha256",
            )
            return str(inventory), entry.get("checkpoint_path"), None
        checkpoint = entry.get("checkpoint_sha256")
        _require(
            bool(checkpoint),
            f"model {model_id!r} has no pinned checkpoint_sha256 or inventory_sha256",
        )
        return str(checkpoint), entry.get("checkpoint_path"), None
    if scope == "per_suite":
        suites = entry.get("suite_checkpoints") or {}
        _require(
            suite in suites,
            f"model {model_id!r} publishes per-suite checkpoints and has none pinned "
            f"for suite {suite!r} (pinned: {sorted(suites)})",
        )
        pinned = suites[suite]
        revision = pinned.get("revision")
        _require(
            pinned.get("artifact_verified") is not False,
            f"the {suite!r} checkpoint of {model_id!r} has resolved provenance but "
            "has not been verified on this machine: knowing where an artifact lives "
            "is not the same as having recomputed its bytes",
        )
        if pinned.get("inventory_sha256") or pinned.get("inventory_manifest"):
            _require(
                pinned.get("artifact_verified") is True,
                f"the {suite!r} checkpoint of {model_id!r} has not been verified on "
                "this machine; a cohort runs bytes that were recomputed, not a "
                "revision that was merely resolved",
            )
            inventory = pinned.get("inventory_sha256")
            _require(
                bool(inventory),
                f"the {suite!r} checkpoint of {model_id!r} claims verification but "
                "declares no inventory_sha256",
            )
            return str(inventory), pinned.get("inventory_manifest"), revision
        _require(
            pinned.get("pin_status") == "pinned",
            f"the {suite!r} checkpoint of {model_id!r} is "
            f"{pinned.get('pin_status')!r}; a cohort requires a locally verified pin",
        )
        checkpoint = pinned.get("sha256")
        _require(
            bool(checkpoint),
            f"the {suite!r} checkpoint of {model_id!r} has no sha256",
        )
        return str(checkpoint), pinned.get("path"), revision
    raise CohortError(
        f"model {model_id!r} declares checkpoint_scope {scope!r}; a cohort needs "
        "'all_suites' or 'per_suite' so a cell can bind the weights it actually ran"
    )


def _validated_dependencies(model_id: str, entry: dict) -> tuple[tuple[str, str], ...]:
    """Require every declared upstream dependency to be pinned as well."""
    dependencies: list[tuple[str, str]] = []
    base = entry.get("base_model")
    if base:
        _require(
            base.get("pin_status") == "pinned",
            f"model {model_id!r} depends on {base.get('repo')!r}, which is "
            f"{base.get('pin_status')!r}; the measurement identity is incomplete without it",
        )
        _require(
            bool(base.get("revision")),
            f"the base model of {model_id!r} has no pinned revision",
        )
        dependencies.append((str(base.get("repo")), str(base.get("revision"))))
    return tuple(dependencies)


def require_clean_worktree(benchmark_root: PathLike) -> str:
    """Return the HEAD commit, refusing to start from a modified worktree."""
    from .runtime import _git_output  # local import keeps the module import cheap

    root = Path(benchmark_root)
    commit = _git_output(root, "rev-parse", "HEAD")
    _require(bool(commit), f"{root} is not a git checkout; a cohort needs a commit identity")
    _require(
        not _git_output(root, "status", "--porcelain"),
        "the benchmark worktree is dirty; a cohort run must run from a clean tree "
        "so its runtime certificate is unambiguous",
    )
    return str(commit)


def audit_cohort_seeds(
    benchmark: str,
    units: Sequence[tuple[str, int]],
    *,
    protocol_version: str = PROTOCOL_VERSION,
    target_bits: int = 32,
) -> SeedProjectionAudit:
    """Refuse a cohort whose protocol seeds collide in the backend seed width."""
    _require(bool(units), "a cohort needs at least one (task, episode_index) unit")
    seeds = [
        episode_seed(benchmark, task, int(episode_index), protocol_version)
        for task, episode_index in units
    ]
    audit = audit_seed_projection(seeds, target_bits)
    if audit.has_collisions:
        collisions = ", ".join(
            f"{c.projected_seed}<-{c.protocol_seeds}" for c in audit.collisions[:5]
        )
        raise CohortError(
            f"{len(audit.collisions)} protocol seeds collide at {target_bits} bits "
            f"({collisions}); distinct units would silently share a simulator scene"
        )
    return audit


def platform_preflight(
    *,
    benchmark: str,
    benchmark_root: PathLike,
    registry_path: PathLike,
    dataset_id: str,
    units: Sequence[tuple[str, int]],
    stage: str,
    platform_camera_ids: Sequence[str],
    target_bits: int = 32,
) -> PlatformPreflight:
    """Model-agnostic checks: protocol, data pin, worktree, units, seeds, cameras."""
    _require(
        stage not in {"stage_0"},
        "stage_0 is the mock and dry-run stage; a real cohort runs at stage_1 or later",
    )
    cameras = tuple(str(camera) for camera in platform_camera_ids)
    _require(bool(cameras), "the platform must declare its camera inventory")
    _require(len(set(cameras)) == len(cameras), "platform_camera_ids contains duplicates")
    try:
        dataset = load_and_validate_dataset(registry_path, dataset_id, stage=stage)
    except RegistryError as exc:
        raise CohortError(f"dataset {dataset_id!r} is not cohort-ready: {exc}") from exc
    _require(
        not dataset.mock,
        "a cohort run may not run against a mock dataset authorization",
    )
    commit = require_clean_worktree(benchmark_root)
    audit = audit_cohort_seeds(
        benchmark, units, protocol_version=PROTOCOL_VERSION, target_bits=target_bits
    )
    return PlatformPreflight(
        protocol_version=PROTOCOL_VERSION,
        dataset=dataset,
        benchmark_commit=commit,
        seed_projection=audit,
        unit_count=len(units),
        platform_camera_ids=cameras,
    )


def reference_model_preflight(
    roster_path: PathLike,
    model_id: str,
    *,
    suite: str,
    track: Optional[str] = None,
    require_main_table: bool = True,
) -> ModelEntry:
    """Extra checks that apply only to this project's own reference baselines."""
    return load_model_entry(
        roster_path, model_id, suite=suite, track=track, require_main_table=require_main_table
    )


def preflight(
    *,
    benchmark: str,
    benchmark_root: PathLike,
    registry_path: PathLike,
    dataset_id: str,
    units: Sequence[tuple[str, int]],
    stage: str,
    platform_camera_ids: Sequence[str],
    roster_path: Optional[PathLike] = None,
    model_id: Optional[str] = None,
    suite: Optional[str] = None,
    track: Optional[str] = None,
    target_bits: int = 32,
    require_main_table: bool = True,
) -> CohortPreflight:
    """Run the platform checks, plus the roster checks when a baseline is named.

    A third-party policy passes ``roster_path``/``model_id`` as ``None`` and is
    validated later against the same runtime contract; being absent from the
    reference roster is not a reason to refuse an evaluation.
    """
    # Argument consistency first: a caller mistake should not be reported as an
    # environment problem, and it costs nothing to check.
    names_a_baseline = model_id is not None or roster_path is not None
    if names_a_baseline:
        _require(
            model_id is not None and roster_path is not None and suite is not None,
            "a reference baseline needs roster_path, model_id and suite together",
        )
    platform = platform_preflight(
        benchmark=benchmark,
        benchmark_root=benchmark_root,
        registry_path=registry_path,
        dataset_id=dataset_id,
        units=units,
        stage=stage,
        platform_camera_ids=platform_camera_ids,
        target_bits=target_bits,
    )
    model = (
        reference_model_preflight(
            roster_path, model_id, suite=suite, track=track, require_main_table=require_main_table
        )
        if names_a_baseline
        else None
    )
    return CohortPreflight(platform=platform, model=model)


@dataclass(frozen=True)
class AdmissionReport:
    """Outcome of the healthy admission gate over a cohort."""

    admitted_units: tuple[tuple[str, int], ...]
    rejected_units: tuple[tuple[str, int], ...]
    clean_success_rate: float
    cohort_target: float
    meets_cohort_target: bool


def admission_gate(
    unit_solvability: Iterable[tuple[tuple[str, int], bool]],
    *,
    cohort_target: float = 0.80,
) -> AdmissionReport:
    """Pair the units with their healthy outcome and report the cohort rate.

    ``unit_solvability`` pairs each ``(task, episode_index)`` with whether its
    canonical healthy rollout succeeded (``clean_solvable`` from
    :func:`mail_bench.experiment.aggregate_healthy_references`; under the
    official profile's single healthy replicate that is the rollout's own
    success flag).  A unit without a healthy success stays in the paired
    matrix but has no fault cells.  The cohort-level rate is reported against
    ``cohort_target`` as a diagnostic and never gates a run: a run proceeds
    whenever any unit has a healthy success, and a changed target or a
    disabled check is recorded in the certificate as a deviation.  A model with
    no healthy success anywhere is diagnosed as a compatibility problem instead
    of being published as poor robustness.
    """
    admitted: list[tuple[str, int]] = []
    rejected: list[tuple[str, int]] = []
    for unit, solvable in unit_solvability:
        (admitted if solvable else rejected).append((str(unit[0]), int(unit[1])))
    total = len(admitted) + len(rejected)
    _require(total > 0, "the admission gate needs at least one evaluated unit")
    rate = len(admitted) / total
    return AdmissionReport(
        admitted_units=tuple(admitted),
        rejected_units=tuple(rejected),
        clean_success_rate=rate,
        cohort_target=float(cohort_target),
        meets_cohort_target=rate >= float(cohort_target),
    )


def cohort_units(tasks: Sequence[str], episode_indices: Sequence[int]) -> tuple[tuple[str, int], ...]:
    """Expand a frozen task list and episode list into ordered cohort units."""
    _require(bool(tasks), "a cohort needs at least one task")
    _require(bool(episode_indices), "a cohort needs at least one episode index")
    _require(len(set(tasks)) == len(tasks), "the cohort task list contains duplicates")
    _require(
        len(set(episode_indices)) == len(episode_indices),
        "the cohort episode list contains duplicates",
    )
    return tuple(
        (str(task), int(episode_index))
        for task in tasks
        for episode_index in episode_indices
    )


@dataclass(frozen=True)
class ExecutionProfile:
    """How much of a predicted chunk is executed before the policy is re-queried.

    ``native`` uses the model's own validated evaluation setting; a matched
    profile fixes the same interval across models so that "more robust" cannot be
    confused with "looked at the cameras more often".  The value reaches
    ``RunnerConfig.max_actions_per_query`` and therefore ``runner_config_hash``,
    so the two never share a cell identity.
    """

    name: str
    max_actions_per_query: Optional[int]

    @classmethod
    def native(cls, model: ModelEntry) -> "ExecutionProfile":
        _require(
            model.native_execution_horizon is not None,
            f"model {model.model_id!r} declares no native_execution_horizon; the roster "
            "must record the interval its published evaluation used",
        )
        return cls.matched(
            int(model.native_execution_horizon),
            predicted_action_chunk=model.native_action_chunk,
        ).renamed("native")

    @classmethod
    def matched(
        cls,
        steps: int,
        *,
        predicted_action_chunk: Optional[int] = None,
    ) -> "ExecutionProfile":
        _require(steps >= 1, "a matched replanning interval must be >= 1")
        # A profile that asks for more actions than one query produces would
        # silently execute the shorter chunk, so the name would misdescribe the
        # exposure it created. Fail closed whenever the chunk length is known.
        if predicted_action_chunk is not None:
            _require(
                int(steps) <= int(predicted_action_chunk),
                f"a replanning interval of {steps} exceeds the {predicted_action_chunk} "
                "actions this model predicts per query; the executed interval would be "
                "the shorter one and the profile name would not describe it",
            )
        return cls(f"matched_replan_{int(steps)}", int(steps))

    def renamed(self, name: str) -> "ExecutionProfile":
        return ExecutionProfile(name, self.max_actions_per_query)

    @classmethod
    def parse(cls, spec: str, model: ModelEntry) -> "ExecutionProfile":
        text = str(spec).strip().lower()
        if text == "native":
            return cls.native(model)
        prefix = "matched_replan_"
        if text.startswith(prefix) and text[len(prefix):].isdigit():
            return cls.matched(
                int(text[len(prefix):]), predicted_action_chunk=model.native_action_chunk
            )
        raise CohortError(
            f"unknown execution profile {spec!r}; expected 'native' or 'matched_replan_<n>'"
        )


@dataclass(frozen=True)
class FaultCellSpec:
    """One fault cell of a unit, before the healthy reference resolves its onset."""

    task: str
    episode_index: int
    faulted_cameras: tuple[str, ...]
    onset_fraction: float
    #: Provenance about the policy's declared camera scope, never a ranking
    #: rule: ``no_op_for_declared_scope`` when the faulted set is disjoint from
    #: what the policy declares it reads, ``policy_scope_unresolved`` when no
    #: scope was declared, ``None`` otherwise. Every cell is ranked regardless.
    scope_note: Optional[str] = None


def fault_cell_specs(
    units: Sequence[tuple[str, int]],
    *,
    platform_camera_ids: Sequence[str],
    onset_fractions: Sequence[float],
    policy_visible_cameras: Optional[Sequence[str]] = None,
    semantic_platform: Optional[str] = None,
) -> tuple[FaultCellSpec, ...]:
    """Expand units into the frozen fault grid, deduplicated and ordered.

    The grid is the ranking's three functional roles -- wrist, third-person,
    both -- on the platform named by ``semantic_platform``, rather than an
    enumeration of physical camera subsets: enumerating subsets would make a
    three-camera platform answer seven questions and a two-camera platform
    three, and no condition would mean the same thing on two platforms.

    The grid is a property of the *platform*, never of the policy: every
    submission faces the same camera subsets, the same onsets and the same
    schedule, whatever it reads internally, and every cell is ranked for every
    policy. ``policy_visible_cameras`` changes neither which cells exist nor
    which count; it is recorded on each cell as provenance:

    * a cell that only removes cameras this policy never reads is a no-op for
      it, and the score says so through the cell itself -- a policy that keeps
      working without a view it never used is reporting how little vision it
      used, which is what the ranking measures for total visual loss too. The
      note lets a reader see that beside the number. The scope is the
      *declared static upper bound*: a router that may select either camera
      declares both, and no scope is ever inferred from an observed routing
      trace;
    * an *unresolved* scope (``policy_visible_cameras=None``) is noted as such.
      An official run resolves the scope from the roster or the server
      identity first; the note makes a cohort planned without one visible.
    """
    platform = [str(camera) for camera in platform_camera_ids]
    _require(len(platform) >= 1, "a cohort needs at least one platform camera")
    _require(len(set(platform)) == len(platform), "platform_camera_ids contains duplicates")
    # ``None`` means UNRESOLVED, never "all cameras": the note on every cell says
    # the cohort was planned before the scope was resolved from the roster or
    # the policy server's identity.
    visible = None
    if policy_visible_cameras is not None:
        visible = {str(camera) for camera in policy_visible_cameras}
        _require(bool(visible), "a policy must read at least one camera")
        unknown = visible - set(platform)
        _require(
            not unknown,
            f"policy declares cameras the platform does not provide: {sorted(unknown)}",
        )
    _require(bool(onset_fractions), "a cohort needs at least one onset fraction")
    for fraction in onset_fractions:
        _require(0.0 <= float(fraction) < 1.0, f"onset fraction {fraction} must be in [0, 1)")

    from .semantic_states import MISSING_STATES, state_cameras, validate_platform

    _require(semantic_platform is not None, "a cohort needs the semantic platform id")
    validate_platform(semantic_platform, platform)
    # The three missing states of the ranking. Total visual loss is ranked:
    # a policy that keeps working with no camera at all is telling us how much
    # it was using vision, and that belongs in the score.
    faulted_sets = [state_cameras(semantic_platform, state) for state in MISSING_STATES]
    specs: list[FaultCellSpec] = []
    for task, episode_index in units:
        for faulted in faulted_sets:
            if visible is None:
                note = "policy_scope_unresolved"
            elif not (set(faulted) & visible):
                # The declared scope is a static upper bound, so this really is a
                # no-op: no timestep of any rollout could have used these cameras.
                note = "no_op_for_declared_scope"
            else:
                note = None
            for fraction in onset_fractions:
                specs.append(
                    FaultCellSpec(
                        task=str(task),
                        episode_index=int(episode_index),
                        faulted_cameras=faulted,
                        onset_fraction=float(fraction),
                        scope_note=note,
                    )
                )
    return tuple(specs)


def shard_units(
    units: Sequence[tuple[str, int]],
    *,
    workers: int,
    worker_index: int,
) -> tuple[tuple[str, int], ...]:
    """Deterministic, non-overlapping static shard of the unit list.

    Sharding is by ``(task, episode_index)`` so that every cell of a unit is
    produced by one worker; the assignment is a scheduling decision and never
    enters a cell's identity.
    """
    _require(workers >= 1, "workers must be >= 1")
    _require(0 <= worker_index < workers, "worker_index must satisfy 0 <= index < workers")
    ordered = tuple(units)
    _require(len(set(ordered)) == len(ordered), "the unit list contains duplicates")
    return tuple(unit for position, unit in enumerate(ordered) if position % workers == worker_index)


def resolve_runtime_contract(
    identity: Mapping[str, Any],
    *,
    platform_camera_ids: Sequence[str],
    suite: Optional[str] = None,
    entry: Optional[ModelEntry] = None,
    platform_action_dimension: Optional[int] = None,
) -> "PolicyDeclaration":
    """Second preflight layer: resolve the contract from the live policy server.

    The static preflight proves the protocol, the data pin and the checkpoint pin
    without starting anything.  This layer runs once the server is up and turns
    what it announces into the declaration the cohort will be executed under.
    Everything a fair comparison needs -- camera scope, predicted chunk, executed
    horizon, action dimension, availability consumption -- must be *resolved*
    here; an official rollout may not start on unknown values.

    ``entry`` is optional on purpose: the reference roster is this project's own
    baseline registry, not a membership requirement. A third party runs their own
    server and is validated against the same contract without appearing in any
    config file. When an entry *is* given, the server and the roster must agree,
    or the cohort would be attributed to weights or semantics it did not run.
    """
    from .policy_api.contract import ContractError, PolicyDeclaration, validate_declaration

    try:
        declaration = validate_declaration(identity, environment_cameras=platform_camera_ids)
    except ContractError as exc:
        raise CohortError(f"the policy server's declaration is not admissible: {exc}") from exc

    if platform_action_dimension is not None:
        # Found at admission, not at the first step of the first healthy cell.
        _require(
            int(declaration.action_dimension) == int(platform_action_dimension),
            f"the policy server declares action_dimension={declaration.action_dimension}, "
            f"the platform executes {platform_action_dimension}-dimensional actions",
        )
    if suite is not None and identity.get("suite") is not None:
        _require(
            str(identity["suite"]) == str(suite),
            f"the policy server serves suite {identity['suite']!r}, but this cohort runs {suite!r}",
        )
    if entry is not None:
        _require(
            declaration.checkpoint_sha256 == entry.checkpoint_sha256,
            f"the server runs checkpoint {declaration.checkpoint_sha256[:12]}..., but the "
            f"frozen entry pins {entry.checkpoint_sha256[:12]}...",
        )
        _require(
            declaration.training_regime == entry.training_regime,
            f"the server declares training_regime {declaration.training_regime!r} but the "
            f"frozen entry says {entry.training_regime!r}; the benchmark does not judge "
            "either answer, but a result must be published under the regime that ran",
        )
        declared = {
            "policy_visible_cameras": (
                tuple(entry.policy_visible_cameras) if entry.policy_visible_cameras else None
            ),
            "predicted_action_chunk": entry.native_action_chunk,
            "native_execution_horizon": entry.native_execution_horizon,
        }
        served = {
            "policy_visible_cameras": declaration.policy_visible_cameras,
            "predicted_action_chunk": declaration.predicted_action_chunk,
            "native_execution_horizon": declaration.native_execution_horizon,
        }
        for field, expected in declared.items():
            # A roster that leaves a field unresolved is filled in by the server;
            # a roster that declares one must agree with it.
            if expected is not None and served[field] != expected:
                raise CohortError(
                    f"the roster declares {field}={expected!r} but the server reports "
                    f"{served[field]!r}; the two must agree before any rollout"
                )
    assert isinstance(declaration, PolicyDeclaration)
    return declaration


def execution_profile_for(spec: str, declaration: "PolicyDeclaration") -> ExecutionProfile:
    """Build the execution profile against the *resolved* contract."""
    text = str(spec).strip().lower()
    if text == "native":
        return ExecutionProfile.matched(
            declaration.native_execution_horizon,
            predicted_action_chunk=declaration.predicted_action_chunk,
        ).renamed("native")
    prefix = "matched_replan_"
    if text.startswith(prefix) and text[len(prefix):].isdigit():
        return ExecutionProfile.matched(
            int(text[len(prefix):]),
            predicted_action_chunk=declaration.predicted_action_chunk,
        )
    raise CohortError(
        f"unknown execution profile {spec!r}; expected 'native' or 'matched_replan_<n>'"
    )
