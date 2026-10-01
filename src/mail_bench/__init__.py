"""MAIL-Bench: paired evaluation of manipulation under visual availability loss.

The simulator-free core, importable without a GPU:

* :mod:`mail_bench.seeds`      -- derived, separated RNG streams;
* :mod:`mail_bench.operators`  -- availability fault operators and injector;
* :mod:`mail_bench.onset`      -- onset / duration resolution;
* :mod:`mail_bench.manifest`   -- manifests, result records, canonical JSON;
* :mod:`mail_bench.registry`   -- dataset validation;
* :mod:`mail_bench.cohort`     -- fail-closed preflight for a cohort run;
* :mod:`mail_bench.driver`     -- healthy phase, onset freezing, fault phase;
* :mod:`mail_bench.experiment` -- paired cells and healthy references;
* :mod:`mail_bench.runner`     -- query-clock rollout and action-chunk audit;
* :mod:`mail_bench.io`         -- atomic, idempotent result persistence;
* :mod:`mail_bench.scoring`, :mod:`mail_bench.aggregate` -- the MAIL score;
* :mod:`mail_bench.interfaces` -- canonical observation and adapter interfaces;
* :mod:`mail_bench.policy_api` -- the contract any submitted policy declares;
* :mod:`mail_bench.platforms`  -- simulator adapters.

The execution kernel never knows which model is behind a policy: a policy is
reached through :class:`~mail_bench.interfaces.PolicyAdapter` and its
declaration, and the reference servers live in ``scripts/``.
"""

from .interfaces import (
    CanonicalObservation,
    EnvironmentAdapter,
    EnvironmentStep,
    EpisodeContext,
    PolicyAdapter,
    PolicyOutput,
)
from .policy_api import (
    TRAINING_REGIME_DISCLOSURE_FIELDS,
    ContractError,
    PolicyDeclaration,
    cameras_outside_declared_scope,
    validate_declaration,
)
from .cohort import (
    AdmissionReport,
    ExecutionProfile,
    FaultCellSpec,
    CohortError,
    CohortPreflight,
    PlatformPreflight,
    ModelEntry,
    admission_gate,
    audit_cohort_seeds,
    cohort_units,
    execution_profile_for,
    fault_cell_specs,
    load_model_entry,
    resolve_runtime_contract,
    preflight,
    require_clean_worktree,
    shard_units,
)
from .driver import (
    DEVIATIONS,
    CohortReport,
    ExecutedFaultCell,
    CohortRunner,
    CohortSpecification,
    UnitOutcome,
    OfficialProfile,
    cohort_admission_status,
    profile_deviations,
    require_resolved_scope,
)
from .io import (
    AtomicResultWriter,
    ResultConflictError,
    semantic_run_key,
    semantic_run_key_fields,
)
from .experiment import (
    CellExecution,
    HealthyReferenceSummary,
    HealthyReferenceRunner,
    HealthyReplicate,
    HealthyReplicateOutcome,
    HealthyReplicateRequest,
    MethodArmExecution,
    PairedCellRunner,
    aggregate_healthy_references,
    freeze_fault_manifest,
    healthy_replicate_manifests,
    resolve_reference_onset,
)
from .manifest import (
    HEALTHY_MODE,
    PROTOCOL_VERSION,
    FaultManifest,
    ManifestError,
    ResultError,
    ResultRecord,
    canonical_json,
    semantic_sha256,
    validate_manifest,
    validate_result,
)
from .onset import resolve_duration, resolve_onset, round_half_up
from .operators import (
    CANONICAL_MODES,
    MODE_ALIASES,
    CausalViolation,
    FaultInjector,
    FaultSchedule,
    Poison,
    TraceRecord,
    burst_hash,
    burst_schedule,
    canonical_mode,
    poison_test,
    zero_like,
)
from .seeds import (
    Streams,
    derive_seed,
    episode_seed,
    fault_seed,
    healthy_replicate_policy_seed,
    healthy_replicate_seeds,
    policy_seed,
)
from .registry import (
    DatasetValidation,
    RegistryError,
    load_and_validate_dataset,
    validate_dataset,
)
from .runner import (
    ActionChunkManager,
    ChunkInvalidation,
    ExecutionBinding,
    RolloutAudit,
    RolloutOutcome,
    RolloutRunner,
    RunnerConfig,
    schedule_from_manifest,
    validate_execution_request,
)

__version__ = "1.0.0"

_AGGREGATE_EXPORTS = (
    "AggregationError", "MeasurementKey", "load_cells", "mail_bench_report",
)
_INVENTORY_EXPORTS = (
    "DatasetInventory", "InventoryFile", "build_inventory", "git_tracked_paths",
    "write_inventory",
)


def __getattr__(name: str):
    # ``aggregate`` and ``inventory`` are imported lazily: the former is heavy
    # and the latter is also a ``python -m`` entry point.
    if name in _AGGREGATE_EXPORTS:
        import importlib

        return getattr(importlib.import_module(".aggregate", __name__), name)
    if name in _INVENTORY_EXPORTS:
        from . import inventory as _inventory

        return getattr(_inventory, name)
    raise AttributeError(f"module 'mail_bench' has no attribute {name!r}")


__all__ = [
    "__version__",
    # cohort preflight
    "AdmissionReport", "CohortError", "CohortPreflight", "ModelEntry", "admission_gate",
    "audit_cohort_seeds", "cohort_units", "load_model_entry", "preflight",
    "require_clean_worktree", "PlatformPreflight", "ExecutionProfile", "FaultCellSpec",
    "fault_cell_specs", "execution_profile_for", "resolve_runtime_contract", "shard_units",
    # cohort orchestration
    "CohortReport", "CohortRunner", "CohortSpecification", "ExecutedFaultCell",
    "UnitOutcome", "cohort_admission_status", "require_resolved_scope", "OfficialProfile",
    "DEVIATIONS", "profile_deviations",
    # seeds
    "Streams", "derive_seed", "episode_seed", "policy_seed", "fault_seed",
    "healthy_replicate_policy_seed", "healthy_replicate_seeds",
    # experiment
    "CellExecution", "HealthyReferenceSummary", "HealthyReferenceRunner", "HealthyReplicate",
    "HealthyReplicateOutcome", "HealthyReplicateRequest", "MethodArmExecution",
    "PairedCellRunner", "aggregate_healthy_references", "freeze_fault_manifest",
    "healthy_replicate_manifests", "resolve_reference_onset",
    # operators
    "CANONICAL_MODES", "MODE_ALIASES", "CausalViolation", "FaultInjector", "FaultSchedule",
    "Poison", "TraceRecord", "burst_hash", "burst_schedule", "canonical_mode", "poison_test",
    "zero_like",
    # onset
    "resolve_onset", "resolve_duration", "round_half_up",
    # manifest
    "HEALTHY_MODE", "PROTOCOL_VERSION", "FaultManifest", "ManifestError", "ResultError",
    "ResultRecord", "canonical_json", "semantic_sha256", "validate_manifest", "validate_result",
    # registry
    "DatasetValidation", "RegistryError", "load_and_validate_dataset", "validate_dataset",
    # adapters / execution / io
    "CanonicalObservation", "EnvironmentAdapter", "EnvironmentStep", "EpisodeContext",
    "PolicyAdapter", "PolicyOutput",
    # policy contract: the only thing a submission must satisfy
    "TRAINING_REGIME_DISCLOSURE_FIELDS", "ContractError", "PolicyDeclaration", "cameras_outside_declared_scope",
    "validate_declaration",
    "ActionChunkManager", "ChunkInvalidation", "ExecutionBinding", "RolloutAudit",
    "RolloutOutcome", "RolloutRunner", "RunnerConfig", "AtomicResultWriter",
    "ResultConflictError", "semantic_run_key", "semantic_run_key_fields",
    "schedule_from_manifest", "validate_execution_request",
    # lazy
    "DatasetInventory", "InventoryFile", "build_inventory", "git_tracked_paths",
    "write_inventory", "AggregationError", "MeasurementKey", "load_cells", "mail_bench_report",
]
