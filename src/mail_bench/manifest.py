"""Fault-cell manifests, result records, canonical JSON and validation.

A :class:`FaultManifest` freezes everything that defines one fault cell (the
field list is the protocol's fault manifest; fields
that do not apply to a perturbation family are ``None``).  A
:class:`ResultRecord` holds the minimum per-run result fields.

Both are serialised through :func:`canonical_json` (sorted keys, compact
separators, ``repr``-stable floats) and identified by
:func:`semantic_sha256`, so the hash is independent of key order and of
whitespace.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from dataclasses import dataclass, fields
from typing import Any, Optional

from .operators import CANONICAL_MODES, MODE_ALIASES

# Manifests must use canonical operator ids; aliases (missing/stale/burst) are
# rejected with a hint naming the canonical id.
AVAILABILITY_MODES = set(CANONICAL_MODES)
KNOWN_NON_TEMPORAL_MODES = AVAILABILITY_MODES | {"healthy"}

REQUIRED_MANIFEST_FIELDS = (
    "dataset_version",
    "split",
    "scene_or_task",
    "episode_or_sequence",
    "camera_ids",
    "faulted_camera_ids",
    "perturbation_family",
    "fault_mode",
    "onset_fraction",
    "environment_seed",
    "policy_seed",
    "fault_seed",
    "official_horizon",
    "adapter_version",
    "protocol_version",
    "onset_basis",
    "healthy_reference_success",
    "onset_reference_hash",
    "onset_fallback",
    "subset_ranking_eligible",
)

ONSET_BASES = {"healthy_reference", "not_applicable"}
# A healthy availability cell has no faulted camera and therefore no onset provenance:
# it *defines* the reference that fault cells resolve their onset against.
HEALTHY_MODE = "healthy"
PROTOCOL_VERSION = "mail_bench_perturbation_v1"
# ``protocol_version`` and ``result_schema_version`` are deliberately separate.
# Bump the protocol only for changes that can alter a scientific result (task
# inclusion, onset, duration, validity, ranking, fault semantics, action/query
# semantics, the healthy definition, pairing, recovery, model input semantics):
# it enters the manifest hash and therefore invalidates every computed cell.
# Bump only the result schema for observability-shaped additions (runtime
# metadata, diagnostic hashes, hardware facts, optional audit fields) that leave
# the rollout trajectory and eligibility untouched; older results stay readable
# and resumable across such a bump.  A released protocol version is never edited
# in place -- a semantic change opens a new version and earlier results are
# neither overwritten nor mixed in.
RESULT_SCHEMA_VERSION = 3


# ---------------------------------------------------------------------------
# Canonical JSON
# ---------------------------------------------------------------------------

def _to_plain(obj: Any) -> Any:
    """Recursively convert ``obj`` to plain JSON types with stable ordering."""
    if obj is None or isinstance(obj, (str, bool, int)):
        return obj
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            raise ValueError("NaN/Inf are not allowed in canonical JSON")
        return obj
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: _to_plain(getattr(obj, f.name)) for f in fields(obj)}
    if isinstance(obj, dict):
        return {str(k): _to_plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_plain(v) for v in obj]
    if isinstance(obj, (set, frozenset)):
        return sorted((_to_plain(v) for v in obj), key=lambda v: json.dumps(v, sort_keys=True))
    if isinstance(obj, (bytes, bytearray)):
        return obj.hex()
    try:  # numpy scalars / arrays without importing numpy eagerly
        import numpy as np

        if isinstance(obj, np.ndarray):
            return _to_plain(obj.tolist())
        if isinstance(obj, np.generic):
            return _to_plain(obj.item())
    except ImportError:  # pragma: no cover
        pass
    raise TypeError(f"object of type {type(obj).__name__} is not canonical-JSON serialisable")


def canonical_json(obj: Any) -> str:
    """Canonical JSON: sorted keys, no whitespace, ``repr``-stable floats.

    Floats are emitted with Python's shortest round-trip ``repr`` (so
    ``1.0`` stays ``1.0`` and ``0.1`` stays ``0.1``).  Sets are sorted,
    dataclasses are expanded, NaN/Inf are rejected.
    """
    return json.dumps(
        _to_plain(obj), sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    )


def semantic_sha256(obj: Any) -> str:
    """SHA256 hex digest of :func:`canonical_json`."""
    return hashlib.sha256(canonical_json(obj).encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------

@dataclass
class FaultManifest:
    """Frozen description of one fault cell.

    Field names follow the protocol; the extra fields ``cell_id``,
    ``protocol_version``, ``onset_step``, ``end_step``, ``stale_k``,
    ``burst_frames``, ``available_rate``, ``burst_sequence_hash`` and
    ``recovery_mode`` carry the resolved availability parameters;
    ``recovery_fraction`` is Visual Recovery's second end (None for every
    other cell, and then excluded from the semantic hash).
    """

    dataset_version: Optional[str] = None
    split: Optional[str] = None
    scene_or_task: Optional[str] = None
    episode_or_sequence: Optional[Any] = None
    camera_ids: Optional[list[str]] = None
    faulted_camera_ids: Optional[list[str]] = None
    perturbation_family: Optional[str] = None
    fault_mode: Optional[str] = None
    onset_fraction: Optional[float] = None
    duration_fraction: Optional[float] = None
    severity: Optional[str] = None
    distance_scale: Optional[float] = None
    translation_norm: Optional[float] = None
    azimuth_delta_deg: Optional[float] = None
    elevation_delta_deg: Optional[float] = None
    rotation_geodesic_deg: Optional[float] = None
    fov_delta_deg: Optional[float] = None
    corruption_operator: Optional[str] = None
    corruption_parameters: Optional[dict[str, Any]] = None
    environment_seed: Optional[int] = None
    policy_seed: Optional[int] = None
    fault_seed: Optional[int] = None
    official_horizon: Optional[int] = None
    adapter_version: Optional[str] = None
    nominal_camera_fps: Optional[float] = None
    nominal_control_hz: Optional[float] = None
    capture_schedule_hash: Optional[str] = None
    packet_trace_hash: Optional[str] = None
    timestamp_visibility_mode: Optional[str] = None
    lag_control_steps: Optional[int] = None
    lag_ms: Optional[float] = None
    fps_ratio: Optional[float] = None
    jitter_profile: Optional[str] = None
    clock_mapping_version: Optional[str] = None
    reorder_profile: Optional[str] = None
    # resolved / bookkeeping extras
    cell_id: Optional[str] = None
    protocol_version: Optional[str] = None
    onset_step: Optional[int] = None
    end_step: Optional[int] = None
    stale_k: Optional[int] = None
    burst_frames: Optional[int] = None
    available_rate: Optional[float] = None
    burst_sequence_hash: Optional[str] = None
    recovery_mode: Optional[str] = None
    onset_basis: Optional[str] = None
    healthy_reference_success: Optional[bool] = None
    onset_reference_hash: Optional[str] = None
    onset_fallback: Optional[bool] = None
    subset_ranking_eligible: Optional[bool] = None
    recovery_fraction: Optional[float] = None

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    def semantic_hash(self) -> str:
        """Semantic SHA256 of the manifest (excluding ``cell_id`` itself)."""
        d = self.to_dict()
        d.pop("cell_id", None)
        if d.get("recovery_fraction") is None:
            d.pop("recovery_fraction", None)
        return semantic_sha256(d)

    def derived_cell_id(self) -> str:
        """Human-readable cell id built from the defining fields."""
        cams = "+".join(sorted(self.faulted_camera_ids or [])) or "none"
        if self.recovery_fraction is not None:
            end = f"rec={self.recovery_fraction:g}"
        else:
            dur = "end" if self.duration_fraction in (None, 1.0) else f"{self.duration_fraction:g}"
            end = f"dur={dur}"
        return (
            f"{self.scene_or_task}|ep{self.episode_or_sequence}|{self.fault_mode}|"
            f"miss={cams}|onset={self.onset_fraction:g}|{end}"
        )


class ManifestError(ValueError):
    """Raised when a manifest is invalid."""


def validate_manifest(m: FaultManifest) -> FaultManifest:
    """Validate a manifest, raising :class:`ManifestError` on problems.

    Checks: required fields present, ``fault_mode`` known, faulted cameras
    are a subset of ``camera_ids``, fractions in ``[0, 1]``, ``onset_step``
    (if given) consistent with ``official_horizon`` and ``end_step``,
    ``onset_basis`` one of :data:`ONSET_BASES` with ``onset_fallback`` False
    for a fault cell, a healthy cell carrying no onset and not ranked, and the
    ``stale_k`` parameters present and consistent for that mode.
    Assigns ``cell_id`` from :meth:`FaultManifest.derived_cell_id` when
    missing and returns the manifest.
    """
    missing = [f for f in REQUIRED_MANIFEST_FIELDS if getattr(m, f) is None]
    if missing:
        raise ManifestError(f"manifest missing required fields: {missing}")
    mode = str(m.fault_mode)
    family = str(m.perturbation_family)
    if mode not in KNOWN_NON_TEMPORAL_MODES:
        hint = ""
        if mode in MODE_ALIASES:
            hint = f" (alias; use canonical id {MODE_ALIASES[mode]!r})"
        raise ManifestError(f"unknown fault_mode {m.fault_mode!r}{hint}")
    cams = set(m.camera_ids)
    if len(cams) != len(m.camera_ids):
        raise ManifestError("camera_ids contains duplicates")
    bad = set(m.faulted_camera_ids) - cams
    if bad:
        raise ManifestError(f"faulted_camera_ids not in camera_ids: {sorted(bad)}")
    if not 0.0 <= float(m.onset_fraction) <= 1.0:
        raise ManifestError("onset_fraction must be in [0, 1]")
    if m.duration_fraction is not None and not 0.0 <= float(m.duration_fraction) <= 1.0:
        raise ManifestError("duration_fraction must be in [0, 1]")
    if int(m.official_horizon) < 1:
        raise ManifestError("official_horizon must be >= 1")
    if m.protocol_version != PROTOCOL_VERSION:
        raise ManifestError(
            f"protocol_version must be {PROTOCOL_VERSION!r}, got {m.protocol_version!r}"
        )
    if m.onset_basis not in ONSET_BASES:
        raise ManifestError(f"onset_basis must be one of {sorted(ONSET_BASES)}")
    if not isinstance(m.healthy_reference_success, bool):
        raise ManifestError("healthy_reference_success must be boolean")
    if not isinstance(m.onset_fallback, bool):
        raise ManifestError("onset_fallback must be boolean")
    if not isinstance(m.onset_reference_hash, str) or (
        len(m.onset_reference_hash) != 64
        or any(c not in "0123456789abcdefABCDEF" for c in m.onset_reference_hash)
    ):
        raise ManifestError("onset_reference_hash must be a 64-character SHA256 hex digest")
    if not isinstance(m.subset_ranking_eligible, bool):
        raise ManifestError("subset_ranking_eligible must be boolean")
    healthy_cell = family == "availability" and not m.faulted_camera_ids
    if healthy_cell != (mode == HEALTHY_MODE):
        raise ManifestError(
            "an availability cell uses fault_mode 'healthy' exactly when no camera is faulted"
        )
    if healthy_cell:
        # The healthy arm is a first-class availability cell so that it shares the
        # rollout kernel, the result schema and the paired pre-onset audit with the
        # fault arms.  It carries no onset provenance because it produces it.
        if m.onset_basis != "not_applicable":
            raise ManifestError("a healthy cell must use onset_basis 'not_applicable'")
        if m.onset_fallback or m.healthy_reference_success:
            raise ManifestError(
                "a healthy cell defines the reference and cannot claim onset provenance"
            )
        if m.onset_step is not None or m.end_step is not None:
            raise ManifestError("a healthy cell must not resolve an onset_step or end_step")
        if m.subset_ranking_eligible:
            raise ManifestError("a healthy cell is never a ranked camera subset")
    else:
        if m.onset_basis == "not_applicable":
            raise ManifestError("onset_basis 'not_applicable' is reserved for healthy cells")
        # A fault cell has one basis. A scene whose healthy rollout
        # failed has no task phase to place an onset in, so it produces no fault
        # cell at all rather than one anchored to the horizon.
        if m.onset_basis != "healthy_reference":
            raise ManifestError(
                "a fault cell resolves its onset from the evaluated policy's own "
                "successful healthy rollout; a scene without one is not evaluable"
            )
        if m.onset_fallback:
            raise ManifestError(
                "onset_fallback must be False: a fault cell never falls back to "
                "a horizon-derived onset; a healthy failure skips its fault cells"
            )
        if not m.healthy_reference_success:
            raise ManifestError(
                "a fault cell requires healthy_reference_success; without it the "
                "onset has nothing to be a fraction of"
            )
        # All Vision Missing is ranked: three of the ten conditions are total
        # visual loss, and a policy that keeps working without any camera is
        # telling us how much it was using vision, which belongs in the score
        # rather than beside it. Ranking eligibility is the caller's to
        # declare -- it depends on the policy's camera scope, which the manifest
        # cannot see -- and nothing about the faulted set forces it either way.
    if m.onset_step is not None:
        if not 0 <= int(m.onset_step) < int(m.official_horizon):
            raise ManifestError(
                f"onset_step {m.onset_step} inconsistent with official_horizon {m.official_horizon}"
            )
    if m.end_step is not None:
        if m.onset_step is None:
            raise ManifestError("end_step given without onset_step")
        if not int(m.onset_step) < int(m.end_step) <= int(m.official_horizon):
            raise ManifestError("end_step must satisfy onset_step < end_step <= official_horizon")
    if m.fault_mode == "stale_k":
        # The delay is either given in steps or derived by the runner from the
        # protocol's 500 ms at nominal_control_hz; a manifest with neither has no
        # executable delay, and a given delay must be at least one step.
        if m.stale_k is None and not m.nominal_control_hz:
            raise ManifestError("stale_k mode requires stale_k >= 1 or nominal_control_hz")
        if m.stale_k is not None and int(m.stale_k) < 1:
            raise ManifestError("stale_k mode requires stale_k >= 1")
    if m.cell_id is None:
        m.cell_id = m.derived_cell_id()
    return m


# ---------------------------------------------------------------------------
# Result record
# ---------------------------------------------------------------------------

@dataclass
class ResultRecord:
    """The protocol's minimum per-cell result fields plus seed echoes."""

    cell_id: Optional[str] = None
    method_id: Optional[str] = None
    checkpoint_hash: Optional[str] = None
    clean_or_fault: Optional[str] = None
    official_metrics: Optional[dict[str, Any]] = None
    success_or_valid: Optional[bool] = None
    availability_trace_hash: Optional[str] = None
    action_or_prediction_hash: Optional[str] = None
    runtime_ms: Optional[float] = None
    peak_vram_mb: Optional[float] = None
    failure_reason: Optional[str] = None
    completed: Optional[bool] = None
    environment_step_count: Optional[int] = None
    policy_query_count: Optional[int] = None
    action_execution_count: Optional[int] = None
    faulted_observations_produced: Optional[int] = None
    faulted_observations_consumed: Optional[int] = None
    #: The step at which the fault first reached the policy. Observed during the
    #: rollout, so it records the query boundary that was actually used rather
    #: than one derived from a nominal replanning interval.
    realized_onset_step: Optional[int] = None
    #: The step the fault was scheduled for, copied from the manifest so a
    #: reader can set realized against target without the manifest in hand.
    onset_step: Optional[int] = None
    source_age_summary_ms: Optional[dict[str, float]] = None
    max_pairwise_capture_skew_summary_ms: Optional[dict[str, float]] = None
    capture_timestamp_trace_hash: Optional[str] = None
    arrival_timestamp_trace_hash: Optional[str] = None
    sequence_trace_hash: Optional[str] = None
    policy_query_trace_hash: Optional[str] = None
    policy_input_trace_hash: Optional[str] = None
    action_chunk_invalidation_trace_hash: Optional[str] = None
    # Prefix-equivalence evidence.  A whole-trace hash cannot prove that two arms
    # shared a trajectory prefix, so the running prefix digests are stored too:
    # ``action_prefix_hash_chain[i]`` covers actions ``0..i`` and
    # ``policy_input_prefix_hash_chain[i]`` covers the first ``i+1`` policy
    # queries.  A fault result additionally carries the digests frozen at its own
    # onset, which are compared against the healthy chain at the same step.
    action_prefix_hash_chain: Optional[list[str]] = None
    policy_input_prefix_hash_chain: Optional[list[str]] = None
    pre_fault_action_hash: Optional[str] = None
    pre_fault_policy_input_hash: Optional[str] = None
    # Fault exposure.  ``faulted_observations_consumed`` is the visual exposure
    # (the number of policy queries that read a faulted observation); these
    # fields are the control exposure that one such query can support.
    post_fault_action_steps: Optional[int] = None
    max_actions_after_fault_query: Optional[int] = None
    native_action_chunk_length: Optional[int] = None
    mean_query_interval: Optional[float] = None
    # seed echoes for pairing audits
    environment_seed: Optional[int] = None
    policy_seed: Optional[int] = None
    fault_seed: Optional[int] = None
    manifest_hash: Optional[str] = None
    schedule_hash: Optional[str] = None
    runner_config_hash: Optional[str] = None
    dataset_authorization_hash: Optional[str] = None
    method_config_hash: Optional[str] = None
    # onset / eligibility audit echoes (frozen in the execution schema)
    onset_basis: Optional[str] = None
    healthy_reference_success: Optional[bool] = None
    fault_reached: Optional[bool] = None
    recovery_eligible: Optional[bool] = None
    onset_reference_hash: Optional[str] = None
    onset_fallback: Optional[bool] = None
    subset_ranking_eligible: Optional[bool] = None
    ranking_eligible: Optional[bool] = None
    # validator outputs
    fault_cell_valid: bool = True
    invalid_reason: Optional[str] = None
    result_schema_version: int = RESULT_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


class ResultError(ValueError):
    """Raised when a result record cannot be matched to its manifest."""


def validate_result(r: ResultRecord, manifest: FaultManifest) -> ResultRecord:
    """Validate a result against its manifest.

    Raises :class:`ResultError` when ``cell_id`` or any echoed seed disagrees
    with the manifest, when ``clean_or_fault`` is not ``'clean'``/``'fault'``
    or when ``completed`` is ``True`` but ``success_or_valid`` is unset.
    A fault-arm result with ``faulted_observations_consumed`` missing or
    ``<= 0`` is *not* rejected but marked ``fault_cell_valid=False`` with an
    ``invalid_reason`` (the fault never reached the policy).  Returns ``r``.
    """
    if r.cell_id is None or manifest.cell_id is None or r.cell_id != manifest.cell_id:
        raise ResultError(f"cell_id mismatch: result={r.cell_id!r} manifest={manifest.cell_id!r}")
    for name in ("environment_seed", "policy_seed", "fault_seed"):
        rv, mv = getattr(r, name), getattr(manifest, name)
        if rv is not None and mv is not None and int(rv) != int(mv):
            raise ResultError(f"{name} mismatch: result={rv} manifest={mv}")
    if r.manifest_hash is not None and r.manifest_hash != manifest.semantic_hash():
        raise ResultError("manifest_hash mismatch")
    version = r.result_schema_version
    if isinstance(version, bool) or not isinstance(version, int):
        raise ResultError("result_schema_version must be an integer")
    if not 1 <= version <= RESULT_SCHEMA_VERSION:
        raise ResultError(
            f"result_schema_version {version} is not readable by this build "
            f"(supported: 1..{RESULT_SCHEMA_VERSION})"
        )
    if r.clean_or_fault not in ("clean", "fault"):
        raise ResultError(f"clean_or_fault must be 'clean' or 'fault', got {r.clean_or_fault!r}")
    # A fault rollout must never be able to declare itself clean: that would skip
    # every fault audit field below *and* let a faulted episode enter the healthy
    # denominator of clean_solvable and the recovery conditions.
    expected_clean_or_fault = "clean" if manifest.fault_mode == HEALTHY_MODE else "fault"
    if r.clean_or_fault != expected_clean_or_fault:
        raise ResultError(
            f"clean_or_fault={r.clean_or_fault!r} is inconsistent with manifest "
            f"fault_mode={manifest.fault_mode!r} (expected {expected_clean_or_fault!r})"
        )
    if r.completed and r.success_or_valid is None:
        raise ResultError("completed result must carry success_or_valid")
    if r.clean_or_fault == "fault":
        audit_fields = (
            "onset_basis",
            "healthy_reference_success",
            "fault_reached",
            "faulted_observations_produced",
            "onset_reference_hash",
            "onset_fallback",
        )
        missing_audit = [name for name in audit_fields if getattr(r, name) is None]
        if missing_audit:
            raise ResultError(f"fault result missing onset audit fields: {missing_audit}")
        for name in ("healthy_reference_success", "fault_reached", "onset_fallback"):
            if not isinstance(getattr(r, name), bool):
                raise ResultError(f"{name} must be boolean")
        for name in (
            "onset_basis",
            "healthy_reference_success",
            "onset_reference_hash",
            "onset_fallback",
        ):
            if getattr(r, name) != getattr(manifest, name):
                raise ResultError(
                    f"{name} mismatch: result={getattr(r, name)!r} "
                    f"manifest={getattr(manifest, name)!r}"
                )
        n = r.faulted_observations_consumed
        if isinstance(r.faulted_observations_produced, bool) or not isinstance(
            r.faulted_observations_produced, int
        ):
            raise ResultError("faulted_observations_produced must be an integer")
        if n is not None and (isinstance(n, bool) or not isinstance(n, int)):
            raise ResultError("faulted_observations_consumed must be an integer or None")
        produced = r.faulted_observations_produced
        if produced < 0:
            raise ResultError("faulted_observations_produced must be >= 0")
        if n is not None and n > produced:
            raise ResultError("faulted observations consumed cannot exceed those produced")
        if r.fault_reached is False and produced > 0:
            raise ResultError("faulted observations cannot be produced before the fault is reached")
        if r.fault_reached is False and n is not None and n > 0:
            raise ResultError("faulted observations cannot be consumed before the fault is reached")
        if n is None or n <= 0:
            r.fault_cell_valid = False
            r.invalid_reason = "fault arm consumed zero faulted observations"
        else:
            r.fault_cell_valid = True
            r.invalid_reason = None
        r.recovery_eligible = bool(r.healthy_reference_success and r.fault_reached)
        r.subset_ranking_eligible = bool(manifest.subset_ranking_eligible)
        r.ranking_eligible = bool(
            manifest.subset_ranking_eligible and r.recovery_eligible and r.fault_cell_valid
        )
    else:
        r.fault_cell_valid = True
        r.invalid_reason = None
        r.recovery_eligible = False
        r.subset_ranking_eligible = False
        r.ranking_eligible = False
    return r
