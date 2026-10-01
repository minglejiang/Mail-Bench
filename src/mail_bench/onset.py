"""Onset and duration resolution for availability fault cells.

The protocol expresses fault timing as fractions:

* ``onset_fraction`` of the reference horizon -- 30 / 45 / 60 % in the
  ranking; a mechanism study may name others;
* ``duration_fraction`` of the remaining episode (T_healthy - onset_step)
  after onset, where 100 % (``dur=end``) is the ranking's only setting.

The reference horizon is the healthy completion step of the paired healthy run
of *the evaluated policy itself*, and only when that run succeeded
(``basis='healthy_reference'``).  There is no fallback: a fraction names a task
phase, and a phase exists only relative to whoever performed the task, so a
scene the policy never solved has no 30% to place a fault at.  Such a scene is
recorded as a clean failure and its fault cells are not executed
(``basis='healthy_unsolved'``, ``fault_evaluable=False``).

Two policies may therefore receive different absolute onset steps on the same
scene.  That is the estimand, not an inconsistency: a fixed absolute step is
early in the task for a slow policy and late for a fast one, and comparing on it
would measure speed rather than robustness.

All rounding is *half-up* and matches ``int(x + 0.5)`` for ``x >= 0``; the
protocol names no other rounding, so none is offered.
"""

from __future__ import annotations

from typing import Any, Optional

from .semantic_states import RANKING_ONSETS


def round_half_up(x: float) -> int:
    """Round a non-negative float half-up (``0.5 -> 1``, ``1.5 -> 2``)."""
    if x < 0:
        raise ValueError("round_half_up is defined for x >= 0 only")
    return int(x + 0.5)


def resolve_onset(
    fraction: float,
    official_horizon: int,
    healthy_completion_step: Optional[int] = None,
    healthy_success: Optional[bool] = None,
) -> dict[str, Any]:
    """Resolve an onset fraction to an absolute control step.

    Parameters
    ----------
    fraction:
        Onset fraction in ``[0, 1]``.
    official_horizon:
        Official maximum number of control steps of the task (``> 0``).
    healthy_completion_step:
        Step at which the paired healthy run completed the task, if any.
    healthy_success:
        Whether the paired healthy run succeeded.  Only a *successful*
        healthy completion is used as reference.
    Returns
    -------
    dict with keys ``onset_step``, ``basis``, ``fault_evaluable``, ``skip_reason``,
    ``recovery_eligible``, ``reference_horizon``, ``fraction``, ``clamped``, ``rounding``.
    The onset is clamped to ``[0, official_horizon - 1]`` so that at least
    one faulted step exists inside the official horizon.
    """
    if not 0.0 <= fraction <= 1.0:
        raise ValueError("onset fraction must be in [0, 1]")
    if official_horizon < 1:
        raise ValueError("official_horizon must be >= 1")

    use_reference = (
        healthy_success is True
        and healthy_completion_step is not None
        and healthy_completion_step > 0
    )
    if not use_reference:
        # No successful healthy trajectory means no task phase to speak of.
        # Thirty percent of a completion that never happened is not thirty
        # percent of anything, and taking a fraction of the horizon instead
        # would quietly answer a different question: how a policy handles a
        # fault at some absolute step. The scene is reported as a clean failure
        # and its fault cells are not executed.
        return {
            "onset_step": None,
            "basis": "healthy_unsolved",
            "fault_evaluable": False,
            "skip_reason": "healthy_unsolved",
            "recovery_eligible": False,
            "reference_horizon": None,
            "fraction": float(fraction),
            "clamped": False,
            "rounding": "half_up",
        }

    reference = int(healthy_completion_step)
    raw = round_half_up(fraction * reference)
    onset = max(0, min(int(official_horizon) - 1, raw))
    return {
        "onset_step": onset,
        "basis": "healthy_reference",
        "fault_evaluable": True,
        "skip_reason": None,
        "recovery_eligible": True,
        "reference_horizon": reference,
        "fraction": float(fraction),
        "clamped": onset != raw,
        "rounding": "half_up",
    }


def realize_onset(target_onset_step: int, query_interval: int) -> int:
    """Where a fault can actually begin for a policy that looks periodically.

    A policy that requests observations every ``k`` steps cannot be faulted
    between two of them: the first moment the intervention can reach it is the
    next visual query at or after the target. The cell means the realized step,
    and both are recorded so a reader can see the difference a long chunk makes.
    """
    if query_interval < 1:
        raise ValueError("query_interval must be >= 1")
    if target_onset_step < 0:
        raise ValueError("target_onset_step must be >= 0")
    remainder = target_onset_step % query_interval
    return target_onset_step if remainder == 0 else target_onset_step + (
        query_interval - remainder)


def resolve_duration(
    duration_fraction: float,
    onset_step: int,
    official_horizon: int,
    healthy_completion_step: int,
) -> Optional[int]:
    """Resolve a fraction of the remaining *episode* to an exclusive end step.

    The specification asks for fractions of the remaining episode, and the
    episode is the policy's own successful trajectory, not the platform's
    horizon. Measured against the horizon's remainder, one fraction would mean
    a different condition per policy, and harsher on the faster one: with a
    450-step horizon and a 45% onset, "0.50" would interrupt a policy
    finishing in 100 steps for 203 steps -- twice its whole trajectory -- and
    one finishing in 400 steps for 135, a third of its own. The onset is
    phase-normalised precisely to avoid that, and the
    duration has to be normalised the same way or the two axes of one table
    disagree.

    ``duration_fraction >= 1.0`` means "until the episode ends" and returns
    ``None``, which is the ranking's own condition. Otherwise
    ``end_step = onset_step + round(fraction * (T_healthy - onset_step))``,
    clamped to the horizon and to at least ``onset_step + 1`` so that every
    fault cell contains one faulted step.
    """
    if not 0.0 <= duration_fraction <= 1.0:
        raise ValueError("duration_fraction must be in [0, 1]")
    if onset_step < 0 or onset_step >= official_horizon:
        raise ValueError("onset_step must satisfy 0 <= onset_step < official_horizon")
    if healthy_completion_step < 1:
        raise ValueError("a duration needs the policy's own completion step")
    if duration_fraction >= 1.0:
        return None
    remaining = max(0, int(healthy_completion_step) - int(onset_step))
    end = int(onset_step) + round_half_up(duration_fraction * remaining)
    return max(int(onset_step) + 1, min(int(official_horizon), end))


# ---------------------------------------------------------------------------
# Per-model onset manifest
# ---------------------------------------------------------------------------
#: The ranking's onsets, the one definition in :mod:`semantic_states`. Each
#: names a task phase, not a step count.
RANKING_ONSET_FRACTIONS = RANKING_ONSETS


def onset_manifest_row(
    *,
    model_id: str,
    checkpoint_identity: str,
    scene_identity: str,
    healthy_success: bool,
    healthy_completion_step: Optional[int],
    official_horizon: int,
    query_interval: int = 1,
    fractions: tuple[float, ...] = RANKING_ONSET_FRACTIONS,
) -> dict[str, Any]:
    """One scene's row of a model's own onset manifest.

    The protocol publishes one of these per model rather than one shared
    reference file per platform.  A shared file would have to name some policy's
    trajectory as the definition of "45% of the task" for every other policy,
    which turns the sweep back into absolute-time perturbation for everyone but
    that one policy.  Each model carries its own manifest instead, and the rows
    are what makes a cross-model comparison legible as phase-normalized.

    The row is self-contained: given it, a reader can recompute every onset
    without the run that produced it.
    """
    row: dict[str, Any] = {
        "model_id": model_id,
        "checkpoint_identity": checkpoint_identity,
        "scene_identity": scene_identity,
        "healthy_success": bool(healthy_success),
        "healthy_completion_step": healthy_completion_step,
        "official_horizon": int(official_horizon),
        "query_interval": int(query_interval),
    }
    evaluable = True
    skip_reason: Optional[str] = None
    for fraction in fractions:
        resolved = resolve_onset(
            fraction,
            official_horizon,
            healthy_completion_step=healthy_completion_step,
            healthy_success=healthy_success,
        )
        key = f"onset_{int(round(fraction * 100)):02d}"
        target = resolved["onset_step"]
        row[f"{key}_target"] = target
        # A prediction, and labelled as one. A policy that looks every k steps
        # cannot be faulted between two queries, but it queries when its action
        # buffer empties, which need not land on a multiple of k. The step the
        # fault actually reached the policy is observed during the rollout and
        # recorded on the fault cell as realized_onset_step; this row is written
        # from the healthy phase, before any fault has run.
        row[f"{key}_realized_predicted"] = (
            realize_onset(target, int(query_interval)) if target is not None else None
        )
        if not resolved["fault_evaluable"]:
            evaluable = False
            skip_reason = resolved["skip_reason"]
    row["fault_evaluable"] = evaluable
    row["skip_reason"] = skip_reason
    return row
