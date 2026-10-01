"""Each model publishes its own onset manifest; there is no shared reference.

These lock the estimand rather than an implementation: a fault is placed at a
phase of *this* policy's successful execution, and where no such execution
exists, no fault is placed at all.
"""

import pytest

from mail_bench.onset import (
    RANKING_ONSET_FRACTIONS,
    onset_manifest_row,
    realize_onset,
    resolve_onset,
)

HORIZON = 520


def row(**kwargs):
    base = dict(
        model_id="m", checkpoint_identity="a" * 12, scene_identity="s0",
        healthy_success=True, healthy_completion_step=200, official_horizon=HORIZON,
    )
    base.update(kwargs)
    return onset_manifest_row(**base)


def test_a_scene_the_policy_never_solved_gets_no_onset_at_all():
    unsolved = row(healthy_success=False, healthy_completion_step=None)
    assert unsolved["fault_evaluable"] is False
    assert unsolved["skip_reason"] == "healthy_unsolved"
    for fraction in RANKING_ONSET_FRACTIONS:
        key = f"onset_{int(round(fraction * 100)):02d}"
        assert unsolved[f"{key}_target"] is None
        assert unsolved[f"{key}_realized_predicted"] is None


def test_there_is_no_horizon_fallback():
    # Falling back to a fraction of the official horizon would answer a different
    # question -- how a policy handles a fault at some absolute step -- and
    # silently mix it into a phase-normalized table.
    resolved = resolve_onset(0.45, HORIZON, healthy_completion_step=None, healthy_success=False)
    assert resolved["onset_step"] is None
    assert resolved["basis"] == "healthy_unsolved"
    assert "fallback" not in resolved["basis"]
    # And a healthy run that failed cannot be rescued by a completion step.
    assert resolve_onset(
        0.45, HORIZON, healthy_completion_step=200, healthy_success=False
    )["onset_step"] is None


def test_two_policies_get_different_onsets_on_the_same_scene():
    # The point of phase normalization: a policy that finishes in 100 steps and
    # one that takes 400 are both faulted at 45% of their own task, not at the
    # same wall-clock step.
    fast = row(model_id="fast", healthy_completion_step=100)
    slow = row(model_id="slow", healthy_completion_step=400)
    assert fast["onset_45_target"] == 45
    assert slow["onset_45_target"] == 180
    assert fast["scene_identity"] == slow["scene_identity"]


def test_an_onset_is_reproducible_from_the_row_alone():
    # A published manifest must let a reader recompute every onset without the
    # run that produced it.
    published = row(healthy_completion_step=337, query_interval=5)
    for fraction in RANKING_ONSET_FRACTIONS:
        key = f"onset_{int(round(fraction * 100)):02d}"
        target = int(fraction * published["healthy_completion_step"] + 0.5)
        assert published[f"{key}_target"] == target
        assert published[f"{key}_realized_predicted"] == realize_onset(target, 5)


def test_the_same_model_and_scene_always_give_the_same_onset():
    assert row(healthy_completion_step=337) == row(healthy_completion_step=337)


def test_a_chunked_policy_records_both_target_and_realized():
    # A policy that looks every eight steps cannot be faulted between queries.
    chunked = row(healthy_completion_step=200, query_interval=8)
    assert chunked["onset_30_target"] == 60
    assert chunked["onset_30_realized_predicted"] == 64   # next query at or after 60
    assert chunked["onset_60_target"] == chunked["onset_60_realized_predicted"] == 120
    # A policy that looks every step has nothing to realize.
    assert row(healthy_completion_step=200)["onset_30_realized_predicted"] == 60


def test_realized_onset_never_precedes_the_target():
    for interval in (1, 3, 8, 50):
        for target in range(0, 60):
            realized = realize_onset(target, interval)
            assert realized >= target
            assert realized % interval == 0
    with pytest.raises(ValueError):
        realize_onset(10, 0)
