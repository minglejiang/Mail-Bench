import pytest

from mail_bench.onset import resolve_duration, resolve_onset, round_half_up


@pytest.mark.parametrize("x", [0.0, 0.4999, 0.5, 1.5, 2.5, 7.49, 7.5, 123.999])
def test_round_half_up_matches_int_plus_half(x):
    assert round_half_up(x) == int(x + 0.5)


def test_round_half_up_rejects_negative():
    with pytest.raises(ValueError):
        round_half_up(-0.1)


def test_onset_healthy_reference():
    r = resolve_onset(0.45, official_horizon=400, healthy_completion_step=210, healthy_success=True)
    assert r["onset_step"] == int(0.45 * 210 + 0.5) == 95
    assert r["basis"] == "healthy_reference"
    assert r["fault_evaluable"] is True
    assert r["recovery_eligible"] is True
    assert r["reference_horizon"] == 210


def test_a_scene_without_a_healthy_success_has_no_onset():
    """Thirty percent of a completion that never happened is not thirty percent.

    Taking a fraction of the horizon instead would answer a different question --
    how a policy handles a fault at some absolute step -- so the benchmark declines to
    place an onset at all and the scene's fault cells are not run.
    """
    for healthy_success, completion in ((False, 210), (None, 210), (True, None)):
        r = resolve_onset(0.15, official_horizon=300,
                          healthy_completion_step=completion,
                          healthy_success=healthy_success)
        assert r["onset_step"] is None
        assert r["basis"] == "healthy_unsolved"
        assert r["fault_evaluable"] is False
        assert r["skip_reason"] == "healthy_unsolved"
        assert r["recovery_eligible"] is False
        assert r["reference_horizon"] is None


def test_a_periodic_policy_is_faulted_at_its_next_look():
    """A policy that looks every k steps cannot be faulted between two looks."""
    from mail_bench.onset import realize_onset

    assert realize_onset(135, 5) == 135          # already on a boundary
    assert realize_onset(135, 8) == 136          # next boundary at or after
    assert realize_onset(0, 8) == 0
    # A longer chunk pushes the realized onset further from the target.
    assert realize_onset(101, 50) == 150


def test_onset_half_up_and_clamp():
    # Every onset is a fraction of a successful healthy completion, so the
    # rounding and clamping are exercised through one.
    def onset(fraction, **kwargs):
        return resolve_onset(fraction, 10, healthy_completion_step=10,
                             healthy_success=True, **kwargs)

    assert onset(0.75)["onset_step"] == 8              # 7.5 -> 8, half-up
    assert onset(0.45)["onset_step"] == 5              # 4.5 -> 5
    r = onset(1.0)
    # One faulted step must remain inside the official horizon.
    assert r["onset_step"] == 9 and r["clamped"] is True


def test_onset_validation():
    with pytest.raises(ValueError):
        resolve_onset(1.5, 10)
    with pytest.raises(ValueError):
        resolve_onset(0.5, 0)
    with pytest.raises(TypeError):
        resolve_onset(0.5, 10, rounding="banker")     # the protocol names no other rounding


def test_duration_half_up_and_to_end():
    # A fraction of the remaining EPISODE: the policy finished at 60 of a
    # 100-step horizon, so a 0.25 loss from step 30 lasts a quarter of the 30
    # steps it had left, not a quarter of the horizon's 70.
    assert resolve_duration(1.0, 30, 100, 60) is None
    assert resolve_duration(0.25, 30, 100, 60) == 30 + int(0.25 * 30 + 0.5)   # 38
    assert resolve_duration(0.5, 91, 100, 95) == 91 + int(0.5 * 4 + 0.5)      # 93
    assert resolve_duration(0.25, 99, 100, 100) == 100   # at least one faulted step
    # The horizon still bounds it: a policy that ran to the horizon cannot be
    # faulted past it.
    assert resolve_duration(1.0, 30, 100, 100) is None
    with pytest.raises(ValueError):
        resolve_duration(0.5, 100, 100, 100)


def test_the_duration_is_the_same_phase_for_a_fast_and_a_slow_policy():
    """One duration_fraction has to mean one thing across policies.

    Against the horizon's remainder it would not: with a 450-step horizon and
    the 45% onset the specification fixes, "0.50" would interrupt a policy
    finishing in 100 steps for twice its whole trajectory and one finishing in
    400 steps for a third of its own -- a sixfold spread, and harsher on the
    faster policy, which is the incomparability the phase-normalised onset
    exists to prevent.
    """
    horizon = 450
    losses = {}
    for completion in (100, 200, 400):
        onset = round_half_up(0.45 * completion)
        end = resolve_duration(0.50, onset, horizon, completion)
        losses[completion] = (end - onset) / completion
    assert all(abs(share - 0.275) < 0.01 for share in losses.values()), losses

    # The three settings stay distinct and 1.00 is exactly "to the end".
    completion, onset = 400, round_half_up(0.45 * 400)
    quarter = resolve_duration(0.25, onset, horizon, completion) - onset
    half = resolve_duration(0.50, onset, horizon, completion) - onset
    assert quarter * 2 == pytest.approx(half, abs=1)
    assert resolve_duration(1.0, onset, horizon, completion) is None
