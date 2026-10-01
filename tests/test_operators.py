import numpy as np
import pytest

from mail_bench.interfaces import CanonicalObservation
from mail_bench.operators import (
    CausalViolation,
    FaultInjector,
    FaultSchedule,
    Poison,
    burst_draw,
    burst_hash,
    burst_schedule,
    canonical_mode,
    poison_test,
    zero_like,
)

CAMS = ["head", "left_wrist", "right_wrist"]


def frame(cam, step):
    return ("frame", cam, step)


def healthy(step):
    return {c: frame(c, step) for c in CAMS}


def run(schedule, n_steps):
    inj = FaultInjector(schedule)
    out = [inj.apply(t, healthy(t)) for t in range(n_steps)]
    return inj, out


def test_canonical_mode_aliases():
    assert canonical_mode("missing") == "hard_missing"
    assert canonical_mode("burst") == "burst_dropout"
    assert canonical_mode("stale") == "stale_k"
    assert canonical_mode("Freeze") == "freeze"
    with pytest.raises(ValueError):
        canonical_mode("fogbank")


def test_hard_missing_semantics():
    inj, out = run(FaultSchedule({"head"}, "missing", onset_step=3), 6)
    for t, (obs, avail, rec) in enumerate(out):
        if t < 3:
            assert obs["head"] == frame("head", t) and avail["head"]
            assert not rec.faulted_observation_produced
        else:
            assert obs["head"] is None and avail["head"] is False
            assert rec.source_step["head"] is None
            assert rec.faulted_observation_produced
        assert obs["left_wrist"] == frame("left_wrist", t)
    assert inj.faulted_observations_produced == 3


def test_blackout_zero_frame_available():
    np = pytest.importorskip("numpy")
    sched = FaultSchedule({"head"}, "blackout", onset_step=1)
    inj = FaultInjector(sched)
    img = np.full((2, 2, 3), 7, dtype=np.uint8)
    inj.apply(0, {"head": img})
    obs, avail, rec = inj.apply(1, {"head": img})
    assert avail["head"] is True
    assert obs["head"].shape == img.shape and obs["head"].dtype == img.dtype
    assert not obs["head"].any()
    assert rec.faulted_observation_produced
    assert zero_like([1.5, (2, 3)]) == [0.0, (0, 0)]


def test_freeze_uses_last_pre_onset_frame():
    inj, out = run(FaultSchedule({"head"}, "freeze", onset_step=4), 8)
    for t in range(4, 8):
        obs, avail, rec = out[t]
        assert obs["head"] == frame("head", 3)
        assert avail["head"] is True
        assert rec.source_step["head"] == 3
        assert rec.source_age_steps["head"] == t - 3


def test_freeze_at_onset_zero_degrades_to_hard_missing():
    inj, out = run(FaultSchedule({"head"}, "freeze", onset_step=0), 3)
    for obs, avail, rec in out:
        assert obs["head"] is None
        assert avail["head"] is False
        assert obs["head"] != 0  # never blackout


def test_stale_k_delivers_delayed_frames_and_history_shortage_is_missing():
    inj, out = run(FaultSchedule({"head"}, "stale", onset_step=0, stale_k=3), 6)
    for t in range(3):
        obs, avail, rec = out[t]
        assert obs["head"] is None and avail["head"] is False
    for t in range(3, 6):
        obs, avail, rec = out[t]
        assert obs["head"] == frame("head", t - 3)
        assert avail["head"] is True
        assert rec.source_age_steps["head"] == 3


def test_stale_k_may_deliver_post_onset_frames_only_while_available():
    inj, out = run(FaultSchedule({"head"}, "stale_k", onset_step=2, stale_k=1), 5)
    obs, avail, rec = out[4]
    assert obs["head"] == frame("head", 3)  # post-onset, but available-but-delayed
    assert avail["head"] is True and rec.source_step["head"] == 3


def test_burst_schedule_is_deterministic_and_per_block():
    bits = burst_schedule(12345, 60, burst_frames=15, available_rate=0.7)
    assert bits == burst_schedule(12345, 60, 15, 0.7)
    assert len(bits) == 60
    # constant within a block; one draw per block, from the second on
    for b in range(1, 4):
        block = bits[b * 15:(b + 1) * 15]
        assert all(x == block[0] for x in block)
        assert block[0] == (burst_draw(12345, b) < 0.7)
    assert burst_schedule(1, 30, 5, 0.0) == [False] * 30
    assert len(burst_hash(bits)) == 64


def test_the_first_burst_block_is_always_missing():
    """Otherwise the fault does not begin at the onset the manifest records.

    A first block that came up available would leave the stream healthy through
    it, so the real onset would be a block later than the recorded one -- and
    two scenes with the same nominal onset would have been faulted at different
    points, with the difference looking like policy behaviour.
    """
    for seed in (1, 7, 12345, 54321):
        for rate in (0.5, 0.7, 1.0):
            bits = burst_schedule(seed, 40, 10, rate)
            assert not any(bits[:10]), (seed, rate)
    # A draw that would have made the first block available is overridden, and
    # the later blocks still follow it.
    assert burst_draw(12345, 0) < 0.7          # would have been available
    assert burst_schedule(12345, 40, 10, 0.7)[0] is False
    # available_rate 1.0 means every block after the first.
    assert burst_schedule(1, 30, 5, 1.0) == [False] * 5 + [True] * 25


def test_burst_dropout_injector_matches_schedule():
    sched = FaultSchedule({"head"}, "burst", onset_step=5, burst_frames=4, available_rate=0.5,
                          fault_seed=99, recovery="periodic_burst")
    inj, out = run(sched, 40)
    bits = burst_schedule(99, 35, 4, 0.5)
    for t in range(5, 40):
        obs, avail, rec = out[t]
        assert avail["head"] == bits[t - 5]
        if bits[t - 5]:
            assert obs["head"] == frame("head", t)
        else:
            assert obs["head"] is None
    assert any(bits) and not all(bits)


def test_recovery_window_end_step_exclusive():
    sched = FaultSchedule({"head"}, "hard_missing", onset_step=2, end_step=4, recovery="single_recovery")
    inj, out = run(sched, 6)
    assert [o[1]["head"] for o in out] == [True, True, False, False, True, True]


def test_schedule_validation():
    with pytest.raises(ValueError):
        FaultSchedule({"a"}, "hard_missing", onset_step=2, end_step=2)
    with pytest.raises(ValueError):
        FaultSchedule({"a"}, "hard_missing", onset_step=2, end_step=5)  # recovery none but finite end
    with pytest.raises(ValueError):
        FaultSchedule({"a"}, "stale_k", onset_step=0)
    with pytest.raises(ValueError):
        FaultSchedule({"a"}, "freeze", recovery="periodic_burst")


@pytest.mark.parametrize("mode,kw", [
    ("hard_missing", {}),
    ("blackout", {}),
    ("freeze", {}),
    ("stale_k", {"stale_k": 2}),
    ("burst_dropout", {"burst_frames": 3, "available_rate": 0.5, "fault_seed": 7}),
])
def test_poison_test_passes_for_all_modes(mode, kw):
    for onset in (0, 4):
        inj = FaultInjector(FaultSchedule({"head", "left_wrist"}, mode, onset_step=onset, **kw))
        summary = poison_test(inj, CAMS, 20)
        assert summary["n_steps"] == 20
        if mode in ("hard_missing", "freeze") and onset == 0:
            assert summary["unavailable_camera_steps"] == 40
        if mode == "freeze":
            assert summary["poison_delivered_while_available"] == 0


def test_poison_never_delivered_while_unavailable_and_freeze_pre_onset():
    inj = FaultInjector(FaultSchedule({"head"}, "freeze", onset_step=3))
    for t in range(10):
        obs, avail, rec = inj.apply(t, {"head": Poison("head", t) if t >= 3 else frame("head", t)})
        assert not (isinstance(obs["head"], Poison))
        if t >= 3:
            assert rec.source_step["head"] == 2


def test_injector_causal_self_check_raises_on_tampering():
    inj = FaultInjector(FaultSchedule({"head"}, "freeze", onset_step=2))
    inj.apply(0, {"head": 0})
    inj.apply(1, {"head": 1})
    # tamper: pretend a post-onset frame is pre-onset history
    inj._history["head"][5] = "hidden"
    object.__setattr__(inj.schedule, "onset_step", 6)
    inj.apply(2, {"head": 2})
    object.__setattr__(inj.schedule, "onset_step", 2)
    inj._pre_onset_frame = lambda cam: (5, "hidden")
    with pytest.raises(CausalViolation):
        inj.apply(3, {"head": 3})


def test_steps_must_increase():
    inj = FaultInjector(FaultSchedule({"head"}, "hard_missing"))
    inj.apply(0, healthy(0))
    with pytest.raises(ValueError):
        inj.apply(0, healthy(0))


def test_availability_trace_hash_stable():
    a, _ = run(FaultSchedule({"head"}, "hard_missing", onset_step=2), 5)
    b, _ = run(FaultSchedule({"head"}, "hard_missing", onset_step=2), 5)
    c, _ = run(FaultSchedule({"head"}, "hard_missing", onset_step=3), 5)
    assert a.availability_trace_hash() == b.availability_trace_hash() != c.availability_trace_hash()


def test_the_burst_block_is_500_ms_at_the_platforms_control_rate():
    """The protocol says 500 ms; the manifest's control rate turns the
    protocol's milliseconds into steps, and a manifest with neither is refused
    rather than defaulted."""
    from mail_bench.manifest import PROTOCOL_VERSION, FaultManifest, validate_manifest
    from mail_bench.operators import CANONICAL_BURST_BLOCK_MS, control_steps_for
    from mail_bench.runner import schedule_from_manifest

    assert control_steps_for(CANONICAL_BURST_BLOCK_MS, 20.0) == 10
    assert control_steps_for(500, 30.0) == 15

    def burst_manifest(**overrides):
        values = dict(
            dataset_version="rev", split="stage_0", scene_or_task="t", episode_or_sequence=0,
            camera_ids=["head", "wrist"], faulted_camera_ids=["head"],
            perturbation_family="availability", fault_mode="burst_dropout",
            onset_fraction=0.5, duration_fraction=1.0, environment_seed=1, policy_seed=2,
            fault_seed=3, official_horizon=40, adapter_version="v", protocol_version=PROTOCOL_VERSION,
            onset_step=20, onset_basis="healthy_reference", healthy_reference_success=True,
            onset_reference_hash="b" * 64, onset_fallback=False, subset_ranking_eligible=True,
            available_rate=0.5,
        )
        values.update(overrides)
        return validate_manifest(FaultManifest(**values))

    assert schedule_from_manifest(burst_manifest(nominal_control_hz=20.0)).burst_frames == 10
    assert schedule_from_manifest(burst_manifest(burst_frames=7)).burst_frames == 7
    with pytest.raises(ValueError, match="burst_frames or nominal_control_hz"):
        schedule_from_manifest(burst_manifest())


def test_the_stale_delay_is_500_ms_at_the_platforms_control_rate():
    """Same rule as the burst block: the manifest's control rate turns the
    protocol's 500 ms into steps, a hand-filled delay must agree with it, and
    a manifest with neither is refused."""
    from mail_bench.manifest import PROTOCOL_VERSION, FaultManifest, ManifestError, validate_manifest
    from mail_bench.runner import schedule_from_manifest

    def stale_manifest(**overrides):
        values = dict(
            dataset_version="rev", split="stage_0", scene_or_task="t", episode_or_sequence=0,
            camera_ids=["head", "wrist"], faulted_camera_ids=["head"],
            perturbation_family="availability", fault_mode="stale_k",
            onset_fraction=0.5, duration_fraction=1.0, environment_seed=1, policy_seed=2,
            fault_seed=3, official_horizon=40, adapter_version="v", protocol_version=PROTOCOL_VERSION,
            onset_step=20, onset_basis="healthy_reference", healthy_reference_success=True,
            onset_reference_hash="b" * 64, onset_fallback=False, subset_ranking_eligible=True,
        )
        values.update(overrides)
        return FaultManifest(**values)

    assert schedule_from_manifest(stale_manifest(nominal_control_hz=20.0)).stale_k == 10
    assert schedule_from_manifest(stale_manifest(nominal_control_hz=30.0)).stale_k == 15
    assert schedule_from_manifest(stale_manifest(nominal_control_hz=20.0, stale_k=10)).stale_k == 10
    assert schedule_from_manifest(stale_manifest(stale_k=3)).stale_k == 3
    with pytest.raises(ValueError, match="not the protocol's 500 ms"):
        schedule_from_manifest(stale_manifest(nominal_control_hz=20.0, stale_k=15))
    with pytest.raises(ManifestError, match="stale_k >= 1 or nominal_control_hz"):
        validate_manifest(stale_manifest())
    with pytest.raises(ManifestError, match="stale_k >= 1"):
        validate_manifest(stale_manifest(nominal_control_hz=20.0, stale_k=0))


def test_freeze_uses_the_last_frame_that_was_actually_there():
    """A pre-onset step whose frame was absent is not a frame to freeze on."""
    schedule = FaultSchedule({"head"}, "freeze", onset_step=3)
    injector = FaultInjector(schedule)
    frames = {0: frame("head", 0), 1: None, 2: None}
    for step in range(3):
        injector.apply(step, {"head": frames[step], "wrist": frame("wrist", step)})
    delivered, available, trace = injector.apply(3, healthy(3))
    assert delivered["head"] == frame("head", 0)
    assert available["head"] is True
    assert trace.source_step["head"] == 0



def test_a_delivered_frame_describes_itself_and_not_the_current_step():
    """What the message says about the frame must be true of that frame.

    A frozen or delayed camera must not arrive with new_frame=True, the current
    step's sequence number and the current capture time beside a
    source_age_steps saying the frame is several steps old: that would be one
    message contradicting itself. An availability-aware policy reads exactly these
    fields to decide whether it is looking at something new.

    blackout is the deliberate exception. Its frame really is captured now and
    delivered now; only the pixels are wrong, which is why it is the one mode
    whose metadata a policy cannot tell from a healthy camera's.
    """
    cameras = ("agentview", "wrist")
    onset, k = 4, 3

    def healthy(step):
        return CanonicalObservation(
            step=step,
            cameras={c: np.full((2, 2, 3), step, np.uint8) for c in cameras},
            availability={c: True for c in cameras},
            capture_time_ms={c: 50.0 * step for c in cameras},
            arrival_time_ms={c: 50.0 * step for c in cameras},
            sequence_id={c: step for c in cameras},
            new_frame={c: True for c in cameras},
            source_step={c: step for c in cameras},
            source_age_steps={c: 0 for c in cameras},
        )

    def delivered(mode, step, **extra):
        schedule = FaultSchedule(frozenset(["wrist"]), mode=mode, onset_step=onset,
                                 end_step=None, fault_seed=11, recovery="none", **extra)
        injector = FaultInjector(schedule)
        for prior in range(step + 1):
            frames, availability, record = injector.apply(prior, healthy(prior).cameras)
        return healthy(step).with_fault(
            frames, availability,
            source_step=record.source_step, source_age_steps=record.source_age_steps)

    step = onset + k
    for mode, fresh in (("freeze", False), ("stale_k", False), ("blackout", True),
                        ("hard_missing", False)):
        extra = {"stale_k": k} if mode == "stale_k" else {}
        observation = delivered(mode, step, **extra)
        assert observation.new_frame["agentview"] is True, mode      # untouched camera
        assert observation.new_frame["wrist"] is fresh, mode
        if fresh:
            assert observation.sequence_id["wrist"] == step, mode
            assert observation.capture_time_ms["wrist"] == 50.0 * step, mode
        else:
            assert observation.sequence_id["wrist"] is None, mode
            assert observation.capture_time_ms["wrist"] is None, mode
        # arrival is when the delivery happened, and it happened now
        assert observation.arrival_time_ms["wrist"] == 50.0 * step, mode
