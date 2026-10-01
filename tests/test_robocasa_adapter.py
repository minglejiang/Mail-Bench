"""The six facts the RoboCasa adapter has to establish.

The simulator is faked -- MuJoCo, robosuite and RoboCasa have their own tests,
and none of them are needed to check what this adapter is responsible for. What
is real here is the adapter, the frozen-scene replay contract, the canonical
observation and the shared fault injector.

The point of most of these is a boundary rather than a behaviour: the adapter
says what RoboCasa observes, and it must not be the thing that decides what a
fault means.
"""

import copy

import numpy as np
import pytest

from mail_bench.operators import FaultInjector, FaultSchedule
from mail_bench.platforms.robocasa import (
    ROBOCASA_ACTION_DIM,
    ROBOCASA_ACTION_SLOTS,
    ROBOCASA_CAMERA_KEYS,
    convert_action,
    RoboCasaEnvironmentAdapter,
    robocasa_robot_state,
)
from mail_bench.scenes import (
    SCENE_BANK_ID,
    SCENE_SCHEMA_VERSION,
    FrozenEpisode,
    ScenePairingError,
    _sha256_bytes,
    semantic_sha256,
    state_digest,
)

EP_META = {"lang": "put the bowl in the cabinet", "layout_id": 3, "objects": [{"n": 1}]}
MODEL_XML = "<mujoco/>"

CAMERAS = tuple(sorted(ROBOCASA_CAMERA_KEYS))
STATE0 = [float(i) * 0.5 for i in range(24)]


def raw_observation(step, state):
    """What RoboCasa's wrapper publishes, keyed the way RoboCasa keys it."""
    observation = {
        raw: np.full((4, 4, 3), step + index, np.uint8)
        for index, raw in enumerate(sorted(ROBOCASA_CAMERA_KEYS.values()))
    }
    observation.update({
        # Widths as RoboCasa's gym wrapper publishes them: the rotations are
        # quaternions and the gripper has two joints.
        "state.end_effector_position_relative": np.array(state[:3], np.float32),
        "state.end_effector_rotation_relative": np.array(state[3:7], np.float32),
        "state.base_position": np.array(state[7:10], np.float32),
        "state.base_rotation": np.array(state[10:14], np.float32),
        "state.gripper_qpos": np.array(state[14:16], np.float32),
    })
    return observation


class FakeSim:
    """Just enough MuJoCo for the replay path."""

    def __init__(self, state, drift=0.0):
        self.state = np.asarray(state, np.float64)
        self.drift = drift

    def reset(self):
        self.state = np.zeros_like(self.state)

    def set_state_from_flattened(self, state):
        self.state = np.asarray(state, np.float64).copy()

    def forward(self):
        # A simulator whose settling moved the scene: the property the replay
        # path exists to rule out.
        self.state = self.state + self.drift

    def get_state(self):
        return self

    def flatten(self):
        return self.state


class FakeInner:
    """The robosuite environment inside the wrapper."""

    def __init__(self, state, drift=0.0):
        self.rng = None
        self.sim = FakeSim(state, drift=drift)
        self.ep_meta = None
        self.xml = None
        self.reset_calls = 0
        self.stepped_directly = 0

    # The replay path scenes.restore_episode drives.
    def set_ep_meta(self, ep_meta):
        self.ep_meta = ep_meta

    def get_ep_meta(self):
        return self.ep_meta

    def reset(self):
        self.reset_calls += 1

    def edit_model_xml(self, xml):
        return xml

    def reset_from_xml_string(self, xml):
        self.xml = xml

    def _get_observations(self, force_update=False):
        return raw_observation(0, self.sim.state)

    def step(self, action):                       # pragma: no cover - must not run
        self.stepped_directly += 1
        raise AssertionError("actions must go through the gym wrapper")


class FakeWrapper:
    """The gym wrapper: it owns action unmapping and success bookkeeping."""

    def __init__(self, inner, succeed_at=None):
        self.env = inner
        self.unwrapped = self
        self.step_index = 0
        self.succeed_at = succeed_at
        self.actions = []
        self.closed = False

    def get_observation(self, raw):
        return raw

    def reset(self, seed=None):
        self.step_index = 0
        self.actions = []

    def step(self, action):
        self.step_index += 1
        assert isinstance(action, dict), "the wrapper is handed RoboCasa's action dict"
        self.actions.append(action)
        success = self.succeed_at is not None and self.step_index >= self.succeed_at
        return (raw_observation(self.step_index, self.env.sim.state), 0.0, success,
                False, {"success": success})

    def close(self):
        self.closed = True


def frozen(state=STATE0, ep_meta=None):
    return FrozenEpisode(
        schema_version=SCENE_SCHEMA_VERSION, scene_bank_id=SCENE_BANK_ID,
        namespace="official", platform="robocasa365",
        task="PnPCounterToCab", episode_index=7, split="target", episode_seed=1234,
        environment_revision="a07e365c", asset_inventory_sha256="f" * 64,
        ep_meta_sha256=semantic_sha256(ep_meta if ep_meta is not None else EP_META),
        model_xml_sha256=_sha256_bytes(MODEL_XML.encode("utf-8")),
        state0_sha256=state_digest(state), state0_size=len(state),
        captured_after_settling=True,
    )


def adapter(*, state=STATE0, succeed_at=None, drift=0.0, ep_meta=None):
    meta = EP_META if ep_meta is None else ep_meta
    inner = FakeInner(state, drift=drift)
    wrapper = FakeWrapper(inner, succeed_at=succeed_at)
    return RoboCasaEnvironmentAdapter(
        wrapper, frozen(state, meta), meta, MODEL_XML, state,
        dataset_revision="a07e365c", official_horizon=500,
    ), wrapper, inner


# -- 1. frozen reset hash exact ---------------------------------------------

def test_a_frozen_scene_restores_to_exactly_the_state_it_was_frozen_at():
    subject, _, inner = adapter()
    subject.reset(4321)
    assert subject.start_state_sha256 == subject.episode.state0_sha256
    assert inner.xml == MODEL_XML
    assert subject.official_metrics()["start_state_sha256"] == state_digest(STATE0)


def test_a_scene_that_restores_elsewhere_fails_closed():
    # Not a harder scene -- a different one. A healthy completion measured here
    # would define an onset for a trajectory the fault arms never take. The gate
    # belongs to the replay path itself, which is why the adapter carries no
    # second check of its own: one that could never fire would only suggest the
    # adapter is what enforces the pairing.
    subject, _, _ = adapter(drift=1e-3)
    with pytest.raises(ScenePairingError, match="landed on state"):
        subject.reset(1)


def test_the_episode_metadata_the_caller_holds_is_never_mutated():
    # RoboCasa rewrites the ep_meta it is handed while building the model, which
    # would corrupt the fixture and make the next replicate reject a correct
    # scene.
    fixture = copy.deepcopy(EP_META)
    before = copy.deepcopy(fixture)
    subject, _, _ = adapter(ep_meta=fixture)
    subject.reset(0)
    subject.ep_meta["objects"][0]["n"] = 999          # the environment's own edit
    assert fixture == before


# -- 2. camera canonicalization ---------------------------------------------

def test_the_three_policy_cameras_arrive_canonically_named_and_all_available():
    subject, _, _ = adapter()
    observation = subject.reset(0)
    assert tuple(sorted(observation.cameras)) == CAMERAS
    assert subject.camera_inventory() == CAMERAS
    # Healthy means every policy camera present. Availability is the injector's
    # to change, downstream of here.
    assert observation.availability == {camera: True for camera in CAMERAS}
    assert all(frame is not None for frame in observation.cameras.values())
    assert observation.language == "put the bowl in the cabinet"


def test_the_proprio_vector_is_robocasas_own_order_not_a_models():
    state = robocasa_robot_state(raw_observation(0, STATE0))
    assert state.shape == (16,)
    assert state.dtype == np.float32
    # 3 eef position + 4 eef rotation + 3 base position + 4 base rotation + 2 gripper
    np.testing.assert_allclose(state, np.array(STATE0[:16], np.float32))
    with pytest.raises(KeyError, match="missing robot-state keys"):
        robocasa_robot_state({"state.base_position": np.zeros(3)})


def test_a_proprio_key_of_the_wrong_width_fails_closed():
    """A by-name consumer slices the flat vector with the transcribed widths.

    The rotation keys are quaternions in RoboCasa; a three-wide rotation would
    still sum to a plausible vector if the gripper absorbed the difference,
    and every downstream slice would be off by one. So the width of each key
    is checked, not only the total.
    """
    observation = raw_observation(0, STATE0)
    observation["state.end_effector_rotation_relative"] = np.zeros(3, np.float32)
    observation["state.gripper_qpos"] = np.zeros(3, np.float32)
    with pytest.raises(ValueError, match="end_effector_rotation_relative"):
        robocasa_robot_state(observation)


# -- 3. the gym action path --------------------------------------------------

def test_actions_go_through_the_wrapper_and_never_the_inner_environment():
    # The wrapper owns action unmapping and the gripper and base-mode
    # thresholding. Stepping the inner environment would be a different
    # controller, and the benchmark would not be measuring the platform.
    subject, wrapper, inner = adapter(succeed_at=3)
    subject.reset(0)
    action = np.arange(12, dtype=np.float32)
    step = subject.step(action)
    assert len(wrapper.actions) == 1
    # The wrapper receives RoboCasa's named action dict, not the flat vector:
    # naming the slots is the platform's action interface, and doing it in the
    # policy would make one platform's action dict part of what a submission has
    # to know.
    delivered = wrapper.actions[0]
    np.testing.assert_array_equal(delivered["action.end_effector_position"], action[0:3])
    np.testing.assert_array_equal(delivered["action.base_motion"], action[7:11])
    assert inner.stepped_directly == 0
    assert step.observation.step == 1
    assert subject.success() is False


def test_success_comes_from_the_wrappers_own_bookkeeping():
    subject, _, _ = adapter(succeed_at=2)
    subject.reset(0)
    assert subject.step(np.zeros(12, np.float32)).terminated is False      # not yet
    step = subject.step(np.zeros(12, np.float32))
    assert step.terminated is True
    assert subject.success() is True
    metrics = subject.official_metrics()
    assert metrics["success"] is True and metrics["control_steps"] == 2
    assert metrics["scene_identity"] == subject.episode.identity


# -- 4. healthy passthrough --------------------------------------------------

def test_a_healthy_rollout_delivers_every_camera_at_every_step():
    subject, _, _ = adapter(succeed_at=4)
    observation = subject.reset(0)
    seen = [observation]
    while not subject.success():
        seen.append(subject.step(np.zeros(12, np.float32)).observation)
    assert [o.step for o in seen] == [0, 1, 2, 3, 4]
    for observation in seen:
        assert observation.availability == {camera: True for camera in CAMERAS}
        assert set(observation.cameras) == set(CAMERAS)
        assert observation.new_frame == {camera: True for camera in CAMERAS}
        assert observation.source_age_steps == {camera: 0 for camera in CAMERAS}


# -- 5. the shared injector, not the adapter, makes a camera missing ---------

def test_hard_missing_is_applied_by_the_shared_injector_downstream():
    # The adapter contains no fault logic at all: the shared FaultInjector turns
    # a healthy RoboCasa observation into a faulted one, so no platform can
    # drift into its own definition of the same fault.
    subject, _, _ = adapter()
    injector = FaultInjector(FaultSchedule(
        faulted_cameras=frozenset({"robot0_agentview_left"}),
        mode="hard_missing", onset_step=2,
    ))
    healthy = subject.reset(0)
    delivered = []
    for step in range(5):
        observation = healthy if step == 0 else subject.step(np.zeros(12, np.float32)).observation
        frames, availability, _ = injector.apply(step, dict(observation.cameras))
        delivered.append((availability, frames))

    for step, (availability, frames) in enumerate(delivered):
        if step < 2:
            assert availability["robot0_agentview_left"] is True
            assert frames["robot0_agentview_left"] is not None
        else:
            assert availability["robot0_agentview_left"] is False
            assert frames["robot0_agentview_left"] is None
        # Only the faulted camera is touched.
        assert availability["robot0_eye_in_hand"] is True
        assert availability["robot0_agentview_right"] is True

    # And the adapter implements none of the modes itself: the docstring names
    # them to say where they live, but no code below it acts on one.
    import ast
    module = ast.parse((
        __import__("pathlib").Path(__file__).resolve().parents[1]
        / "src" / "mail_bench" / "platforms" / "robocasa.py"
    ).read_text(encoding="utf-8"))
    module.body = [n for n in module.body if not (
        isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant)
    )]
    code = ast.unparse(module)
    for mode in ("blackout", "stale_k", "burst_dropout", "hard_missing"):
        assert mode not in code, f"{mode} must live in the shared injector"


# -- 6. pre-fault trajectory equivalence ------------------------------------

def test_the_healthy_and_faulted_arms_are_identical_before_onset():
    onset = 3
    trajectories = []
    for schedule in (None, FaultSchedule(faulted_cameras=frozenset({"robot0_eye_in_hand"}),
                                         mode="hard_missing", onset_step=onset)):
        subject, _, _ = adapter()
        injector = FaultInjector(schedule) if schedule else None
        observation = subject.reset(99)
        frames = []
        for step in range(6):
            if step:
                observation = subject.step(np.full(12, 0.25, np.float32)).observation
            cameras = dict(observation.cameras)
            if injector is not None:
                cameras, _, _ = injector.apply(step, cameras)
            frames.append({
                camera: None if frame is None else bytes(np.asarray(frame).tobytes())
                for camera, frame in cameras.items()
            })
        trajectories.append(frames)

    healthy_arm, fault_arm = trajectories
    assert healthy_arm[:onset] == fault_arm[:onset], "arms diverged before onset"
    assert healthy_arm[onset:] != fault_arm[onset:], "the fault never arrived"


def test_the_action_slots_cover_the_vector_exactly_once():
    # Transcribed from robocasa.utils.env_utils rather than imported, so the
    # layout is checked here and verified against the installed RoboCasa at
    # server startup.
    covered = []
    for _, start, stop in ROBOCASA_ACTION_SLOTS:
        covered.extend(range(start, stop))
    assert covered == list(range(ROBOCASA_ACTION_DIM))
    named = convert_action(np.arange(12, dtype=np.float32))
    assert set(named) == {name for name, _, _ in ROBOCASA_ACTION_SLOTS}
    np.testing.assert_array_equal(named["action.gripper_close"], [6.0])
    np.testing.assert_array_equal(named["action.control_mode"], [11.0])
    with pytest.raises(ValueError, match="flat action of 12"):
        convert_action(np.zeros(7, np.float32))


def test_a_task_may_not_run_past_its_own_official_horizon():
    """RoboCasa's horizon is per task; the kernel's cap is per cohort.

    The kernel necessarily caps on one figure for the whole cohort, because that
    figure enters the cell identity and a per-task one would split a cohort into
    per-task measurements. If the adapter did not enforce its own task's budget,
    a short-horizon task would be allowed to run to the longest task's and would
    report a higher success rate than the official protocol permits.
    """
    inner = FakeInner(STATE0)
    wrapper = FakeWrapper(inner, succeed_at=None)          # never succeeds
    subject = RoboCasaEnvironmentAdapter(
        wrapper, frozen(), EP_META, MODEL_XML, STATE0,
        dataset_revision="a07e365c", official_horizon=3,
    )
    subject.reset(0)
    steps = [subject.step(np.zeros(12, np.float32)) for _ in range(3)]
    assert [s.terminated for s in steps] == [False, False, True]
    assert subject.success() is False
    metrics = subject.official_metrics()
    assert metrics["horizon_reached"] is True
    assert metrics["horizon_exhausted"] is True
    assert metrics["control_steps"] == metrics["official_horizon"] == 3


def test_a_task_that_finishes_early_reaches_no_horizon():
    subject, _, _ = adapter(succeed_at=2)
    subject.reset(0)
    subject.step(np.zeros(12, np.float32))
    step = subject.step(np.zeros(12, np.float32))
    assert step.terminated is True and subject.success() is True
    metrics = subject.official_metrics()
    assert metrics["horizon_reached"] is False
    assert metrics["horizon_exhausted"] is False


def test_success_on_the_last_permitted_step_is_not_a_budget_exhaustion():
    # Reaching the horizon and running out of budget are two different facts,
    # and they coincide on every step but this one. Conflating them would report
    # a rollout as having run out of time on the step it finished the task.
    inner = FakeInner(STATE0)
    wrapper = FakeWrapper(inner, succeed_at=3)
    subject = RoboCasaEnvironmentAdapter(
        wrapper, frozen(), EP_META, MODEL_XML, STATE0,
        dataset_revision="a07e365c", official_horizon=3,
    )
    subject.reset(0)
    for _ in range(3):
        step = subject.step(np.zeros(12, np.float32))
    assert step.terminated is True
    assert subject.success() is True
    metrics = subject.official_metrics()
    assert metrics["horizon_reached"] is True
    assert metrics["horizon_exhausted"] is False
    assert metrics["control_steps"] == 3


def test_the_environment_seed_never_reaches_the_trajectory():
    """The invariant a protocol bump depends on.

        same frozen state + same actions + same execution semantics
            => same execution

    The environment seed recorded in a manifest is derived from the protocol
    version. If it drove anything after the restore, bumping the protocol would
    move every healthy completion step and every onset. It is recorded and
    never used.
    """
    trajectories = []
    for env_seed in (11111, 99999):
        subject, wrapper, _ = adapter(succeed_at=None)
        subject.reset(env_seed)
        for _ in range(5):
            subject.step(np.full(12, 0.25, np.float32))
        delivered = [
            {name: np.asarray(value).tolist() for name, value in sorted(action.items())}
            for action in wrapper.actions
        ]
        trajectories.append((
            subject.start_state_sha256,
            delivered,
            subject.official_metrics()["control_steps"],
        ))
    assert trajectories[0] == trajectories[1]
    # The seed is carried into the record, so a reader can still see which one
    # the protocol derived for this unit.
    subject, _, _ = adapter()
    subject.reset(4242)
    assert subject.official_metrics()["requested_environment_seed"] == 4242
