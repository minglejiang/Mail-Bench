"""RoboCasa environment adapter with lazy optional simulator imports.

The adapter answers one question -- what does RoboCasa actually observe right
now -- and nothing else.  Fault semantics live in the shared execution kernel:
``hard_missing``, ``blackout``, ``freeze`` and the rest are applied to the
canonical observation *after* it leaves here, so that no platform adapter can
drift into its own subtly different definition of the same fault.  This
adapter therefore always reports every policy camera as available; changing that
is the injector's job.

Two boundaries are load-bearing and easy to cross by accident:

* **Restoring a frozen scene reaches the inner robosuite environment; executing
  an action does not.**  The gym wrapper owns action unmapping, gripper and
  base-mode thresholding, and the official success bookkeeping.  Stepping the
  inner environment directly would be a different controller, and a benchmark
  whose actions bypass the platform's own execution path is measuring something
  other than the platform.

* **The proprioceptive vector is RoboCasa's, not a model's.**  It is assembled
  from the platform's own observation keys in the order the official RoboCasa /
  OpenPI path uses.  A policy that wants a different proprio converts it in its
  own adapter; hard-coding one checkpoint's preference here would make one
  model's convenience the scientific definition of the platform.
"""

from __future__ import annotations

import copy
from typing import Any, Mapping, Optional, Sequence

from ..interfaces import CanonicalObservation, EnvironmentAdapter, EnvironmentStep
from ..scenes import FrozenEpisode, restore_episode


#: Two identifiers, deliberately not merged.
#:
#: ``ROBOCASA_DATASET_ID`` is the dataset registry's id, which is what an
#: authorization is checked against.  ``ROBOCASA_PLATFORM_ID`` is the scene
#: bank's platform id, which is baked into the bank's directory layout and into
#: every frozen scene's seed, so renaming it would invalidate an existing bank.
#: They are different strings for different jobs; one id per job keeps the two
#: vocabularies from drifting apart.
ROBOCASA_DATASET_ID = "robocasa"
ROBOCASA_PLATFORM_ID = "robocasa365"

#: Canonical camera id -> the key the gym wrapper publishes.  The canonical ids
#: are the ones the protocol and the manifests speak; the raw
#: keys are RoboCasa's and may change without the protocol changing.
ROBOCASA_CAMERA_KEYS = {
    "robot0_agentview_left": "video.robot0_agentview_left",
    "robot0_agentview_right": "video.robot0_agentview_right",
    "robot0_eye_in_hand": "video.robot0_eye_in_hand",
}

#: The proprioceptive vector, in RoboCasa's own order, with the width its gym
#: wrapper publishes for each key (``robocasa/wrappers/gym_wrapper.py`` at
#: a07e365c): end-effector position (3) and rotation as the base-to-eef
#: quaternion (4), base position (3) and quaternion (4), gripper qpos (2).
#: Sixteen in all. A policy that consumes the keys by name, as GR00T does,
#: slices the flat vector with these widths, so they are checked on every
#: observation rather than assumed.
ROBOCASA_STATE_LAYOUT = (
    ("state.end_effector_position_relative", 3),
    ("state.end_effector_rotation_relative", 4),
    ("state.base_position", 3),
    ("state.base_rotation", 4),
    ("state.gripper_qpos", 2),
)
ROBOCASA_STATE_KEYS = tuple(key for key, _ in ROBOCASA_STATE_LAYOUT)
ROBOCASA_STATE_DIM = sum(width for _, width in ROBOCASA_STATE_LAYOUT)

DEFAULT_CONTROL_HZ = 20.0

#: RoboCasa's twelve-dimensional flat action, named into the slots its gym
#: environment accepts. Transcribed from ``robocasa.utils.env_utils`` at
#: revision a07e365c rather than imported, so that the action interface is
#: visible in this repository and testable without the simulator; the server
#: verifies the transcription against the installed RoboCasa at startup.
#:
#: This is the platform's action interface, not a model's output format. A
#: policy emits a flat vector like it does on every other platform, and the
#: naming happens here -- putting it in the policy would make one platform's
#: action dict part of what a submission has to know.
ROBOCASA_ACTION_SLOTS = (
    ("action.end_effector_position", 0, 3),
    ("action.end_effector_rotation", 3, 6),
    ("action.gripper_close", 6, 7),
    ("action.base_motion", 7, 11),
    ("action.control_mode", 11, 12),
)
ROBOCASA_ACTION_DIM = 12


def convert_action(action: Any) -> dict[str, Any]:
    """Name a flat RoboCasa action vector into the gym environment's dict."""
    np = _numpy()
    flat = np.asarray(action)
    if flat.shape != (ROBOCASA_ACTION_DIM,):
        raise ValueError(
            f"RoboCasa expects a flat action of {ROBOCASA_ACTION_DIM} values, "
            f"got shape {flat.shape}"
        )
    flat = flat.copy()
    return {name: flat[start:stop] for name, start, stop in ROBOCASA_ACTION_SLOTS}


def _numpy():
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - runtime dependency
        raise RuntimeError("NumPy is required by RoboCasaEnvironmentAdapter") from exc
    return np


def robocasa_robot_state(observation: Mapping[str, Any]) -> Any:
    """Concatenate RoboCasa's proprioceptive keys in the platform's own order."""
    np = _numpy()
    missing = [key for key in ROBOCASA_STATE_KEYS if key not in observation]
    if missing:
        raise KeyError(f"RoboCasa observation missing robot-state keys: {missing}")
    parts = []
    for key, width in ROBOCASA_STATE_LAYOUT:
        part = np.asarray(observation[key], dtype=np.float32).reshape(-1)
        if part.shape != (width,):
            # A different width means this is not the RoboCasa the layout was
            # transcribed from, and a by-name consumer would slice the wrong
            # numbers into the wrong slots.
            raise ValueError(
                f"RoboCasa publishes {key} with {width} values, got {part.shape}"
            )
        parts.append(part)
    return np.concatenate(parts).astype(np.float32)


class RoboCasaEnvironmentAdapter(EnvironmentAdapter):
    """Adapt one frozen RoboCasa scene to canonical observations."""

    def __init__(
        self,
        environment: Any,
        episode: FrozenEpisode,
        ep_meta: Mapping[str, Any],
        model_xml: str,
        state0: Sequence[float],
        *,
        language: str = "",
        dataset_revision: str,
        official_horizon: int,
        adapter_version: str = "robocasa-adapter-v1",
        control_hz: float = DEFAULT_CONTROL_HZ,
    ) -> None:
        if not dataset_revision:
            raise ValueError("dataset_revision is required")
        if official_horizon < 1:
            raise ValueError("official_horizon must be >= 1")
        if control_hz <= 0:
            raise ValueError("control_hz must be > 0")
        self.environment = environment
        self.episode = episode
        # Deep copies in both directions: RoboCasa mutates the episode metadata
        # it is handed while building the model, which would corrupt the caller's
        # fixture and make the next replicate reject a correct scene.
        self.ep_meta = copy.deepcopy(dict(ep_meta))
        self.model_xml = model_xml
        self.state0 = list(state0)
        self.language = language or str(self.ep_meta.get("lang", ""))
        self.dataset_revision = dataset_revision
        self.official_horizon = int(official_horizon)
        self.adapter_version = adapter_version
        self.control_hz = float(control_hz)
        self._step = 0
        self._observation: Optional[Mapping[str, Any]] = None
        self._success = False
        self._done = False
        self._reward = 0.0
        self._info: Mapping[str, Any] = {}
        self._start_state_sha256: Optional[str] = None
        self._horizon_reached = False
        self._requested_environment_seed: Optional[int] = None

    # -- construction --------------------------------------------------------

    @classmethod
    def from_frozen_scene(
        cls,
        *,
        bank: Any,
        task: str,
        episode_index: int,
        dataset_revision: str,
        split: str = "target",
        seed: int = 0,
        environment: Any = None,
        **kwargs: Any,
    ) -> "RoboCasaEnvironmentAdapter":
        """Construct from the official scene bank without importing RoboCasa here."""
        try:
            import gymnasium as gym
            import robocasa  # noqa: F401
            from robocasa.utils.dataset_registry_utils import get_task_horizon
        except ImportError as exc:  # pragma: no cover - depends on simulator env
            missing = exc.name or "unknown module"
            raise RuntimeError(
                f"RoboCasa runtime dependency is unavailable: {missing}"
            ) from exc
        episode, ep_meta, model_xml, state0 = bank.read(
            ROBOCASA_PLATFORM_ID, task, episode_index)
        if environment is None:
            environment = gym.make(f"robocasa/{task}", split=split, seed=seed)
        kwargs.setdefault("official_horizon", get_task_horizon(task))
        return cls(environment, episode, ep_meta, model_xml, state0,
                   dataset_revision=dataset_revision, **kwargs)

    # -- identity ------------------------------------------------------------

    def dataset_identity(self) -> tuple[str, str]:
        return ROBOCASA_DATASET_ID, self.dataset_revision

    def manifest_identity(self) -> tuple[str, str, int]:
        return self.adapter_version, self.episode.task, self.episode.episode_index

    def camera_inventory(self) -> tuple[str, ...]:
        return tuple(sorted(ROBOCASA_CAMERA_KEYS))

    @property
    def start_state_sha256(self) -> Optional[str]:
        """The state hash the last reset actually landed on."""
        return self._start_state_sha256

    # -- observations --------------------------------------------------------

    def _inner(self) -> Any:
        """The robosuite environment inside the gym wrapper.

        Used for restoring a frozen scene and for reading raw observations --
        never for stepping.
        """
        return self.environment.unwrapped.env

    def _wrapper_observation(self, raw: Mapping[str, Any]) -> Mapping[str, Any]:
        return self.environment.unwrapped.get_observation(raw)

    def _canonical(self, observation: Mapping[str, Any]) -> CanonicalObservation:
        np = _numpy()
        cameras = {}
        for camera_id, raw_key in ROBOCASA_CAMERA_KEYS.items():
            if raw_key not in observation:
                raise KeyError(f"RoboCasa observation missing camera key {raw_key!r}")
            # A copy, not a view: the injector keeps frames by step for freeze
            # and stale, and a wrapper that reuses its render buffer would
            # otherwise rewrite a "pre-onset" frame in place.
            cameras[camera_id] = np.array(observation[raw_key], copy=True, order="C")
        time_ms = self._step * 1000.0 / self.control_hz
        return CanonicalObservation(
            step=self._step,
            cameras=cameras,
            # Every policy camera is present here by construction. Availability
            # is the injector's to change, downstream of this adapter.
            availability={camera: True for camera in cameras},
            robot_state=robocasa_robot_state(observation),
            language=self.language,
            capture_time_ms={camera: time_ms for camera in cameras},
            arrival_time_ms={camera: time_ms for camera in cameras},
            sequence_id={camera: self._step for camera in cameras},
            new_frame={camera: True for camera in cameras},
            source_step={camera: self._step for camera in cameras},
            source_age_steps={camera: 0 for camera in cameras},
        )

    # -- execution -----------------------------------------------------------

    def reset(self, environment_seed: int) -> CanonicalObservation:
        """Replay the frozen scene.

        Landing anywhere but the frozen state raises
        :class:`~mail_bench.scenes.ScenePairingError` from the replay itself: the
        paired design rests on the healthy rollout and every fault arm starting
        from the same state, and a scene that restored elsewhere is not a harder
        scene but a different one.

        ``environment_seed`` is recorded but does not choose the scene: the scene
        is the frozen triple, and a seed that could still move it would defeat
        the point of freezing it. It is carried so that a cell records which seed
        the protocol derived for this unit.
        """
        self._requested_environment_seed = int(environment_seed)
        # The gym-level reset draws the platform's layout too; seeded with the
        # frozen episode's seed so a reused environment draws what a fresh one
        # would, before the frozen state is restored over it.
        self.environment.reset(seed=int(self.episode.episode_seed))
        landed = restore_episode(
            self._inner(), self.episode, self.ep_meta, self.model_xml, self.state0
        )
        # No second check here: restore_episode already refuses to return unless
        # the simulator landed on the frozen state, so a check at this level
        # could never fire and would only suggest the adapter is what enforces
        # the pairing. It raises ScenePairingError, which is left to propagate.
        self._start_state_sha256 = landed
        raw = self._inner()._get_observations(force_update=True)
        observation = self._wrapper_observation(raw)
        self._step = 0
        self._observation = observation
        self._success = False
        self._done = False
        self._horizon_reached = False
        self._reward = 0.0
        self._info = {}
        return self._canonical(observation)

    def step(self, action: Any) -> EnvironmentStep:
        """Execute one action through the gym wrapper, never the inner env.

        The wrapper owns action unmapping and the gripper and base-mode
        thresholding; stepping the inner environment would be a different
        controller, and the benchmark would not be measuring RoboCasa.
        """
        observation, reward, terminated, truncated, info = self.environment.step(
            convert_action(action)
        )
        self._step += 1
        self._observation = observation
        self._reward = float(reward)
        self._info = dict(info or {})
        self._success = bool(self._info.get("success", False))
        # RoboCasa's horizon is per task, not per suite, and the
        # rollout kernel caps on a single figure shared by the whole cohort --
        # necessarily so, because that figure enters the cell identity and a
        # per-task one would split a cohort into per-task measurements. So the
        # task's own official horizon is enforced here, where the per-task
        # knowledge lives. Without this a short-horizon task would be allowed to
        # run to the longest task's budget and report a higher success rate than
        # the official protocol permits.
        #
        # Two facts, not one. Reaching the horizon is what ends the rollout, and
        # a task that succeeds on exactly its last permitted step reaches it
        # while succeeding; calling that a budget exhaustion would say the
        # rollout ran out of time when it finished the task.
        self._horizon_reached = self._step >= self.official_horizon
        self._done = bool(terminated or truncated or self._horizon_reached)
        return EnvironmentStep(
            observation=self._canonical(observation),
            reward=self._reward,
            terminated=self._done or self._success,
            info=self._info,
        )

    def success(self) -> bool:
        return self._success

    def official_metrics(self) -> Mapping[str, Any]:
        return {
            "success": self._success,
            "reward": self._reward,
            "environment_done": self._done,
            "control_steps": self._step,
            "official_horizon": self.official_horizon,
            # Reached the last permitted step, whatever the outcome.
            "horizon_reached": self._horizon_reached,
            # Reached it without solving the task: the budget ran out. A success
            # on exactly the last step is not this.
            "horizon_exhausted": self._horizon_reached and not self._success,
            "task": self.episode.task,
            "episode_index": self.episode.episode_index,
            "scene_identity": self.episode.identity,
            "state0_sha256": self.episode.state0_sha256,
            "start_state_sha256": self._start_state_sha256,
            "requested_environment_seed": self._requested_environment_seed,
        }

    def close(self) -> None:
        close = getattr(self.environment, "close", None)
        if close is None:
            return
        try:
            close()
        except Exception:                                   # noqa: BLE001
            # Closing a simulator that is already gone must not mask the result
            # of the rollout that just finished.
            pass
