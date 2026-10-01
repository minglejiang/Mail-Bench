"""RoboCasa through the shared driver, from the command line to standard cells.

The simulator and the checkpoint are faked -- MuJoCo, RoboCasa and openpi have
their own tests -- but the entry point, the frozen-scene replay, the adapter,
the socket contract, the driver, the fault injection, the cell writer and the
aggregator are all the real ones.

The point is that RoboCasa produces standard cells from the shared kernel,
rather than a bespoke JSON shape that would have to be converted later.
The driver path exchanges RoboCasa's own observation keys and
`convert_action`-named actions.
"""

import importlib.util
import json
import socket
import threading
from pathlib import Path

import numpy as np
import pytest

yaml = pytest.importorskip("yaml")

from mail_bench.aggregate import mail_bench_report
from mail_bench.net import recv_message, send_message
from mail_bench.platforms.robocasa import ROBOCASA_CAMERA_KEYS, convert_action
from mail_bench.scenes import (
    SCENE_BANK_ID,
    SCENE_SCHEMA_VERSION,
    FrozenEpisode,
    SceneBank,
    _sha256_bytes,
    semantic_sha256,
    state_digest,
)

ROOT = Path(__file__).resolve().parents[1]
CAMERAS = tuple(sorted(ROBOCASA_CAMERA_KEYS))
TASKS = ("PnPCounterToCab", "OpenSingleDoor")
MODEL_ID = "somebody/robocasa-policy"
STATE = [float(i) * 0.25 for i in range(24)]
EP_META = {"lang": "put the bowl in the cabinet", "layout_id": 3}
MODEL_XML = "<mujoco/>"


def load_script():
    spec = importlib.util.spec_from_file_location(
        "run_robocasa_cohort", ROOT / "scripts" / "run_robocasa_cohort.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# --------------------------------------------------------------------------
# A third-party policy server speaking mcr-policy-v1
# --------------------------------------------------------------------------
class Server:
    def __init__(self):
        self.acts = []
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.bind(("127.0.0.1", 0))
        self.socket.listen(4)
        self.port = self.socket.getsockname()[1]
        threading.Thread(target=self._serve, daemon=True).start()

    @property
    def identity(self):
        return {
            "ok": True, "protocol": "mcr-policy-v1", "model_id": MODEL_ID,
            "checkpoint_sha256": "c" * 64, "policy_visible_cameras": list(CAMERAS),
            "availability_consumed_by_policy": False, "predicted_action_chunk": 4,
            "native_execution_horizon": 2, "action_dimension": 12,
            "stateful_policy": False, "reset_semantics": "stateless",
            "training_regime": "camera dropout p=0.15 during finetuning",
            "fault_augmentation_used": True,
        }

    def _serve(self):
        while True:
            try:
                connection, _ = self.socket.accept()
            except OSError:
                return
            with connection:
                while True:
                    try:
                        message = recv_message(connection)
                    except (ConnectionError, OSError):
                        break
                    kind = message.get("type")
                    if kind == "health":
                        send_message(connection, {"ok": True, "ready": True, "status": "ready"})
                    elif kind == "identity":
                        send_message(connection, self.identity)
                    elif kind == "reset":
                        send_message(connection, {"ok": True})
                    elif kind == "act":
                        self.acts.append(message)
                        send_message(connection, {
                            "ok": True,
                            "actions": np.full((4, 12), 0.5, np.float32)})
                    elif kind == "disconnect":
                        send_message(connection, {"ok": True})
                        break
                    else:
                        send_message(connection, {"ok": False, "error": "unknown"})

    def close(self):
        self.socket.close()


# --------------------------------------------------------------------------
# A fake RoboCasa: enough of the replay path and the gym wrapper
# --------------------------------------------------------------------------
def raw_observation(step, state):
    observation = {
        raw: np.full((4, 4, 3), (step + index) % 251, np.uint8)
        for index, raw in enumerate(sorted(ROBOCASA_CAMERA_KEYS.values()))
    }
    observation.update({
        "state.end_effector_position_relative": np.array(state[:3], np.float32),
        # Widths as RoboCasa publishes them: quaternion rotations, two gripper joints.
        "state.end_effector_rotation_relative": np.array(state[3:7], np.float32),
        "state.base_position": np.array(state[7:10], np.float32),
        "state.base_rotation": np.array(state[10:14], np.float32),
        "state.gripper_qpos": np.array(state[14:16], np.float32),
    })
    return observation


class FakeSim:
    def __init__(self, state):
        self.state = np.asarray(state, np.float64)

    def reset(self):
        self.state = np.zeros_like(self.state)

    def set_state_from_flattened(self, state):
        self.state = np.asarray(state, np.float64).copy()

    def forward(self):
        return None

    def get_state(self):
        return self

    def flatten(self):
        return self.state


class FakeInner:
    def __init__(self):
        self.rng = None
        self.sim = FakeSim(STATE)
        self.ep_meta = None

    def set_ep_meta(self, ep_meta):
        self.ep_meta = ep_meta

    def get_ep_meta(self):
        return self.ep_meta

    def reset(self):
        return None

    def edit_model_xml(self, xml):
        return xml

    def reset_from_xml_string(self, xml):
        return None

    def _get_observations(self, force_update=False):
        return raw_observation(0, self.sim.state)


class FakeGym:
    """Succeeds on the fourth action, so a healthy rollout has a completion."""

    def __init__(self):
        self.env = FakeInner()
        self.unwrapped = self
        self.step_index = 0
        self.actions = []

    def get_observation(self, raw):
        return raw

    def reset(self, seed=None):
        self.step_index = 0
        self.actions = []

    def step(self, action):
        assert isinstance(action, dict), "the wrapper receives RoboCasa's action dict"
        self.step_index += 1
        self.actions.append(action)
        success = self.step_index >= 4
        return (raw_observation(self.step_index, self.env.sim.state), 0.0, success,
                False, {"success": success})

    def close(self):
        return None


def build_bank(root):
    bank = SceneBank(root, namespace="official")
    for task in TASKS:
        for index in range(2):
            state = [v + index for v in STATE]
            episode = FrozenEpisode(
                schema_version=SCENE_SCHEMA_VERSION, scene_bank_id=SCENE_BANK_ID,
        namespace="official", platform="robocasa365",
                task=task, episode_index=index, split="target", episode_seed=100 + index,
                environment_revision="a07e365c", asset_inventory_sha256="f" * 64,
                ep_meta_sha256=semantic_sha256(EP_META),
                model_xml_sha256=_sha256_bytes(MODEL_XML.encode("utf-8")),
                state0_sha256=state_digest(state), state0_size=len(state),
                captured_after_settling=True,
            )
            bank.write(episode, EP_META, MODEL_XML, state)
    return bank


@pytest.fixture
def cohort(tmp_path, monkeypatch):
    script = load_script()
    server = Server()
    build_bank(tmp_path / "bank")

    import mail_bench.cohort as cohort_module
    import mail_bench.platforms.robocasa as robocasa_module

    monkeypatch.setattr(cohort_module, "require_clean_worktree", lambda root: "0" * 40)

    def fake_from_frozen(cls, *, bank, task, episode_index, dataset_revision,
                         split="target", seed=0, environment=None, **kwargs):
        episode, ep_meta, model_xml, state0 = bank.read("robocasa365", task, episode_index)
        kwargs.setdefault("official_horizon", 40)
        return cls(FakeGym(), episode, ep_meta, model_xml, state0,
                   dataset_revision=dataset_revision, **kwargs)

    monkeypatch.setattr(
        robocasa_module.RoboCasaEnvironmentAdapter, "from_frozen_scene",
        classmethod(fake_from_frozen),
    )
    # RoboCasa's own registry, faked: task list and per-task horizons only.
    registry = type(sys)("robocasa.utils.dataset_registry")
    registry.TARGET_TASKS = {"atomic_seen": list(TASKS)}
    utils = type(sys)("robocasa.utils.dataset_registry_utils")
    utils.get_task_horizon = lambda task: 40
    monkeypatch.setitem(sys.modules, "robocasa.utils.dataset_registry", registry)
    monkeypatch.setitem(sys.modules, "robocasa.utils.dataset_registry_utils", utils)

    argv = [
        "--scene-bank", str(tmp_path / "bank"),
        "--policy", "custom", "--policy-port", str(server.port),
        "--onsets", "0.45",
        "--output-dir", str(tmp_path / "out"),
        "--phase", "all",
        "--healthy-replicates", "1",
    ]
    monkeypatch.setattr("sys.argv", ["run_robocasa_cohort.py", *argv])
    try:
        script.main()
        yield {"server": server, "out": tmp_path / "out", "bank": tmp_path / "bank"}
    finally:
        server.close()


import sys  # noqa: E402  (used by the fixture's module fakes)


def test_robocasa_produces_standard_cells_through_the_shared_driver(cohort):
    out = cohort["out"]
    certificate = json.loads(
        (out / "certificate_all_worker0.json").read_text(encoding="utf-8")
    )
    assert certificate["policy_declaration"]["model_id"] == MODEL_ID
    # The regime is carried and shown, not judged.
    assert certificate["policy_declaration"]["training_regime"] == (
        "camera dropout p=0.15 during finetuning"
    )
    assert certificate["result_valid"] is True

    # The official scorer in its neutral shape: two tasks of two scenes at one
    # onset is an experiment, not the suite, and it says so.
    report = mail_bench_report(out / "cells", expected_tasks=TASKS, scenes_per_task=2,
                               require_official=False)
    assert {entry["task"] for entry in report["tasks"]} == set(TASKS)
    assert all(entry["scenes"] == 2 for entry in report["tasks"])   # one canonical rollout each
    assert report["official"] is False and report["mail_bench_score"] is None
    assert report["audit_failures"] == []


def test_every_cell_records_the_frozen_scene_it_replayed(cohort):
    cells = list((cohort["out"] / "cells").glob("*.json"))
    assert cells
    for path in cells:
        result = json.loads(path.read_text(encoding="utf-8"))["result"]
        # The identity a reader needs in order to know which frozen scene
        # produced this number, carried by the adapter's official metrics.
        metrics = result["official_metrics"]
        assert metrics["task"] in TASKS
        assert metrics["episode_index"] in (0, 1)
        assert metrics["scene_identity"]
        # The replay landed on the frozen state, and the cell says so rather
        # than leaving a reader to trust that it did.
        assert metrics["start_state_sha256"] == metrics["state0_sha256"]
        assert result["onset_basis"] in ("healthy_reference", "not_applicable")
        # There is no fallback; a fault cell exists only where a healthy one
        # succeeded, so no cell may claim one.
        assert result["onset_fallback"] is False
        if result["clean_or_fault"] == "fault":
            assert result["onset_basis"] == "healthy_reference"
            assert result["healthy_reference_success"] is True


def test_the_injected_fault_reaches_the_policy_as_an_absent_camera(cohort):
    acts = cohort["server"].acts
    missing = [a for a in acts if not all(a["availability"].values())]
    assert missing, "no act carried an unavailable camera"
    for act in missing:
        for camera, available in act["availability"].items():
            if not available:
                # Nothing between the injector and the server filled it in.
                assert act["cameras"][camera] is None
    healthy = [a for a in acts if all(a["availability"].values())]
    assert healthy
    for act in healthy:
        assert set(act["cameras"]) == set(CAMERAS)
        assert all(act["cameras"][camera] is not None for camera in CAMERAS)


def test_the_driver_exchanges_robocasas_own_observation_and_action_keys(cohort):
    """The driver path exchanges RoboCasa's own observation keys and
    ``convert_action``-named actions: the policy element is built from the
    wrapper's own observation keys and the gym environment is stepped with
    ``convert_action`` applied to a flat vector.
    """
    act = cohort["server"].acts[0]
    # Observation: the three canonical cameras and RoboCasa's own 16-D proprio.
    assert sorted(act["cameras"]) == sorted(CAMERAS)
    state = np.asarray(act["robot_state"])
    assert state.shape == (16,)
    expected = raw_observation(0, STATE)
    np.testing.assert_allclose(
        state[:3], expected["state.end_effector_position_relative"])
    np.testing.assert_allclose(state[14:16], expected["state.gripper_qpos"])
    assert act["language"] == EP_META["lang"]

    # Action: a flat twelve-vector leaves the policy and the platform names the
    # slots, exactly as the validated path did.
    named = convert_action(np.full(12, 0.5, np.float32))
    assert set(named) == {
        "action.end_effector_position", "action.end_effector_rotation",
        "action.gripper_close", "action.base_motion", "action.control_mode",
    }
    np.testing.assert_allclose(named["action.base_motion"], np.full(4, 0.5))


def test_the_frozen_scene_is_never_mutated_by_a_rollout(cohort):
    # Four cells replayed each unit; the bank on disk must be the bank that was
    # written, or later cells would be pairing against a changed scene.
    bank = SceneBank(cohort["bank"], namespace="official")
    for task in TASKS:
        for index in range(2):
            episode, ep_meta, model_xml, state0 = bank.read("robocasa365", task, index)
            assert semantic_sha256(ep_meta) == episode.ep_meta_sha256
            assert state_digest(state0) == episode.state0_sha256
            assert ep_meta == EP_META
            assert model_xml == MODEL_XML
