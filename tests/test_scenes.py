"""Freezing a scene, replaying it, and refusing anything that is not it.

The simulator is faked -- MuJoCo has its own tests -- but the capture order, the
replay order, the hashes and the gate are the real ones. What these check is
that a paired comparison cannot be produced from two different starting states,
however that difference arose.
"""

import json
import tempfile
import shutil
from pathlib import Path

import numpy as np
import pytest

from mail_bench.scenes import (
    SceneBank,
    SceneBankError,
    ScenePairingError,
    capture_episode,
    restore_episode,
    state_digest,
)


class FakeSimState:
    def __init__(self, values):
        self.values = np.asarray(values, dtype=np.float64)

    def flatten(self):
        return self.values


class FakeSim:
    def __init__(self, xml, state):
        self._xml = xml
        self._state = np.asarray(state, dtype=np.float64)
        self.model = self
        self.calls = []

    def get_xml(self):
        return self._xml

    def get_state(self):
        return FakeSimState(self._state)

    def reset(self):
        self.calls.append("sim.reset")

    def set_state_from_flattened(self, state):
        self.calls.append("set_state")
        self._state = np.asarray(state, dtype=np.float64)

    def forward(self):
        self.calls.append("forward")


class FakeEnv:
    """A stand-in with the replay API, and a scene that resamples on reset."""

    def __init__(self, *, drift_on_reset=False):
        self.ep_meta = {"lang": "open the drawer", "layout_id": 3, "style_id": 4}
        self._settled = np.arange(12, dtype=np.float64) / 7.0
        self.sim = FakeSim("<mujoco><worldbody/></mujoco>", self._settled)
        self.drift_on_reset = drift_on_reset
        self.calls = []
        self.rng = None
        self.robots = []

    def get_ep_meta(self):
        return dict(self.ep_meta)

    def set_ep_meta(self, meta):
        self.calls.append("set_ep_meta")
        self.ep_meta = dict(meta)

    def reset(self):
        self.calls.append("reset")
        if self.drift_on_reset:
            # What an unfrozen environment does: a fresh draw every reset.
            self.sim._state = self.sim._state + 0.5

    def edit_model_xml(self, xml):
        self.calls.append("edit_model_xml")
        return xml

    def reset_from_xml_string(self, xml):
        self.calls.append("reset_from_xml_string")
        self.sim._xml = xml


def frozen(env, **overrides):
    arguments = dict(
        namespace="official", platform="robocasa365", task="OpenDrawer",
        episode_index=0, split="target", episode_seed=7,
        environment_revision="a" * 40,
    )
    arguments.update(overrides)
    return capture_episode(env, **arguments)


def test_a_scene_is_frozen_only_after_it_has_settled():
    env = FakeEnv()
    with pytest.raises(SceneBankError, match="has not settled"):
        frozen(env, settled=False)
    episode, _, _, _ = frozen(env)
    assert episode.captured_after_settling is True


def test_pilot_scenes_cannot_be_written_into_the_official_bank(tmp_path):
    env = FakeEnv()
    with pytest.raises(SceneBankError, match="not one of"):
        frozen(env, namespace="scratch")

    episode, meta, xml, state = frozen(env, namespace="pilot")
    with pytest.raises(SceneBankError, match="namespace"):
        SceneBank(tmp_path, namespace="official").write(episode, meta, xml, state)
    # Its own bank accepts it, and the two banks are different directories.
    SceneBank(tmp_path, namespace="pilot").write(episode, meta, xml, state)
    assert (tmp_path / "pilot").is_dir()
    assert not (tmp_path / "official").exists()


def test_replay_restores_the_frozen_state_in_the_platforms_own_order():
    env = FakeEnv(drift_on_reset=True)
    episode, meta, xml, state = frozen(env)

    # The environment moves on: a later episode has a different scene.
    env.reset()
    assert state_digest(env.sim.get_state().flatten()) != episode.state0_sha256

    digest = restore_episode(env, episode, meta, xml, state)
    assert digest == episode.state0_sha256
    order = [call for call in env.calls if call in
             ("set_ep_meta", "reset", "reset_from_xml_string")]
    assert order[-3:] == ["set_ep_meta", "reset", "reset_from_xml_string"]
    assert env.sim.calls[-3:] == ["sim.reset", "set_state", "forward"]


def test_a_fixture_that_does_not_match_its_hashes_is_refused():
    env = FakeEnv()
    episode, meta, xml, state = frozen(env)

    with pytest.raises(ScenePairingError, match="metadata does not match"):
        restore_episode(env, episode, {**meta, "layout_id": 99}, xml, state)
    with pytest.raises(ScenePairingError, match="XML does not match"):
        restore_episode(env, episode, meta, xml + "<!-- edited -->", state)
    with pytest.raises(ScenePairingError, match="stored state does not match"):
        restore_episode(env, episode, meta, xml, np.asarray(state) + 1e-9)


def test_a_replay_that_lands_somewhere_else_is_caught():
    class Disobedient(FakeEnv):
        def edit_model_xml(self, xml):
            return xml

        def reset_from_xml_string(self, xml):
            super().reset_from_xml_string(xml)
            self.sim._state = self.sim._state + 1.0     # silently resampled

        def _restore(self, state):
            pass

    env = Disobedient()
    episode, meta, xml, state = frozen(env)
    # set_state_from_flattened is what should win; make it a no-op to simulate a
    # platform that ignored the restore.
    env.sim.set_state_from_flattened = lambda values: env.sim.calls.append("ignored")
    with pytest.raises(ScenePairingError, match="landed on state"):
        restore_episode(env, episode, meta, xml, state)



def test_a_bank_round_trip_restores_the_same_scene(tmp_path):
    env = FakeEnv(drift_on_reset=True)
    episode, meta, xml, state = frozen(env)
    bank = SceneBank(tmp_path, namespace="official")
    bank.write(episode, meta, xml, state)

    assert bank.episodes("robocasa365", "OpenDrawer") == (0,)
    restored, ep_meta, xml, state0 = bank.read("robocasa365", "OpenDrawer", 0)
    digest = restore_episode(env, restored, ep_meta, xml, state0)
    assert restored.identity == episode.identity
    assert digest == episode.state0_sha256


def test_a_missing_episode_is_an_error_not_a_fresh_sample(tmp_path):
    bank = SceneBank(tmp_path, namespace="official")
    with pytest.raises(SceneBankError, match="no frozen episode"):
        bank.read("robocasa365", "OpenDrawer", 4)


def test_an_edited_manifest_is_detected(tmp_path):
    env = FakeEnv()
    episode, meta, xml, state = frozen(env)
    bank = SceneBank(tmp_path, namespace="official")
    manifest = bank.write(episode, meta, xml, state)

    document = json.loads(manifest.read_text(encoding="utf-8"))
    document["episode_seed"] = 999            # identity left as it was
    manifest.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(SceneBankError, match="was edited"):
        bank.read("robocasa365", "OpenDrawer", 0)


def test_the_identity_does_not_depend_on_where_it_is_stored():
    env = FakeEnv()
    official, _, _, _ = frozen(env, namespace="official")
    pilot, _, _, _ = frozen(env, namespace="pilot")
    # The same scene is the same scene; the bank it sits in is not part of it.
    assert official.identity == pilot.identity
    assert official.namespace != pilot.namespace


def test_the_module_carries_no_fault_or_model_logic():
    source = (Path(__file__).resolve().parents[1]
              / "src" / "mail_bench" / "scenes.py").read_text(encoding="utf-8")
    import ast

    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef)):
            body = node.body
            if body and isinstance(body[0], ast.Expr) and \
                    isinstance(body[0].value, ast.Constant) and \
                    isinstance(body[0].value.value, str):
                node.body = body[1:] or [ast.Pass()]
    code = ast.unparse(tree)
    for forbidden in ("availability", "fault", "camera", "policy", "randomize"):
        assert forbidden not in code, f"the scene bank carries {forbidden} logic"


def test_a_fixture_survives_an_environment_that_edits_its_own_metadata():
    """RoboCasa rewrites its episode metadata while loading a model.

    A shallow copy would share the nested structures, so the second replay of a
    scene would compare its fixture against something the first replay changed
    and refuse a scene that is in fact correct.
    """
    class Mutating(FakeEnv):
        def set_ep_meta(self, meta):
            super().set_ep_meta(meta)
            # What loading does: reach into the nested structure and edit it.
            self.ep_meta.setdefault("object_cfgs", []).append({"resampled": True})
            self.ep_meta["layout_id"] = 99

    env = Mutating()
    env.ep_meta["object_cfgs"] = [{"name": "drawer"}]
    episode, meta, xml, state = frozen(env)

    for _ in range(3):
        assert restore_episode(env, episode, meta, xml, state) == episode.state0_sha256
    # The caller's copy is untouched however many times the environment edited its own.
    assert meta["object_cfgs"] == [{"name": "drawer"}]
    assert meta["layout_id"] == 3


def test_a_scene_is_never_shared_between_a_diagnostic_bank_and_the_official_one():
    """Namespace and bank id are part of what a unit is; the protocol is not.

    A diagnostic bank and the official one must not hand out the same scene,
    and two different banks must not either. The protocol version is deliberately
    absent: with it, a scene bank would become invalid whenever the scoring rule,
    the fault grid or the benchmark's name changed, and renaming a benchmark is
    not a new set of kitchens.
    """
    from mail_bench.scenes import SCENE_BANK_ID, episode_seed

    def seed(namespace, bank=SCENE_BANK_ID):
        return episode_seed("robocasa365", "OpenDrawer", 0,
                            namespace=namespace, scene_bank_id=bank)

    admission = seed("admission")
    official = seed("official")
    other_bank = seed("official", "mail_robocasa_atomic_unseen_v1")
    assert len({admission, official, other_bank}) == 3

    # Same unit, same answer, every time.
    assert seed("official") == official
    # And different episodes of one task differ.
    assert episode_seed("robocasa365", "OpenDrawer", 1,
                        namespace="official") != official


def test_the_protocol_version_cannot_reach_the_scene_seed():
    """The protocol version cannot reach the scene seed: a scoring-rule or
    name change must not redraw 900 scenes. The seed takes no protocol
    argument.
    """
    import inspect

    from mail_bench.scenes import episode_seed

    parameters = inspect.signature(episode_seed).parameters
    assert "scene_bank_id" in parameters
    assert not any("protocol" in name for name in parameters), sorted(parameters)


def test_a_replay_does_not_depend_on_the_episodes_the_environment_already_ran():
    """The platform's reset draws a layout from the environment's RNG and
    initialises the controllers from that pose; on a reused environment the
    RNG has moved on. Restoring qpos/qvel alone would leave the controllers
    aimed at a pose the healthy rollout never saw, and identical actions would
    diverge within a few steps. The replay seeds the RNG from the frozen episode
    and resets the controllers from the restored state."""
    import numpy as np

    class Controller:
        def __init__(self):
            self.calls = []

        def update_state(self):
            self.calls.append("update_state")

        def reset(self):
            self.calls.append("reset")

    class Robot:
        def __init__(self):
            self.composite_controller = Controller()

    env = FakeEnv()
    env.robots = [Robot(), Robot()]
    env.rng = np.random.default_rng(999)
    env.rng.random()                                     # the RNG has moved on
    episode, ep_meta, xml, state0 = frozen(env)
    restore_episode(env, episode, ep_meta, xml, state0)
    # Seeded from the frozen episode: the first draw is the seed's first draw.
    assert env.rng.random() == np.random.default_rng(episode.episode_seed).random()
    for robot in env.robots:
        assert robot.composite_controller.calls == ["update_state", "reset"]
    # An environment whose replay could depend on its history is refused.
    bare = FakeEnv()
    del bare.rng
    with pytest.raises(ScenePairingError, match="no rng"):
        restore_episode(bare, episode, ep_meta, xml, state0)



def test_a_bank_that_is_not_the_published_one_is_refused(tmp_path):
    """The frozen scenes are what make two runs on two machines one experiment.

    Anyone may build the bank locally, and a local build is not guaranteed to
    reproduce the reference bytes: the simulator's version, its assets and its
    sampler all feed the settled initial state. So an official run checks the
    bank it was given against the published identity, and a bank that differs
    by one scene is refused rather than scored.
    """
    import json

    from mail_bench.scenes import SCENE_BANK_ID, SceneBank, SceneBankError

    bank = SceneBank(tmp_path, namespace="official")
    env = FakeEnv()
    episode, meta, xml, state = frozen(env)
    bank.write(episode, meta, xml, state)
    written = {f"{episode.task}#{episode.episode_index:04d}": episode.identity}
    identity = {"scene_bank_id": SCENE_BANK_ID, "namespace": "official",
                "platform": episode.platform, "scene_identities": dict(written)}

    bank.verify_against(identity)                       # the bank it describes

    other = dict(identity, scene_bank_id="some_other_bank_v1")
    with pytest.raises(SceneBankError, match="published bank is"):
        bank.verify_against(other)

    changed = json.loads(json.dumps(identity))
    key = sorted(written)[0]
    changed["scene_identities"][key] = "0" * 64
    with pytest.raises(SceneBankError, match="not the published one"):
        bank.verify_against(changed)

    extra = json.loads(json.dumps(identity))
    extra["scene_identities"][f"{episode.task}#0099"] = "1" * 64
    with pytest.raises(SceneBankError, match="could not be read"):
        bank.verify_against(extra)


def test_a_scene_whose_payload_bytes_were_altered_is_refused_before_any_rollout(tmp_path):
    """The identity record alone would pass: it is recomputed from its own
    fields. The payload files are hashed against that record, so a bank edited
    after freezing is refused at the preflight, not when the scene is restored."""
    import json

    import numpy as np

    from mail_bench.scenes import SCENE_BANK_ID, SceneBank, SceneBankError

    env = FakeEnv()
    episode, meta, xml, state = frozen(env)
    identity = {"scene_bank_id": SCENE_BANK_ID, "namespace": "official",
                "platform": episode.platform,
                "scene_identities": {f"{episode.task}#{episode.episode_index:04d}": episode.identity}}
    for payload, tamper in (
        ("model.xml", lambda p: p.write_text(p.read_text(encoding="utf-8") + " ", encoding="utf-8")),
        ("ep_meta.json", lambda p: p.write_text(json.dumps(dict(json.loads(p.read_text(encoding="utf-8")), extra=1)), encoding="utf-8")),
        ("state0.npy", lambda p: np.save(p, np.load(p) + 1.0)),
    ):
        root = tmp_path / payload.replace(".", "_")
        bank = SceneBank(root, namespace="official")
        bank.write(episode, meta, xml, state)
        bank.verify_against(identity)
        tamper(next(root.rglob(payload)))
        with pytest.raises(SceneBankError, match="payload bytes"):
            bank.verify_against(identity)


def test_the_shipped_bank_archive_is_the_published_one():
    """The bank an outsider gets must be the bank the identity describes.

    An official run is held to the published identity and the only way to
    obtain that bank is this archive, so the two have to agree or nobody
    outside this repository can produce an official score at all.
    """
    import hashlib
    import subprocess
    import tarfile

    root = Path(__file__).resolve().parents[1]
    archive = root / "scene_bank" / "mail_bench_scene_bank_official.tar.zst"
    assert archive.is_file(), "the frozen bank does not ship with the repository"

    recorded = (root / "scene_bank" / "SHA256SUMS").read_text().split()[0]
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    assert digest == recorded, "the archive does not match its own checksum"

    if not shutil.which("zstd"):
        pytest.skip("zstd is not installed; the checksum was still checked")
    with tempfile.TemporaryDirectory() as tmp:
        raw = subprocess.run(["zstd", "-dq", "-c", str(archive)], check=True,
                             capture_output=True).stdout
        tar_path = Path(tmp) / "bank.tar"
        tar_path.write_bytes(raw)
        with tarfile.open(tar_path) as tar:
            tar.extractall(tmp, filter="data")
        identity = json.loads(
            (root / "configs" / "scene_bank_identity.json").read_text(encoding="utf-8"))
        SceneBank(tmp, namespace="official").verify_against(identity)
