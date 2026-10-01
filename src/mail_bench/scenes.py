"""Frozen episodes, and the rule that every arm may only replay one.

Some platforms hand out a fixed list of initial states; others sample the whole
scene when the environment resets. RoboCasa365 is the second kind: layout,
style, textures, fixture placement, object identity, object pose and the robot's
base offset all come out of one seeded generator, and the loader retries
placement on failure, so the number of draws depends on what was drawn before.
Hoping that the same seed reproduces the same kitchen is therefore a fragile
basis for a paired comparison -- one extra draw anywhere and the two arms are in
different rooms.

So the scene is frozen once, into a fixture, and every arm afterwards replays
it. Pairing is then a property of what was loaded rather than a hope about a
generator:

    same model XML + same settled state + same exogenous and policy seeds

This module is deliberately thin. It freezes an episode, restores one, and says
whether two restorations agree. Fault injection, camera handling and anything
about a model live elsewhere.
"""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from .manifest import canonical_json, semantic_sha256

PathLike = Any

#: Schema 2: the frozen record names its scene_bank_id.
SCENE_SCHEMA_VERSION = 2

#: Banks are namespaced so that a scene from a diagnostic bank is never
#: resumed into an official run.
NAMESPACES = ("official", "admission", "pilot")


class ScenePairingError(RuntimeError):
    """Two arms did not start from the same state. No result may be produced."""


class SceneBankError(RuntimeError):
    """The bank cannot supply the episode that was asked for."""


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def state_digest(state: Sequence[float]) -> str:
    """Hash a flattened simulator state.

    Rounding is deliberate and fixed: MuJoCo writes float64, and two restores of
    the same bytes are bit-identical, so no tolerance is needed. Anything that
    differs at all is a different starting condition and has to be caught.
    """
    import numpy as np

    array = np.asarray(state, dtype=np.float64)
    return _sha256_bytes(array.tobytes(order="C"))


@dataclass(frozen=True)
class FrozenEpisode:
    """One (task, episode_index) pinned down to bytes."""

    schema_version: int
    #: Which bank of scenes this belongs to. Part of the identity, because two
    #: banks may legitimately hold a scene for the same (task, episode_index).
    scene_bank_id: str
    namespace: str
    platform: str
    task: str
    episode_index: int
    split: str
    episode_seed: int
    environment_revision: str
    ep_meta_sha256: str
    model_xml_sha256: str
    state0_sha256: str
    state0_size: int
    captured_after_settling: bool
    asset_inventory_sha256: Optional[str] = None

    @property
    def identity(self) -> str:
        """What names this episode, independent of where it is stored."""
        return semantic_sha256({
            "schema_version": self.schema_version,
            "scene_bank_id": self.scene_bank_id,
            "platform": self.platform,
            "task": self.task,
            "episode_index": self.episode_index,
            "split": self.split,
            "episode_seed": self.episode_seed,
            "environment_revision": self.environment_revision,
            "ep_meta_sha256": self.ep_meta_sha256,
            "model_xml_sha256": self.model_xml_sha256,
            "state0_sha256": self.state0_sha256,
        })

    def as_dict(self) -> dict[str, Any]:
        document = {field: getattr(self, field) for field in (
            "schema_version", "scene_bank_id", "namespace", "platform",
            "task", "episode_index",
            "split", "episode_seed", "environment_revision", "ep_meta_sha256",
            "model_xml_sha256", "state0_sha256", "state0_size",
            "captured_after_settling", "asset_inventory_sha256",
        )}
        document["identity"] = self.identity
        return document


#: What a bank of frozen scenes is. It names the suite, the task list and the
#: number of episodes per task -- everything that decides which scenes exist --
#: and nothing else.
SCENE_BANK_ID = "mail_robocasa_atomic_seen_v1"


def episode_seed(platform: str, task: str, episode_index: int, *, namespace: str,
                 scene_bank_id: str = SCENE_BANK_ID) -> int:
    """The seed a unit is entitled to, derived from what the unit is.

    The derivation deliberately does **not** include the protocol version.
    Including it would invalidate a bank whenever the scoring rule, the fault
    grid or the benchmark's name changed, for a change that cannot affect
    which scenes exist. Renaming a benchmark is not a new set of kitchens.

    What decides which scenes exist is the bank: the suite, its task list, how
    many episodes each task has. That is ``scene_bank_id``, and a bank drawn
    under one is never confused with a bank drawn under another. ``namespace``
    keeps a diagnostic bank and the official one apart within the same bank id.

    Deriving per episode rather than taking the k-th draw of a single seeded run
    keeps the recorded seed a genuine answer to "which scene is this": the
    fixture is the first reset of a fresh environment under it.
    """
    digest = semantic_sha256({
        "scene_bank_id": scene_bank_id,
        "namespace": namespace,
        "platform": platform,
        "task": task,
        "episode": int(episode_index),
    })
    return int(digest[:8], 16)


def capture_episode(
    environment: Any,
    *,
    namespace: str,
    platform: str,
    task: str,
    episode_index: int,
    split: str,
    episode_seed: int,
    environment_revision: str,
    scene_bank_id: str = SCENE_BANK_ID,
    asset_inventory_sha256: Optional[str] = None,
    settled: bool = True,
) -> tuple[FrozenEpisode, Mapping[str, Any], str, Any]:
    """Freeze a scene that has already been reset and allowed to settle.

    ``settled`` is not a convenience flag to be passed blindly: the state has to
    be read after the environment's own settling steps, or the fixture stores a
    scene mid-fall and every replay starts from an object that has not landed.
    """
    import numpy as np

    if namespace not in NAMESPACES:
        raise SceneBankError(
            f"namespace {namespace!r} is not one of {list(NAMESPACES)}; official "
            "cohorts and pilot runs may not share a bank"
        )
    if not settled:
        raise SceneBankError(
            "refusing to freeze a scene that has not settled: capture state0 after "
            "reset and the environment's settling steps, never before"
        )

    # A deep copy, because the environment keeps mutating its own episode
    # metadata: handing back a structure that shares nested objects with the live
    # one means the fixture changes underneath whoever holds it.
    ep_meta = copy.deepcopy(environment.get_ep_meta())
    model_xml = environment.sim.model.get_xml()
    state0 = np.asarray(environment.sim.get_state().flatten(), dtype=np.float64)

    episode = FrozenEpisode(
        schema_version=SCENE_SCHEMA_VERSION,
        scene_bank_id=scene_bank_id,
        namespace=namespace,
        platform=platform,
        task=task,
        episode_index=int(episode_index),
        split=split,
        episode_seed=int(episode_seed),
        environment_revision=environment_revision,
        ep_meta_sha256=semantic_sha256(ep_meta),
        model_xml_sha256=_sha256_bytes(model_xml.encode("utf-8")),
        state0_sha256=state_digest(state0),
        state0_size=int(state0.size),
        captured_after_settling=True,
        asset_inventory_sha256=asset_inventory_sha256,
    )
    return episode, ep_meta, model_xml, state0


def restore_episode(environment: Any, episode: FrozenEpisode,
                    ep_meta: Mapping[str, Any], model_xml: str,
                    state0: Sequence[float]) -> str:
    """Put the environment back into the frozen scene, and prove it landed there.

    The order is the platform's own replay path, not an invention: episode
    metadata first, then a reset so the model is rebuilt, then the recorded XML,
    then the recorded state. Restoring the XML sets the deterministic-reset flag
    that stops the loader from resampling object placements.
    """
    import numpy as np

    if semantic_sha256(ep_meta) != episode.ep_meta_sha256:
        raise ScenePairingError(
            f"episode metadata does not match the fixture for {episode.task} "
            f"#{episode.episode_index}"
        )
    if _sha256_bytes(model_xml.encode("utf-8")) != episode.model_xml_sha256:
        raise ScenePairingError(
            f"model XML does not match the fixture for {episode.task} "
            f"#{episode.episode_index}"
        )
    state = np.asarray(state0, dtype=np.float64)
    if state_digest(state) != episode.state0_sha256:
        raise ScenePairingError(
            f"stored state does not match the fixture for {episode.task} "
            f"#{episode.episode_index}"
        )

    # Deep, not shallow: a shallow copy shares nested dicts and lists with the
    # caller's fixture, and the environment edits them while loading the model.
    environment.set_ep_meta(copy.deepcopy(dict(ep_meta)))
    # A replay may not depend on how many episodes this environment has run.
    # The platform's reset draws a layout from the environment's own RNG and
    # initialises the robot's controllers from that pose; on a reused
    # environment the RNG has moved on, the pre-restore pose differs, and
    # restoring qpos/qvel alone would leave the controllers aimed at a pose
    # the healthy rollout never saw, and identical actions would diverge
    # within a few steps. The RNG is put back to the frozen episode's seed
    # first.
    if not hasattr(environment, "rng"):
        raise ScenePairingError(
            "the environment has no rng to reset; a replay on it would depend on "
            "the episodes it has already run"
        )
    # The same generator the platform's own seeded reset builds (PCG64).
    environment.rng = np.random.Generator(np.random.PCG64(int(episode.episode_seed)))
    environment.reset()
    environment.reset_from_xml_string(environment.edit_model_xml(model_xml))
    environment.sim.reset()
    environment.sim.set_state_from_flattened(state)
    environment.sim.forward()
    # And the controllers take their reference from the restored state, not
    # from whatever pose the reset above happened to sample.
    for robot in getattr(environment, "robots", ()):
        controller = getattr(robot, "composite_controller", None)
        if controller is not None:
            controller.update_state()
            controller.reset()

    observed = state_digest(environment.sim.get_state().flatten())
    if observed != episode.state0_sha256:
        raise ScenePairingError(
            f"replay of {episode.task} #{episode.episode_index} landed on state "
            f"{observed[:12]}, not the frozen {episode.state0_sha256[:12]}"
        )
    return observed


class SceneBank:
    """Frozen episodes on disk, one directory per namespace."""

    def __init__(self, root: PathLike, *, namespace: str) -> None:
        if namespace not in NAMESPACES:
            raise SceneBankError(f"namespace {namespace!r} is not one of {list(NAMESPACES)}")
        self.namespace = namespace
        self.root = Path(root) / namespace

    def _paths(self, episode: FrozenEpisode) -> tuple[Path, Path, Path, Path]:
        directory = self.root / episode.platform / episode.task / f"{episode.episode_index:04d}"
        return (directory / "episode.json", directory / "ep_meta.json",
                directory / "model.xml", directory / "state0.npy")

    def write(self, episode: FrozenEpisode, ep_meta: Mapping[str, Any],
              model_xml: str, state0: Sequence[float]) -> Path:
        import numpy as np

        if episode.namespace != self.namespace:
            raise SceneBankError(
                f"episode belongs to namespace {episode.namespace!r}, not {self.namespace!r}"
            )
        manifest, meta_path, xml_path, state_path = self._paths(episode)
        manifest.parent.mkdir(parents=True, exist_ok=True)
        meta_path.write_text(canonical_json(dict(ep_meta)), encoding="utf-8")
        xml_path.write_text(model_xml, encoding="utf-8")
        np.save(state_path, np.asarray(state0, dtype=np.float64))
        manifest.write_text(canonical_json(episode.as_dict()), encoding="utf-8")
        return manifest

    def read(self, platform: str, task: str, episode_index: int
             ) -> tuple[FrozenEpisode, Mapping[str, Any], str, Any]:
        import numpy as np

        directory = self.root / platform / task / f"{episode_index:04d}"
        manifest = directory / "episode.json"
        if not manifest.is_file():
            raise SceneBankError(
                f"no frozen episode for {platform}/{task} #{episode_index} in the "
                f"{self.namespace!r} bank at {self.root}"
            )
        document = json.loads(manifest.read_text(encoding="utf-8"))
        stored_identity = document.pop("identity", None)
        episode = FrozenEpisode(**document)
        if stored_identity is not None and stored_identity != episode.identity:
            raise SceneBankError(
                f"the manifest for {task} #{episode_index} was edited: it records "
                f"identity {stored_identity[:12]} but hashes to {episode.identity[:12]}"
            )
        # A fixture from another bank, namespace or slot that was copied into
        # this directory would otherwise be replayed as if it were this scene.
        expected = {
            "scene_bank_id": SCENE_BANK_ID, "namespace": self.namespace,
            "platform": platform, "task": task, "episode_index": int(episode_index),
        }
        actual = {key: getattr(episode, key) for key in expected}
        if actual != expected:
            raise SceneBankError(
                f"the fixture at {directory} belongs to {actual}, not to {expected}"
            )
        ep_meta = json.loads((directory / "ep_meta.json").read_text(encoding="utf-8"))
        model_xml = (directory / "model.xml").read_text(encoding="utf-8")
        state0 = np.load(directory / "state0.npy")
        return episode, ep_meta, model_xml, state0

    def verify_against(self, identity: Mapping[str, Any]) -> None:
        """Refuse unless this bank is scene-for-scene the published one.

        A frozen scene bank is what makes two runs on two machines the same
        experiment. The bank ships with the repository; a local rebuild is not
        guaranteed to reproduce the reference bytes: the
        simulator's version, its assets and its sampler all feed the settled
        initial state. So a run that wants to carry the benchmark's name checks
        its bank against the published identity, scene by scene, rather than
        reporting a score for a different nine hundred scenes.

        Every scene's identity record is recomputed from its stored fields, and
        the three payload files (``ep_meta.json``, ``model.xml``, ``state0.npy``)
        are hashed against the digests that record carries, so a bank whose
        bytes were altered after freezing is refused here rather than hours
        later when the altered scene is restored.
        """
        import numpy as np

        if identity.get("scene_bank_id") not in (None, SCENE_BANK_ID):
            raise SceneBankError(
                f"the published bank is {identity['scene_bank_id']!r} and this build is "
                f"{SCENE_BANK_ID!r}"
            )
        if identity.get("namespace") not in (None, self.namespace):
            raise SceneBankError(
                f"the published bank is the {identity['namespace']!r} namespace and this "
                f"is {self.namespace!r}"
            )
        platform = str(identity.get("platform"))
        published = identity.get("scene_identities") or {}
        if not published:
            raise SceneBankError("the published identity lists no scenes")
        missing: list[str] = []
        mismatched: list[str] = []
        corrupt: list[str] = []
        for key, expected in sorted(published.items()):
            task, _, index = key.rpartition("#")
            try:
                episode, ep_meta, model_xml, state0 = self.read(platform, task, int(index))
            except Exception as exc:  # unreadable is as disqualifying as different
                missing.append(f"{key} ({type(exc).__name__})")
                continue
            if episode.identity != expected:
                mismatched.append(key)
            elif (semantic_sha256(ep_meta) != episode.ep_meta_sha256
                  or _sha256_bytes(model_xml.encode("utf-8")) != episode.model_xml_sha256
                  or state_digest(np.asarray(state0, dtype=np.float64)) != episode.state0_sha256):
                corrupt.append(key)
        if missing or mismatched or corrupt:
            raise SceneBankError(
                f"this scene bank is not the published one: {len(missing)} of "
                f"{len(published)} scenes could not be read (e.g. {missing[:3]}), "
                f"{len(mismatched)} differ (e.g. {mismatched[:3]}), {len(corrupt)} carry "
                f"payload bytes that do not match their own record (e.g. {corrupt[:3]}). "
                "A local build does not "
                "always reproduce the reference bytes, and a run on a different bank is an "
                "experiment on different scenes rather than this benchmark."
            )

    def episodes(self, platform: str, task: str) -> tuple[int, ...]:
        directory = self.root / platform / task
        if not directory.is_dir():
            return ()
        return tuple(sorted(int(child.name) for child in directory.iterdir()
                            if child.is_dir() and child.name.isdigit()))


__all__ = [
    "NAMESPACES",
    "SCENE_BANK_ID",
    "SCENE_SCHEMA_VERSION",
    "FrozenEpisode",
    "SceneBank",
    "SceneBankError",
    "ScenePairingError",
    "capture_episode",
    "episode_seed",
    "restore_episode",
    "state_digest",
]
