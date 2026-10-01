"""The public contract, exercised against a fake third-party server.

Nothing here imports a reference model: this is what an outside submission sees.
"""

import socket
import threading

import numpy as np
import pytest

from mail_bench.interfaces import CanonicalObservation
from mail_bench.net import recv_message, send_message
from mail_bench.policy_api import GenericSocketPolicyAdapter, PolicyConnection
from mail_bench.policy_api.contract import ContractError


CAMERAS = ("agentview", "eye_in_hand")


def declaration(**overrides):
    base = {
        "ok": True,
        "protocol": "mcr-policy-v1",
        "model_id": "somebody/router-vla",
        "checkpoint_sha256": "a" * 64,
        "policy_visible_cameras": list(CAMERAS),
        "availability_consumed_by_policy": True,
        "predicted_action_chunk": 4,
        "native_execution_horizon": 2,
        "action_dimension": 7,
        "stateful_policy": True,
        "reset_semantics": "clears_all_state",
        "training_regime": "clean_trained",
    }
    base.update(overrides)
    return base


class FakeServer:
    """A minimal third-party policy server: health, identity, reset, act."""

    def __init__(self, *, ready=True, identity=None, actions=None):
        self.ready = ready
        self.identity = identity if identity is not None else declaration()
        self.actions = actions if actions is not None else np.ones((4, 7), np.float32)
        self.received = []
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.bind(("127.0.0.1", 0))
        self.socket.listen(1)
        self.port = self.socket.getsockname()[1]
        self.thread = threading.Thread(target=self._serve, daemon=True)
        self.thread.start()

    def _serve(self):
        try:
            connection, _ = self.socket.accept()
        except OSError:
            return
        with connection:
            while True:
                try:
                    message = recv_message(connection)
                except (ConnectionError, OSError):
                    return
                self.received.append(message)
                kind = message.get("type")
                if kind == "health":
                    send_message(connection, {
                        "ok": True, "ready": self.ready,
                        "status": "ready" if self.ready else "loading",
                    })
                elif kind == "identity":
                    send_message(connection, self.identity)
                elif kind == "reset":
                    send_message(connection, {"ok": True})
                elif kind in ("invalidate", "visual_memory_reset"):
                    send_message(connection, {"ok": True})
                elif kind == "act":
                    send_message(connection, {"ok": True, "actions": self.actions})
                elif kind == "disconnect":
                    send_message(connection, {"ok": True})
                    return
                else:
                    send_message(connection, {"ok": False, "error": f"unknown {kind!r}"})

    def close(self):
        self.socket.close()


def adapter_for(server, **kwargs):
    return GenericSocketPolicyAdapter(
        port=server.port, timeout_seconds=10.0, environment_cameras=CAMERAS, **kwargs
    )


def observation(*, agentview_available=True):
    return CanonicalObservation(
        step=3,
        cameras={
            "agentview": np.full((4, 4, 3), 7, np.uint8) if agentview_available else None,
            "eye_in_hand": np.full((4, 4, 3), 9, np.uint8),
        },
        availability={"agentview": agentview_available, "eye_in_hand": True},
        robot_state=np.zeros(8, np.float32),
        language="pick up the bowl",
        source_age_steps={"agentview": 0, "eye_in_hand": 0},
    )


def test_a_server_that_is_still_loading_cannot_start_a_rollout():
    server = FakeServer(ready=False)
    try:
        with pytest.raises(RuntimeError, match="not ready"):
            adapter_for(server).connect()
        # Liveness was asked for; identity was never requested.
        assert [message["type"] for message in server.received] == ["health"]
    finally:
        server.close()


def test_health_precedes_identity_on_a_ready_server():
    server = FakeServer()
    adapter = adapter_for(server)
    try:
        identity = adapter.connect()
        assert identity["model_id"] == "somebody/router-vla"
        assert [message["type"] for message in server.received] == ["health", "identity"]
        assert adapter.declaration.training_regime == "clean_trained"
    finally:
        adapter.close()
        server.close()


def test_a_malformed_declaration_is_rejected_at_the_wire():
    server = FakeServer(identity=declaration(reset_semantics=None))
    try:
        with pytest.raises(ContractError, match="missing required fields"):
            adapter_for(server).connect()
    finally:
        server.close()


def test_the_canonical_observation_reaches_the_server_untranslated():
    server = FakeServer()
    adapter = adapter_for(server)
    try:
        adapter.connect()
        adapter.reset(1234)
        adapter.act(observation())
        act = server.received[-1]
        assert act["type"] == "act"
        assert act["step"] == 3
        assert set(act["cameras"]) == set(CAMERAS)
        assert np.all(act["cameras"]["agentview"] == 7)
        assert act["availability"] == {"agentview": True, "eye_in_hand": True}
        assert act["language"] == "pick up the bowl"
        assert act["source_age_steps"] == {"agentview": 0, "eye_in_hand": 0}
        reset = next(m for m in server.received if m["type"] == "reset")
        assert reset["seed"] == 1234
    finally:
        adapter.close()
        server.close()


def test_a_missing_camera_arrives_absent_and_is_not_filled_in():
    # The reference adapters substitute a frozen zero tensor because their models
    # have a fixed input shape. The public protocol must not: an availability
    # aware policy has to be able to see that the camera is gone.
    server = FakeServer()
    adapter = adapter_for(server)
    try:
        adapter.connect()
        adapter.act(observation(agentview_available=False))
        act = server.received[-1]
        assert act["availability"]["agentview"] is False
        assert act["cameras"]["agentview"] is None
    finally:
        adapter.close()
        server.close()


def test_an_action_chunk_must_match_the_declaration():
    server = FakeServer(actions=np.ones((4, 6), np.float32))
    adapter = adapter_for(server)
    try:
        adapter.connect()
        with pytest.raises(RuntimeError, match="action dimension 6"):
            adapter.act(observation())
    finally:
        adapter.close()
        server.close()

    server = FakeServer(actions=np.ones((9, 7), np.float32))
    adapter = adapter_for(server)
    try:
        adapter.connect()
        with pytest.raises(RuntimeError, match="declaration says it predicts 4"):
            adapter.act(observation())
    finally:
        adapter.close()
        server.close()


def test_a_pinned_checkpoint_is_still_enforced():
    server = FakeServer()
    adapter = adapter_for(server, expected_identity={"checkpoint_sha256": "b" * 64})
    try:
        with pytest.raises(RuntimeError, match="checkpoint_sha256"):
            adapter.connect()
    finally:
        server.close()


def test_a_connection_config_may_not_describe_the_policy(tmp_path):
    yaml = pytest.importorskip("yaml")
    good = tmp_path / "good.yaml"
    good.write_text(
        yaml.safe_dump({"name": "my-router", "connection": {"host": "h", "port": 9001}}),
        encoding="utf-8",
    )
    connection = PolicyConnection.load(good)
    assert (connection.name, connection.host, connection.port) == ("my-router", "h", 9001)

    bad = tmp_path / "bad.yaml"
    bad.write_text(
        yaml.safe_dump({
            "name": "my-router",
            "connection": {"host": "h", "port": 9001},
            "policy_visible_cameras": ["agentview"],
        }),
        encoding="utf-8",
    )
    # A config copy of the policy's properties could only ever disagree with the
    # process that actually ran.
    with pytest.raises(ValueError, match="describe the policy, not the connection"):
        PolicyConnection.load(bad)


def test_reset_carries_the_announced_episode_beside_the_seed():
    from mail_bench.interfaces import EpisodeContext
    server = FakeServer()
    with adapter_for(server) as adapter:
        adapter.announce_episode(EpisodeContext(
            task="OpenDrawer", episode_index=4, official_horizon=500))
        adapter.reset(11)
    reset = next(m for m in server.received if m["type"] == "reset")
    assert reset["seed"] == 11
    assert reset["task"] == "OpenDrawer"
    assert reset["episode_index"] == 4 and reset["official_horizon"] == 500


def test_a_stateless_server_is_never_sent_an_invalidation():
    server = FakeServer(identity=declaration(stateful_policy=False,
                                             reset_semantics="stateless"))
    with adapter_for(server) as adapter:
        adapter.reset(1)
        adapter.invalidate_action_chunk("availability_change", 3)
        adapter.reset_visual_memory("availability_recovery", 5)
    assert not [m for m in server.received
                if m["type"] in ("invalidate", "visual_memory_reset")]


def test_a_stateful_server_is_sent_every_invalidation_with_its_step():
    server = FakeServer(identity=declaration(stateful_policy=True,
                                             reset_semantics="clears_all_state"))
    with adapter_for(server) as adapter:
        adapter.reset(1)
        adapter.invalidate_action_chunk("availability_change", 3)
        adapter.reset_visual_memory("availability_recovery", 5)
    kinds = [(m["type"], m.get("step"), m.get("reason")) for m in server.received
             if m["type"] in ("invalidate", "visual_memory_reset")]
    assert kinds == [("invalidate", 3, "availability_change"),
                     ("visual_memory_reset", 5, "availability_recovery")]


def test_a_chunk_with_nan_or_of_the_wrong_length_is_refused_before_execution():
    """Checked at the wire, before anything reaches the simulator: a NaN would
    otherwise reach the simulator and fail at the trace hasher; a chunk shorter
    than declared would let a policy be re-queried more often than its
    declaration says."""
    nan = np.ones((4, 7), np.float32)
    nan[2, 3] = np.nan
    with adapter_for(FakeServer(actions=nan)) as adapter:
        adapter.reset(1)
        with pytest.raises(RuntimeError, match="NaN or Inf"):
            adapter.act(observation())
    with adapter_for(FakeServer(actions=np.ones((3, 7), np.float32))) as adapter:
        adapter.reset(1)
        with pytest.raises(RuntimeError, match="declaration says it predicts 4"):
            adapter.act(observation())
    with adapter_for(FakeServer(actions=np.array([["a"] * 7] * 4))) as adapter:
        adapter.reset(1)
        with pytest.raises(RuntimeError, match="non-numeric"):
            adapter.act(observation())


def test_reset_carries_the_scenes_instruction():
    from mail_bench.interfaces import EpisodeContext
    server = FakeServer()
    with adapter_for(server) as adapter:
        adapter.announce_episode(EpisodeContext(task="OpenDrawer", episode_index=1,
                                                official_horizon=500, instruction="open the drawer"))
        adapter.reset(3)
    reset = next(m for m in server.received if m["type"] == "reset")
    assert reset["instruction"] == "open the drawer"
