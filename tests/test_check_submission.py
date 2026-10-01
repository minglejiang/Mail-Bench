"""The preflight must refuse whatever an official run would refuse.

Each server below is well formed except in one way, and the check that names
that way is the one that has to fail.  A preflight that passed everything, or
that failed a server the kernel would have accepted, would be worse than none:
a submitter would trust it.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path
import socket
import sys
import threading

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mail_bench.net import recv_message, send_message  # noqa: E402


def load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    # Registered before execution: a dataclass in the script resolves its own
    # module while the class body runs.
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


check_submission = load_script("check_submission")
CAMERAS = check_submission.platform_cameras(ROOT / "configs" / "platform_profiles.yaml")
CHUNK, DIM = 4, 12


class Server:
    """A third-party server speaking mcr-policy-v1, breakable one way at a time."""

    def __init__(self, *, identity_overrides=None, drift=False, die_on_absent=0,
                 refuse_hooks=False, ignore_seed=False, nan=False):
        self.identity_overrides = identity_overrides or {}
        self.drift = drift
        self.nan = nan
        #: Die only when at least this many cameras arrive as None, so that a
        #: server which survives one loss and not total loss can be expressed.
        self.die_on_absent = int(die_on_absent)
        self.refuse_hooks = refuse_hooks
        self.ignore_seed = ignore_seed
        self.calls = 0
        self.seed = 0
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.bind(("127.0.0.1", 0))
        self.socket.listen(4)
        self.port = self.socket.getsockname()[1]
        threading.Thread(target=self._serve, daemon=True).start()

    @property
    def identity(self):
        return {
            "ok": True, "protocol": "mcr-policy-v1", "model_id": "somebody/policy",
            "checkpoint_sha256": "a" * 64, "policy_visible_cameras": list(CAMERAS),
            "availability_consumed_by_policy": False, "predicted_action_chunk": CHUNK,
            "native_execution_horizon": 2, "action_dimension": DIM,
            "stateful_policy": False, "reset_semantics": "stateless",
            "training_regime": "clean_trained",
            **self.identity_overrides,
        }

    def _chunk(self):
        self.calls += 1
        base = 0.0 if self.ignore_seed else float(self.seed % 97)
        drift = 1e-3 * self.calls if self.drift else 0.0
        chunk = np.full((CHUNK, DIM), base + drift, np.float32)
        if self.nan:
            chunk[0, 0] = np.nan
        return chunk

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
                        self.seed = int(message["seed"])
                        send_message(connection, {"ok": True})
                    elif kind == "act":
                        absent = [c for c, frame in message["cameras"].items() if frame is None]
                        if self.die_on_absent and len(absent) >= self.die_on_absent:
                            send_message(connection, {"ok": False,
                                                      "error": "expected an array, got None"})
                        else:
                            send_message(connection, {"ok": True, "actions": self._chunk()})
                    elif kind in ("invalidate", "visual_memory_reset"):
                        send_message(connection, {"ok": not self.refuse_hooks,
                                                  "error": "unsupported"})
                    elif kind == "disconnect":
                        send_message(connection, {"ok": True})
                        break
                    else:
                        send_message(connection, {"ok": False, "error": f"unknown {kind}"})

    def close(self):
        self.socket.close()


def statuses(server, **kwargs):
    argv = ["--policy-port", str(server.port), "--image-size", "32"]
    for key, value in kwargs.items():
        argv += [f"--{key.replace('_', '-')}", str(value)]
    args = check_submission.parse_args(argv)
    return {check.name: check.status for check in check_submission.run_checks(args)}


@pytest.fixture
def server_factory():
    made = []

    def make(**kwargs):
        server = Server(**kwargs)
        made.append(server)
        return server

    yield make
    for server in made:
        server.close()


def test_a_well_formed_server_is_not_refused(server_factory):
    report = statuses(server_factory())
    assert not [name for name, status in report.items() if status == check_submission.FAIL]
    assert report["declaration"] == check_submission.PASS
    assert report["seed repeatability"] == check_submission.PASS
    # Nothing was started twice, so the one condition a single process cannot
    # show must be reported as unchecked rather than passed.
    assert report["cross-process determinism"] == check_submission.UNCHECKED


def test_a_chunk_that_drifts_between_episodes_is_refused(server_factory):
    assert statuses(server_factory(drift=True))["seed repeatability"] == check_submission.FAIL


def test_every_ranked_missing_state_is_tried_not_one_camera(server_factory):
    """A server can survive losing the wrist view and still die on total loss.

    Ranking removes a role, not a camera: agentview_missing takes both
    third-person views together. A preflight that dropped one camera at a time
    would call this server healthy.
    """
    report = statuses(server_factory(die_on_absent=3))
    assert report["wrist_missing"] == check_submission.PASS
    assert report["agentview_missing"] == check_submission.PASS
    assert report["all_vision_missing"] == check_submission.FAIL


def test_a_chunk_carrying_a_nan_is_refused(server_factory):
    """NaN passes every shape check and only fails inside the simulator."""
    report = statuses(server_factory(nan=True))
    assert report["action chunk"] == check_submission.FAIL


def test_a_malformed_declaration_is_refused_before_anything_runs(server_factory):
    server = server_factory(identity_overrides={"native_execution_horizon": CHUNK + 1})
    report = statuses(server)
    assert report == {"declaration": check_submission.FAIL}


def test_a_policy_that_never_reads_the_seed_is_flagged_but_not_refused(server_factory):
    """Repeatability alone cannot tell a seeded policy from one ignoring seeds.

    Nothing in the kernel refuses this, so calling it a refusal would fail a
    server an official run accepts. It is a warning: the run finishes and the
    pairing the score rests on is void.
    """
    report = statuses(server_factory(ignore_seed=True))
    assert report["seed repeatability"] == check_submission.PASS
    assert report["seed reaches the policy"] == check_submission.WARN


def test_a_declared_deterministic_policy_may_ignore_the_seed(server_factory):
    report = statuses(server_factory(
        ignore_seed=True, identity_overrides={"policy_determinism": "deterministic"}))
    assert report["seed reaches the policy"] == check_submission.PASS


def test_a_stateful_server_must_answer_the_hooks(server_factory):
    good = server_factory(identity_overrides={"stateful_policy": True,
                                              "reset_semantics": "clears_all_state"})
    assert statuses(good)["invalidate"] == check_submission.PASS
    bad = server_factory(refuse_hooks=True,
                         identity_overrides={"stateful_policy": True,
                                             "reset_semantics": "clears_all_state"})
    assert statuses(bad)["invalidate"] == check_submission.FAIL


def test_a_second_process_that_disagrees_is_refused(server_factory):
    first = server_factory()
    second = server_factory(identity_overrides={}, drift=False)
    second.seed = 0
    args = check_submission.parse_args(
        ["--policy-port", str(first.port), "--second-port", str(second.port),
         "--image-size", "32"])
    report = {check.name: check.status for check in check_submission.run_checks(args)}
    # Two identical servers agree; the check is real only if it can also fail.
    assert report["cross-process determinism"] == check_submission.PASS
    drifting = server_factory(drift=True)
    args = check_submission.parse_args(
        ["--policy-port", str(first.port), "--second-port", str(drifting.port),
         "--image-size", "32"])
    report = {check.name: check.status for check in check_submission.run_checks(args)}
    assert report["cross-process determinism"] == check_submission.FAIL


def test_an_action_dimension_the_platform_cannot_execute_is_refused(server_factory):
    """resolve_runtime_contract refuses this before the first scene, so the
    preflight has to as well; it is the cheapest hours-saving check there is."""
    report = statuses(server_factory(identity_overrides={"action_dimension": 7}))
    assert report["action dimension"] == check_submission.FAIL
    assert statuses(server_factory())["action dimension"] == check_submission.PASS
