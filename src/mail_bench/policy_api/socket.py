"""Shared client for policies served from a dedicated GPU process.

Every model in the roster runs in its own Python environment, so each one is
reached over the same length-prefixed local socket protocol (:mod:`mail_bench.net`)
rather than being imported into the simulator process.  This base class owns the
connection, the identity check and the teardown; a subclass only has to turn one
:class:`CanonicalObservation` into a request and the reply into a
:class:`PolicyOutput`.

The identity check is the point of the class: a server whose protocol,
checkpoint hash or upstream revision differs from what the frozen method
declares is refused before a single action is taken, so a cell can never be
attributed to weights it did not run.
"""

from __future__ import annotations

import time

import socket
from typing import Any, Mapping, Optional

from ..interfaces import EpisodeContext, PolicyAdapter


class SocketPolicyAdapter(PolicyAdapter):
    """Query a local policy server, binding the connection to a frozen identity."""

    #: Protocol string the server must announce.
    protocol: str = ""
    #: ``True`` for a server that answers a single ``ping`` carrying its
    #: identity; ``False`` for the public protocol, which answers ``health``
    #: and ``identity`` separately, because an open TCP socket says nothing
    #: about whether a model has finished loading.
    legacy_ping: bool = False

    def __init__(
        self,
        *,
        host: str = "127.0.0.1",
        port: int,
        timeout_seconds: float = 600.0,
        expected_identity: Optional[Mapping[str, Any]] = None,
    ) -> None:
        self.host = host
        self.port = int(port)
        self.timeout_seconds = float(timeout_seconds)
        # Only non-None entries are enforced, so a caller may pin as much or as
        # little of the identity as it has frozen.
        self.expected_identity = {
            key: value for key, value in dict(expected_identity or {}).items() if value is not None
        }
        self._socket: Any = None
        self.server_identity: Optional[Mapping[str, Any]] = None
        #: Cost of the most recent request. Diagnostic only; the difference
        #: between the two is serialisation, transport and Python overhead.
        self.last_rtt_ms: Optional[float] = None
        self.last_server_inference_ms: Optional[float] = None
        self._episode: EpisodeContext = EpisodeContext()

    def connect(self) -> Mapping[str, Any]:
        if self._socket is None:
            self._socket = socket.create_connection(
                (self.host, self.port), timeout=self.timeout_seconds
            )
            self._socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            try:
                if self.legacy_ping:
                    identity = self._request({"type": "ping"})
                else:
                    self.require_ready()
                    identity = self._request({"type": "identity"})
                self.verify_identity(identity)
            except Exception:
                self._socket.close()
                self._socket = None
                raise
            self.server_identity = identity
        return dict(self.server_identity or {})

    def health(self) -> Mapping[str, Any]:
        """Ask whether the model is loaded, not merely whether the port is open."""
        return self._request({"type": "health"})

    def require_ready(self) -> Mapping[str, Any]:
        """Refuse to start anything while the policy is still coming up."""
        status = self.health()
        if not status.get("ready"):
            raise RuntimeError(
                f"policy server is not ready (status={status.get('status')!r}); "
                "no rollout may start before the model has finished loading"
            )
        return status

    def verify_identity(self, identity: Mapping[str, Any]) -> None:
        if identity.get("protocol") != self.protocol:
            raise RuntimeError(
                f"policy server protocol is {identity.get('protocol')!r}, expected {self.protocol!r}"
            )
        for key, expected in self.expected_identity.items():
            actual = identity.get(key)
            if actual != expected:
                raise RuntimeError(
                    f"policy server {key} is {actual!r}, but the frozen method declares {expected!r}"
                )

    def _request(self, message: Mapping[str, Any]) -> Mapping[str, Any]:
        from ..net import recv_message, send_message

        if self._socket is None:
            raise RuntimeError("policy server is not connected")
        # Round-trip time costs a clock read and no synchronisation, so it is
        # always recorded; the server's own timing arrives only when that server
        # was started with profiling on. Both are observability only and are
        # never read by the runner: they must not reach a result, a hash or a
        # cell identity.
        started = time.perf_counter_ns()
        send_message(self._socket, dict(message))
        response = recv_message(self._socket)
        self.last_rtt_ms = (time.perf_counter_ns() - started) / 1e6
        self.last_server_inference_ms = (
            response.get("server_inference_ms") if isinstance(response, dict) else None
        )
        if not isinstance(response, dict):
            raise RuntimeError("policy server returned a non-object response")
        if not response.get("ok"):
            raise RuntimeError(f"policy server error: {response.get('error', 'unknown error')}")
        return response

    def announce_episode(self, context: EpisodeContext) -> None:
        self._episode = context

    def reset(self, policy_seed: int) -> None:
        """Start an episode: the seed, and what a deployed policy would know.

        ``task``, ``episode_index`` and ``official_horizon`` ride along with
        the seed. They are public in the frozen suite and a policy that
        budgets against the horizon needs them; nothing about the fault
        schedule is sent.
        """
        self.connect()
        message: dict[str, Any] = {"type": "reset", "seed": int(policy_seed)}
        message.update(self._episode.as_message_fields())
        self._request(message)

    def _stateful(self) -> bool:
        identity = self.server_identity or {}
        return bool(identity.get("stateful_policy"))

    def invalidate_action_chunk(self, reason: str, step: int) -> None:
        """Forward the kernel's invalidation to a policy that keeps state.

        The kernel drops queued actions when availability changes; a policy
        that holds a chunk, a lease or a plan of its own has to hear that too,
        or it would keep acting on a decision the kernel has already discarded.
        A stateless server has nothing to drop and is not sent the message.
        """
        if self._stateful():
            self.connect()
            self._request({"type": "invalidate", "reason": str(reason), "step": int(step)})

    def reset_visual_memory(self, reason: str, step: int) -> None:
        if self._stateful():
            self.connect()
            self._request({"type": "visual_memory_reset",
                           "reason": str(reason), "step": int(step)})

    def close(self) -> None:
        if self._socket is not None:
            try:
                # A goodbye, not a query: a server that is wedged mid-message
                # must not hold the teardown for the whole policy timeout.
                self._socket.settimeout(5.0)
                self._request({"type": "disconnect"})
            except (ConnectionError, OSError, RuntimeError, ValueError):
                # A best-effort goodbye must never keep the socket open or mask
                # the outcome of the rollout that is being torn down.
                pass
            finally:
                self._socket.close()
                self._socket = None

    def __enter__(self) -> "SocketPolicyAdapter":
        self.connect()
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()
