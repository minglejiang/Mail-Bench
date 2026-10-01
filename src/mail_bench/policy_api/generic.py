"""The public wire protocol: any policy server, no benchmark-side translation.

A third party implements one small socket contract and is evaluated without
forking this repository, writing an adapter here, or appearing in any roster:

    health      -> {"ok": true, "ready": bool, "status": str}
    identity    -> the PolicyDeclaration (see contract.py)
    reset       -> {"ok": true}                       once per episode
    act         -> {"ok": true, "actions": [[...]]}    one canonical observation
    disconnect  -> {"ok": true}

``reset`` carries ``seed`` and, beside it, ``task``, ``episode_index`` and
``official_horizon``: what a deployed policy knows before it starts, and
nothing about the fault schedule.  The seed must drive every source of
randomness in the policy, its own sampling included, so that the healthy
rollout and every fault arm of a scene draw the same policy stream.

A server that declares ``stateful_policy: true`` is additionally sent

    invalidate          {"reason": str, "step": int}   the kernel dropped queued actions
    visual_memory_reset {"reason": str, "step": int}   a preregistered memory reset

and must answer ``{"ok": true}``; a stateless server never receives them.

``act`` carries the canonical observation exactly as the kernel produced it.
Nothing is translated on the way out: image layout, resolution, normalisation,
camera concatenation and rotation are the server's business, because they are
properties of a model rather than of the benchmark.

**Missing cameras are not filled in here.**  A camera the manifest declares
unavailable arrives as it is, with ``availability[camera] = false``.  The
reference adapters substitute a frozen zero tensor because the models behind
them have a fixed input shape and no notion of an absent slot -- that is
reference-model preprocessing, not protocol.  An availability-aware router, a
memory model, an imputer or a mask-aware policy must be free to see the absence
and decide for itself; the benchmark may not make that decision on its behalf.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional

from ..interfaces import CanonicalObservation, PolicyOutput
from .contract import PolicyDeclaration, validate_declaration
from .socket import SocketPolicyAdapter


PROTOCOL = "mcr-policy-v1"


class GenericSocketPolicyAdapter(SocketPolicyAdapter):
    """Talk to any server that speaks :data:`PROTOCOL`."""

    protocol = PROTOCOL
    legacy_ping = False

    def __init__(
        self,
        *,
        host: str = "127.0.0.1",
        port: int,
        timeout_seconds: float = 600.0,
        expected_identity: Optional[Mapping[str, Any]] = None,
        environment_cameras: Optional[tuple[str, ...]] = None,
    ) -> None:
        super().__init__(
            host=host,
            port=port,
            timeout_seconds=timeout_seconds,
            expected_identity=expected_identity,
        )
        self.environment_cameras = environment_cameras
        self.declaration: Optional[PolicyDeclaration] = None

    def verify_identity(self, identity: Mapping[str, Any]) -> None:
        super().verify_identity(identity)
        # The declaration is validated at the wire, so a malformed server is
        # rejected before the first reset rather than mid-cohort.
        self.declaration = validate_declaration(
            identity, environment_cameras=self.environment_cameras
        )

    def act(self, observation: CanonicalObservation) -> PolicyOutput:
        self.connect()
        message: dict[str, Any] = {
            "type": "act",
            "step": observation.step,
            "cameras": dict(observation.cameras),
            "availability": dict(observation.availability),
            "robot_state": observation.robot_state,
            "language": observation.language,
            "capture_time_ms": dict(observation.capture_time_ms),
            "arrival_time_ms": dict(observation.arrival_time_ms),
            "sequence_id": dict(observation.sequence_id),
            "new_frame": dict(observation.new_frame),
            "source_step": dict(observation.source_step),
            "source_age_steps": dict(observation.source_age_steps),
        }
        response = self._request(message)
        return self._action_chunk(response.get("actions"))

    def _action_chunk(self, actions: Any) -> PolicyOutput:
        import numpy as np

        if actions is None or getattr(actions, "ndim", None) != 2:
            raise RuntimeError("policy server returned no two-dimensional action chunk")
        # Checked before anything is executed: a NaN would otherwise reach the
        # simulator first and fail at the trace hasher afterwards, blamed on
        # canonical JSON rather than on the policy.
        if not np.issubdtype(np.asarray(actions).dtype, np.number):
            raise RuntimeError(f"policy server returned a non-numeric chunk ({actions.dtype})")
        if not np.all(np.isfinite(np.asarray(actions, dtype=np.float64))):
            raise RuntimeError("policy server returned a chunk with NaN or Inf")
        declaration = self.declaration
        if declaration is not None:
            if actions.shape[1] != declaration.action_dimension:
                raise RuntimeError(
                    f"policy server returned action dimension {actions.shape[1]}, but its "
                    f"declaration says {declaration.action_dimension}"
                )
            if actions.shape[0] != declaration.predicted_action_chunk:
                # Exactly what was declared: fewer actions would let a policy
                # be re-queried more often than its declaration says.
                raise RuntimeError(
                    f"policy server returned {actions.shape[0]} actions, but its "
                    f"declaration says it predicts {declaration.predicted_action_chunk}"
                )
        return PolicyOutput(tuple(actions))
