"""Connection settings for a policy server -- and nothing else.

A config file may say where the server is; it may not say what the policy is.
Camera scope, chunk sizes, checkpoint identity and training regime come from
the live server's declaration, so there is never a config that disagrees with a
running process and no rule about which of the two to believe.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Union

PathLike = Union[str, Path]

#: Fields that describe the policy rather than the connection. Seeing one of
#: these in a connection file is an error, not a default to merge.
POLICY_FIELDS = (
    "policy_visible_cameras",
    "predicted_action_chunk",
    "native_execution_horizon",
    "checkpoint_sha256",
    "training_regime",
    "action_dimension",
    "stateful_policy",
    "reset_semantics",
    "availability_consumed_by_policy",
)


@dataclass(frozen=True)
class PolicyConnection:
    name: str
    host: str = "127.0.0.1"
    port: int = 9000
    timeout_seconds: float = 600.0

    @classmethod
    def load(cls, path: PathLike) -> "PolicyConnection":
        from ..registry import load_yaml

        document = load_yaml(path)
        connection = dict(document.get("connection") or {})
        declared = sorted(set(document) & set(POLICY_FIELDS)) + sorted(
            set(connection) & set(POLICY_FIELDS)
        )
        if declared:
            raise ValueError(
                f"{path}: {declared} describe the policy, not the connection. "
                "The benchmark reads those from the server's identity response, so a "
                "config copy could only ever disagree with the process that ran."
            )
        return cls(
            name=str(document.get("name", "custom-policy")),
            host=str(connection.get("host", "127.0.0.1")),
            port=int(connection.get("port", 9000)),
            timeout_seconds=float(connection.get("timeout_seconds", 600.0)),
        )
