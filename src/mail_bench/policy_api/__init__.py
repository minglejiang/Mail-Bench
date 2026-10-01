"""The boundary between MAIL-Bench and any policy under evaluation.

A submission is a policy server, not a patch to this repository.  The kernel
speaks one socket protocol and validates one declaration; what runs behind it --
a VLA, a world-action model, a router in front of several policies, a custom
architecture -- is deliberately invisible to the benchmark.
"""

from ..interfaces import CanonicalObservation, PolicyAdapter, PolicyOutput
from .connection import PolicyConnection
from .contract import (
    TRAINING_REGIME_DISCLOSURE_FIELDS,
    REQUIRED_IDENTITY_FIELDS,
    RESET_SEMANTICS,
    ContractError,
    PolicyDeclaration,
    cameras_outside_declared_scope,
    validate_declaration,
)
from .generic import PROTOCOL, GenericSocketPolicyAdapter
from .socket import SocketPolicyAdapter

__all__ = [
    "CanonicalObservation",
    "PolicyAdapter",
    "PolicyOutput",
    "SocketPolicyAdapter",
    "GenericSocketPolicyAdapter",
    "PROTOCOL",
    "PolicyConnection",
    "TRAINING_REGIME_DISCLOSURE_FIELDS",
    "REQUIRED_IDENTITY_FIELDS",
    "RESET_SEMANTICS",
    "ContractError",
    "PolicyDeclaration",
    "cameras_outside_declared_scope",
    "validate_declaration",
]
