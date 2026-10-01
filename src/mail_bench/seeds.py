"""Seed derivation with separated RNG streams.

MAIL-Bench pairs every healthy / fault / recovery arm on the same task,
initial state and randomness.  To make that pairing auditable, every seed is
*derived* (never drawn) from a canonical description of the evaluation unit:

    (benchmark, task, episode_index, protocol_version)

Three streams are derived under three different namespaces:

* ``environment`` -- simulator reset, object placement, physics noise;
* ``policy``      -- policy-side sampling noise (diffusion noise, dropout, ...);
* ``fault``       -- fault-injection randomness (burst sequences, corruption
  parameters, ...).

Invariants (tested in ``tests/test_seeds.py``):

1. The policy stream is a pure function of the evaluation unit.  Its byte
   sequence is therefore *identical across arms* (the healthy cell and every
   fault cell) and identical for any fault configuration.
2. The fault stream never touches the policy stream: fault parameters may be
   mixed into ``fault_seed`` but they are never mixed into ``policy_seed`` or
   ``episode_seed``.
3. Seeds are stable across Python processes, platforms and hash
   randomisation because they are SHA256 digests over canonical JSON.

Only the standard library is required; NumPy generators are exposed lazily
and only when NumPy is importable.
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from typing import Any, Iterable

SEED_BITS = 64
SEED_MASK = (1 << SEED_BITS) - 1

# Namespaces are fixed by configs/perturbation_protocol.yaml (pairing.seed_derivation).
NAMESPACE_ENVIRONMENT = "mcr.env"
NAMESPACE_POLICY = "mcr.policy"
NAMESPACE_FAULT = "mcr.fault"


@dataclass(frozen=True)
class SeedProjectionCollision:
    projected_seed: int
    protocol_seeds: tuple[int, ...]


@dataclass(frozen=True)
class SeedProjectionAudit:
    target_bits: int
    protocol_seed_count: int
    unique_projected_seed_count: int
    collisions: tuple[SeedProjectionCollision, ...]

    @property
    def has_collisions(self) -> bool:
        return bool(self.collisions)


def project_seed(seed: int, target_bits: int) -> int:
    """Project a protocol seed to a backend's unsigned seed width."""
    if target_bits < 1:
        raise ValueError("target_bits must be >= 1")
    return int(seed) & ((1 << target_bits) - 1)


def audit_seed_projection(seeds: Iterable[int], target_bits: int) -> SeedProjectionAudit:
    """Report distinct protocol seeds that collide after backend projection."""
    unique_protocol_seeds = tuple(sorted({int(seed) for seed in seeds}))
    groups: dict[int, list[int]] = {}
    for seed in unique_protocol_seeds:
        groups.setdefault(project_seed(seed, target_bits), []).append(seed)
    collisions = tuple(
        SeedProjectionCollision(projected, tuple(protocol_seeds))
        for projected, protocol_seeds in sorted(groups.items())
        if len(protocol_seeds) > 1
    )
    return SeedProjectionAudit(
        target_bits=target_bits,
        protocol_seed_count=len(unique_protocol_seeds),
        unique_projected_seed_count=len(groups),
        collisions=collisions,
    )


def _jsonable(value: Any) -> Any:
    """Convert ``value`` to a JSON-serialisable, order-stable structure."""
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (set, frozenset)):
        return sorted((_jsonable(v) for v in value), key=repr)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if hasattr(value, "__dataclass_fields__"):
        return {k: _jsonable(getattr(value, k)) for k in value.__dataclass_fields__}
    return repr(value)


def canonical_parts_json(namespace: str, parts: Iterable[Any]) -> str:
    """Return the canonical JSON string hashed by :func:`derive_seed`."""
    payload = {"namespace": namespace, "parts": [_jsonable(p) for p in parts]}
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    )


def derive_seed(namespace: str, *parts: Any) -> int:
    """Derive a deterministic 64-bit unsigned seed.

    The seed is the first eight bytes (big-endian) of
    ``SHA256(canonical_json({"namespace": namespace, "parts": parts}))``.
    Different namespaces yield statistically independent seeds for the same
    parts, which is how the environment / policy / fault streams are kept
    separate.

    Parameters
    ----------
    namespace:
        Free-form string identifying the stream (see module constants).
    parts:
        Any JSON-serialisable values (ints, strings, floats, lists, dicts,
        dataclasses, sets).  Sets are sorted; dataclasses are expanded to
        field dictionaries.
    """
    digest = hashlib.sha256(canonical_parts_json(namespace, parts).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") & SEED_MASK


def episode_seed(benchmark: str, task: str, episode_index: int, protocol_version: str) -> int:
    """Environment seed for one paired evaluation unit."""
    return derive_seed(NAMESPACE_ENVIRONMENT, benchmark, task, int(episode_index), protocol_version)


def policy_seed(benchmark: str, task: str, episode_index: int, protocol_version: str) -> int:
    """Policy-noise seed for one paired evaluation unit.

    Deliberately has *no* fault-parameter arguments: every method arm and
    every fault cell evaluated on this unit must use the same policy seed.
    """
    return derive_seed(NAMESPACE_POLICY, benchmark, task, int(episode_index), protocol_version)


def healthy_replicate_policy_seed(
    benchmark: str,
    task: str,
    episode_index: int,
    protocol_version: str,
    replicate_index: int,
) -> int:
    """Policy seed for one healthy replicate while preserving replicate zero pairing.

    Replicate zero is the canonical policy stream used by every fault arm.
    Additional healthy replicates get deterministic, namespace-separated policy
    streams without changing the frozen environment seed.
    """
    if replicate_index < 0:
        raise ValueError("replicate_index must be >= 0")
    if replicate_index == 0:
        return policy_seed(benchmark, task, episode_index, protocol_version)
    return derive_seed(
        NAMESPACE_POLICY,
        benchmark,
        task,
        int(episode_index),
        protocol_version,
        "healthy_replicate",
        int(replicate_index),
    )


def healthy_replicate_seeds(
    benchmark: str,
    task: str,
    episode_index: int,
    protocol_version: str,
    replicates: int = 1,
) -> tuple[tuple[int, int], ...]:
    """Return ``(environment_seed, policy_seed)`` for healthy references."""
    if replicates < 1:
        raise ValueError("replicates must be >= 1")
    env_seed = episode_seed(benchmark, task, episode_index, protocol_version)
    return tuple(
        (
            env_seed,
            healthy_replicate_policy_seed(
                benchmark, task, episode_index, protocol_version, replicate_index
            ),
        )
        for replicate_index in range(replicates)
    )


def fault_seed(
    benchmark: str,
    task: str,
    episode_index: int,
    protocol_version: str,
    *fault_parts: Any,
) -> int:
    """Fault-injection seed for one evaluation unit.

    ``fault_parts`` (e.g. mode, faulted camera ids, onset fraction) may be
    mixed in so that different fault cells get different burst sequences or
    corruption parameters.  They never influence the policy seed.
    """
    return derive_seed(
        NAMESPACE_FAULT, benchmark, task, int(episode_index), protocol_version, *fault_parts
    )


@dataclass(frozen=True)
class Streams:
    """Independent RNG streams for one evaluation unit.

    Attributes
    ----------
    environment_seed, policy_seed, fault_seed:
        The three derived 64-bit seeds.

    Each accessor returns a *fresh* generator seeded from the corresponding
    seed so that consumers cannot accidentally share generator state.
    """

    environment_seed: int
    policy_seed: int
    fault_seed: int

    @classmethod
    def for_unit(
        cls,
        benchmark: str,
        task: str,
        episode_index: int,
        protocol_version: str,
        *fault_parts: Any,
    ) -> "Streams":
        """Build the three streams for an evaluation unit."""
        return cls(
            environment_seed=episode_seed(benchmark, task, episode_index, protocol_version),
            policy_seed=policy_seed(benchmark, task, episode_index, protocol_version),
            fault_seed=fault_seed(benchmark, task, episode_index, protocol_version, *fault_parts),
        )

    # --- standard library generators -------------------------------------
    def environment(self) -> random.Random:
        """Fresh ``random.Random`` for the environment stream."""
        return random.Random(self.environment_seed)

    def policy(self) -> random.Random:
        """Fresh ``random.Random`` for the policy stream."""
        return random.Random(self.policy_seed)

    def fault(self) -> random.Random:
        """Fresh ``random.Random`` for the fault stream."""
        return random.Random(self.fault_seed)
    def as_dict(self) -> dict[str, int]:
        """Seeds as a plain dictionary (for manifests)."""
        return {
            "environment_seed": self.environment_seed,
            "policy_seed": self.policy_seed,
            "fault_seed": self.fault_seed,
        }
