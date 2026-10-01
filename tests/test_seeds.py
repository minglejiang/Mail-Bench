import random

import pytest

from mail_bench.seeds import (
    audit_seed_projection,
    Streams,
    derive_seed,
    episode_seed,
    fault_seed,
    healthy_replicate_policy_seed,
    healthy_replicate_seeds,
    policy_seed,
)

UNIT = ("fake", "task_a", 7, "v1")


def stream_bytes(rng: random.Random, n: int = 64) -> bytes:
    """Bytes drawn from a stream, to compare streams."""
    return rng.randbytes(n)


def test_derive_seed_is_deterministic_64bit():
    a = derive_seed("ns", "x", 1, [1, 2], {"b": 2, "a": 1})
    b = derive_seed("ns", "x", 1, [1, 2], {"a": 1, "b": 2})
    assert a == b
    assert 0 <= a < 2**64


def test_namespaces_separate_streams():
    assert episode_seed(*UNIT) != policy_seed(*UNIT) != fault_seed(*UNIT)
    assert derive_seed("a", 1) != derive_seed("b", 1)


def test_policy_stream_identical_across_fault_params():
    s_missing = Streams.for_unit(*UNIT, "hard_missing", ["cam0"], 0.3)
    s_freeze = Streams.for_unit(*UNIT, "freeze", ["cam1"], 0.6)
    assert s_missing.policy_seed == s_freeze.policy_seed
    assert s_missing.environment_seed == s_freeze.environment_seed
    assert s_missing.fault_seed != s_freeze.fault_seed
    assert stream_bytes(s_missing.policy(), 256) == stream_bytes(s_freeze.policy(), 256)


def test_fault_stream_never_touches_policy_stream():
    s = Streams.for_unit(*UNIT, "burst_dropout")
    pol_before = stream_bytes(s.policy(), 128)
    f = s.fault()
    for _ in range(1000):
        f.random()
    assert stream_bytes(s.policy(), 128) == pol_before  # fresh generators each time
    assert stream_bytes(s.fault(), 32) != stream_bytes(s.policy(), 32)


def test_episode_index_changes_all_streams():
    a = Streams.for_unit("b", "t", 0, "v1")
    b = Streams.for_unit("b", "t", 1, "v1")
    assert a.environment_seed != b.environment_seed
    assert a.policy_seed != b.policy_seed
    assert a.fault_seed != b.fault_seed


def test_healthy_replicates_freeze_environment_and_separate_policy_streams():
    replicate_seeds = healthy_replicate_seeds(*UNIT, replicates=3)
    assert len({environment for environment, _ in replicate_seeds}) == 1
    assert len({policy for _, policy in replicate_seeds}) == 3
    assert replicate_seeds[0][1] == policy_seed(*UNIT)
    assert healthy_replicate_policy_seed(*UNIT, 1) == replicate_seeds[1][1]
    with pytest.raises(ValueError, match="replicate_index"):
        healthy_replicate_policy_seed(*UNIT, -1)


def test_seed_projection_audit_reports_only_distinct_protocol_seed_collisions():
    audit = audit_seed_projection([1, 1, (1 << 32) + 1, 2], target_bits=32)
    assert audit.protocol_seed_count == 3
    assert audit.unique_projected_seed_count == 2
    assert audit.has_collisions is True
    assert audit.collisions[0].projected_seed == 1
    assert audit.collisions[0].protocol_seeds == (1, (1 << 32) + 1)


def test_seed_projection_audit_rejects_invalid_width():
    with pytest.raises(ValueError, match="target_bits"):
        audit_seed_projection([1], target_bits=0)
