from pathlib import Path

from mail_bench.manifest import semantic_sha256
from mail_bench.runtime import collect_runtime_fingerprint


ROOT = Path(__file__).resolve().parents[1]


def test_runtime_fingerprint_is_self_hashing_and_ci_safe():
    fingerprint = collect_runtime_fingerprint(ROOT)
    digest = fingerprint.pop("fingerprint_hash")
    assert digest == semantic_sha256(fingerprint)
    assert fingerprint["schema_version"] == 2
    assert fingerprint["python_version"]
    assert set(fingerprint["rendering"]) == {"MUJOCO_GL", "PYOPENGL_PLATFORM"}
    assert isinstance(fingerprint["gpu_names"], list)
