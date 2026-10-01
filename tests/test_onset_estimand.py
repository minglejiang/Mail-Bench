"""The onset fractions name a task phase, and a phase belongs to a performer.

A shared platform reference would fault every model at the same step, which
puts a fast model past the midpoint while a slow one is still in the first
half. That is a coherent experiment -- absolute-time perturbation -- and it is
not the one the main ranking runs.
"""

import re
from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "configs" / "perturbation_protocol.yaml"


@pytest.fixture(scope="module")
def protocol():
    return yaml.safe_load(PROTOCOL.read_text(encoding="utf-8"))


def test_the_healthy_reference_is_the_evaluated_model(protocol):
    rule = protocol["onset_definition"]["rule"]
    assert "THE EVALUATED MODEL ITSELF" in rule
    assert "T_healthy(model, scene)" in rule


def test_the_onset_reference_is_the_evaluated_policy_only(protocol):
    """Guard the assumption, not just the wording."""
    text = PROTOCOL.read_text(encoding="utf-8")
    for key in re.findall(r"^\s*(onset_healthy_reference|reference_policy_for_onset):\s*(\S+)",
                          text, re.M):
        name, value = key
        assert name == "onset_healthy_reference", "a shared reference key is not permitted"
        assert value == "evaluated_policy_healthy_reference", value

    # And nothing may ask a platform profile for a reference checkpoint.
    profiles = yaml.safe_load(
        (ROOT / "configs" / "platform_profiles.yaml").read_text(encoding="utf-8"))
    for platform, profile in profiles["platforms"].items():
        assert not any("reference" in key for key in profile), platform


def test_a_failed_healthy_scene_keeps_its_place_in_the_overall_matrix(protocol):
    rule = protocol["onset_definition"]["rule"]
    # No onset is invented for it, and its fault cells are not run.
    assert "ITS FAULT ROLLOUTS ARE NOT EXECUTED" in rule
    assert "No\n    horizon-derived onset is assigned" in rule or \
        "No horizon-derived onset is assigned" in " ".join(rule.split())
    # But the scene is still counted, so a weak model cannot look robust by
    # solving only the easy scenes.
    assert "stays in the clean and fault matrices as a clean failure" in " ".join(rule.split())
    handling = protocol["onset_definition"]["healthy_failure_handling"]
    assert handling["fault_evaluable"] is False
    assert handling["fault_rollouts_executed"] == 0
    # The scene keeps its place in the capability number and is absent from
    # the conditional one, which is undefined without a healthy reference.
    assert "healthy_score" in handling["retained_in"]
    assert "missing_condition_score" in handling["excluded_from"]


def test_each_model_publishes_its_own_onset_manifest(protocol):
    rule = protocol["onset_definition"]["rule"]
    for field in ("scene_identity", "healthy_success", "T_healthy", "fault_evaluable", "skip_reason"):
        assert field in rule, field
    assert "rather than the platform publishing one shared reference file" in rule


def test_the_estimand_is_stated_and_the_alternative_named(protocol):
    estimand = protocol["onset_definition"]["estimand"]
    assert "phase-normalised visual robustness" in estimand
    # The design it is not, said out loud so a reader cannot assume otherwise.
    assert "Identical absolute-time perturbation" in estimand


def test_phase_alignment_is_what_model_relative_onsets_buy():
    """The arithmetic the decision rests on."""
    def onset(t_healthy, fraction):
        return int(fraction * t_healthy + 0.5)

    fast, slow = 300, 600
    assert onset(fast, 0.45) == 135
    assert onset(slow, 0.45) == 270
    # A shared step cannot be mid-task for both.
    shared = 180
    assert shared > onset(fast, 0.45) and shared < onset(slow, 0.45)
