"""What admission asks: correctness of the adapter, not clean-success stability.

A clean-success threshold would select for stability and exclude the models the
benchmark exists to measure.
"""

from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")

ROOT = Path(__file__).resolve().parents[1]
ROSTER = ROOT / "configs" / "mail_bench_roster.yaml"
PROTOCOL = ROOT / "configs" / "perturbation_protocol.yaml"


@pytest.fixture(scope="module")
def roster():
    return yaml.safe_load(ROSTER.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def protocol():
    return yaml.safe_load(PROTOCOL.read_text(encoding="utf-8"))


def test_clean_success_is_reported_not_gated(roster):
    """Clean success is reported, never gated on: a threshold would select for
    stability rather than for a correct adapter. Clean success is published as
    H; nothing is withheld for scoring badly."""
    admission = roster["admission"]
    assert admission["clean_success_is"] == "reported_not_gated"
    assert admission["stops_only_when"] == "no_scene_has_a_healthy_success"


def test_admission_asks_about_correctness_not_stability(roster):
    required = roster["admission"]["requires"]
    assert "execution_correctness_check" in required
    assert "a_real_healthy_rollout_completes_the_task" in required
    assert "clean_performance_published_in_full" in required
    # One fixed configuration faces every task; no task may be dropped.
    assert "one_configuration_across_all_eighteen_tasks" in required


def test_a_semantic_change_opens_a_new_protocol_version(protocol):
    from mail_bench.manifest import PROTOCOL_VERSION

    assert protocol["protocol_id"] == PROTOCOL_VERSION
    # healthy_definition is on the list that requires a bump.
    assert "healthy_definition" in protocol["version_policy"]["bump_protocol_version_when_changing"]
    # Released versions are immutable, and the policy says so.
    assert "released_protocol_versions_are_immutable" in protocol["version_policy"]


def test_low_clean_coverage_does_not_stop_the_official_fault_phase(tmp_path):
    """Low clean coverage is a result, not an admission failure.

    A single global target of 0.80 demands a model far stronger than a
    platform's own leading baseline -- at RoboCasa's published PI0.5 rate of
    0.396 a scene reaches two of three with probability about 0.346 -- so
    gating on it selects for stability rather than for a correct adapter and
    excludes the models the benchmark exists to measure. Low clean performance
    is a result, published as H; the scenes the policy did solve still define
    their own task phases and still run their fault cells.
    """
    from mail_bench.driver import cohort_admission_status
    from mail_bench.cohort import admission_gate

    # Nine of thirty solvable: C = 0.30, far below the 0.80 target.
    units = [((f"t{i}", 0), i < 9) for i in range(30)]
    admission = admission_gate(units, cohort_target=0.80)
    assert admission.meets_cohort_target is False
    status = cohort_admission_status(admission)
    assert status == "insufficient_clean_success"
    # Reported, and not a refusal.
    assert status != "checkpoint_incompatible"


def test_solving_nothing_is_a_different_kind_of_outcome(tmp_path):
    # With no healthy success anywhere there is no trajectory to take 30% of, so
    # there is no fault cell to run -- not a decision to withhold one.
    from mail_bench.driver import cohort_admission_status
    from mail_bench.cohort import admission_gate

    admission = admission_gate([((f"t{i}", 0), False) for i in range(30)],
                               cohort_target=0.80)
    assert cohort_admission_status(admission) == "checkpoint_incompatible"
