"""MAIL-Bench: ten equal conditions, three visual roles, eighteen tasks."""

import pytest

from mail_bench.scoring import (
    benchmark_task_count,
    MISSING_CONDITIONS,
    RANKING_COMPONENTS,
    BenchmarkScore,
    ScoringError,
    TaskScore,
    condition_scores,
    one_policy_configuration,
)
from mail_bench.semantic_states import (
    MISSING_STATES,
    SemanticStateError,
    state_cameras,
    state_of,
    validate_platform,
)

ROBOCASA = ["robot0_agentview_left", "robot0_agentview_right", "robot0_eye_in_hand"]


# -- semantic states ---------------------------------------------------------

def test_the_platform_answers_the_same_ten_questions_as_any_other_would():
    # A condition is a functional role, not a camera name: a platform answers
    # ten questions whatever its camera count, so a later platform's numbers
    # can be averaged with these.
    assert RANKING_COMPONENTS == 10
    assert [f"{s}@{o:.2f}" for s, o in MISSING_CONDITIONS] == [
        f"{s}@{o:.2f}" for s in MISSING_STATES for o in (0.30, 0.45, 0.60)]
    for state in MISSING_STATES:
        assert state_cameras("robocasa365", state)


def test_a_role_served_by_two_cameras_loses_both_together():
    # Losing one of two redundant onboard views is a materially milder condition
    # than losing the third-person role, so it is not that condition.
    assert state_cameras("robocasa365", "agentview_missing") == (
        "robot0_agentview_left", "robot0_agentview_right")
    assert state_cameras("robocasa365", "wrist_missing") == ("robot0_eye_in_hand",)


def test_all_vision_missing_is_every_camera_of_both_roles():
    assert set(state_cameras("robocasa365", "all_vision_missing")) == set(ROBOCASA)


def test_a_partial_combination_is_recognised_but_is_not_a_ranked_state():
    # Still executed and reported if someone runs it; simply outside a ranking
    # whose conditions have to mean the same thing on every platform.
    assert state_of("robocasa365", ["robot0_agentview_left"]) is None
    assert state_of("robocasa365", ROBOCASA) == "all_vision_missing"
    assert state_of("robocasa365", []) == "healthy"


def test_a_camera_outside_every_role_is_refused():
    # Silently leaving a camera out of the ranking is a design decision, not
    # something that should happen because a profile grew a fourth view.
    with pytest.raises(SemanticStateError, match="belong to no semantic role"):
        validate_platform("robocasa365", ROBOCASA + ["robot0_frontview"])
    with pytest.raises(SemanticStateError, match="does not provide"):
        validate_platform("robocasa365", ["robot0_eye_in_hand"])
    with pytest.raises(SemanticStateError, match="no semantic camera roles"):
        state_cameras("unknown_platform", "wrist_missing")


# -- scoring -----------------------------------------------------------------

def rows(state, onset, successes, evaluated):
    return ([{"state": state, "onset_fraction": onset, "success": True}] * successes
            + [{"state": state, "onset_fraction": onset, "success": False}]
            * (evaluated - successes))


def task(scenes, healthy, per_condition, task_name="t"):
    every = []
    for state, onset in MISSING_CONDITIONS:
        every.extend(rows(state, onset, round(per_condition * healthy), healthy))
    return TaskScore(task_name, scenes, healthy,
                     condition_scores(every, healthy_successes=healthy))


def test_the_task_score_is_ten_equal_components():
    # 50 scenes, 30 healthy successes, 18 of 30 surviving every fault.
    scored = task(50, 30, 0.6)
    assert scored.healthy_score == pytest.approx(0.6)
    assert all(c.score == pytest.approx(0.6) for c in scored.conditions)
    assert len(scored.conditions) == 9 and RANKING_COMPONENTS == 10
    assert scored.score == pytest.approx((0.6 + 9 * 0.6) / 10)


def test_the_missing_score_is_conditional_and_has_no_second_reading():
    scored = task(50, 30, 0.6)
    condition = scored.conditions[0]
    # Conditional on the policy being able to do the task at all.
    assert condition.written == 30
    assert condition.score == pytest.approx(18 / 30)
    # And that is the only reading of it. Publishing a second denominator for
    # one condition invites a reader to pick whichever supports the claim; the
    # healthy score already stands beside it saying how much of the suite the
    # policy can do at all.
    assert not hasattr(condition, "all_scene_score")
    assert condition.written == 30 and condition.valid == 30
    assert "all_scene_score" not in scored.as_record()["conditions"][condition.key]


def test_a_task_with_no_healthy_success_scores_zero_rather_than_vanishing():
    # Dropping it would let a policy raise its platform score by failing a task
    # completely.
    empty = TaskScore("t", 50, 0, condition_scores([], healthy_successes=0))
    assert all(c.score is None for c in empty.conditions)
    assert all(c.ranked_value == 0.0 for c in empty.conditions)
    assert empty.healthy_score == 0.0
    assert empty.score == 0.0


def test_a_missing_condition_cannot_silently_shorten_the_denominator():
    # Only two of the nine conditions were run; the rest must still be present
    # and count as zero, or the task score would be an average over a different
    # number of things than the protocol defines.
    partial = condition_scores(
        rows("wrist_missing", 0.30, 10, 10) + rows("all_vision_missing", 0.60, 0, 10),
        healthy_successes=10,
    )
    assert len(partial) == 9
    scored = TaskScore("t", 50, 10, partial)
    assert scored.score == pytest.approx((0.2 + 1.0 + 0.0) / 10)
    with pytest.raises(ScoringError, match="the ranking needs"):
        TaskScore("t", 50, 10, partial[:5]).score


def test_tasks_are_weighted_equally():
    # A task with more scenes must not weigh more: the benchmark score is an
    # average of task scores, not of rollouts.
    big = task(900, 900, 0.2, "big")
    small = task(10, 10, 1.0, "small")
    benchmark = BenchmarkScore((big, small), expected_tasks=2)
    assert benchmark.score == pytest.approx((big.score + small.score) / 2)
    pooled = (900 * big.score + 10 * small.score) / 910
    assert benchmark.score != pytest.approx(pooled)


def test_a_partial_task_set_is_an_experiment_not_a_submission():
    # An average over a subset of tasks is a different quantity; letting it
    # carry the benchmark's name would put two numbers meaning different things
    # in one column.
    partial = BenchmarkScore(tuple(task(50, 30, 0.6, f"t{i}") for i in range(5)))
    assert partial.official is False
    assert partial.as_record()["mail_bench_score"] is None
    with pytest.raises(ScoringError, match="may not carry"):
        partial.score
    full = BenchmarkScore(tuple(task(50, 30, 0.6, f"t{i}") for i in range(benchmark_task_count())))
    assert full.official is True
    assert full.score == pytest.approx((0.6 + 9 * 0.6) / 10)


def test_the_condition_denominator_is_healthy_successes_not_written_rollouts():
    """A fault cell that was owed and never produced must lower the score.

    Dividing by the rollouts that exist would let a condition whose cells are
    half missing score on the half that ran, and the gap would be invisible in
    the number.
    """
    half = condition_scores(rows("wrist_missing", 0.30, 10, 10),
                            healthy_successes=30)
    condition = half[0]
    assert condition.written == 10 and condition.healthy_successes == 30
    assert condition.score == pytest.approx(10 / 30)
    assert condition.complete is False
    whole = condition_scores(rows("wrist_missing", 0.30, 10, 30),
                             healthy_successes=30)
    assert whole[0].complete is True


def test_one_fixed_policy_configuration_across_every_task():
    # Eighteen specialists is a benchmark of eighteen specialists, which
    # measures something other than a policy.
    assert one_policy_configuration(["sha-a"] * 18) == "sha-a"
    with pytest.raises(ScoringError, match="not an official submission"):
        one_policy_configuration(["sha-a"] * 17 + ["sha-b"])
    with pytest.raises(ScoringError, match="no checkpoint identity"):
        one_policy_configuration([])
