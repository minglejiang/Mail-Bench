"""The whole chain, once: cells to a MAIL-Bench score.

This is the test the separate unit tests could not replace. Each of those
validates one layer against the protocol it was written for, so a suite of them
can be entirely green while two layers disagree.

So this walks the chain the protocol defines, on synthetic cells, and checks the
claims that only survive if the layers agree:

    18 tasks x (healthy + 9 semantic fault conditions)
        -> all_vision_missing is ranked
        -> M_c denominator is the healthy-success count
        -> ten equally weighted components make a task score
        -> one policy configuration across every task
        -> eighteen task scores make the benchmark score
"""

import json
from pathlib import Path

import pytest

from mail_bench.aggregate import AggregationError, mail_bench_report
from mail_bench.manifest import PROTOCOL_VERSION
from mail_bench.runner import EMPTY_PREFIX_HASH
from mail_bench.scoring import MISSING_CONDITIONS, RANKING_COMPONENTS, ScoringError
from mail_bench.semantic_states import state_cameras

PLATFORM = "robocasa365"
TASKS = tuple(f"Task{i:02d}" for i in range(18))
SCENES = 6                      # small, but every task has the full condition set
HEALTHY_OK = 4                  # scenes each task solves healthily
CHECKPOINT = "c" * 64


T_HEALTHY = 100                 # every synthetic healthy success completes here
HORIZON = 500


def prefix_chain(length, salt=""):
    """A runner-shaped action prefix chain: one digest per executed action."""
    return [f"{i:04d}{salt}".ljust(64, "a") for i in range(length)]


def onset_step(onset):
    """What the runner executes: half_up(f * T_healthy), as the driver freezes it."""
    from mail_bench.onset import resolve_onset
    return resolve_onset(onset, HORIZON, healthy_completion_step=T_HEALTHY,
                         healthy_success=True)["onset_step"]


def write_cell(root, name, *, task, episode, cameras, onset, success,
               checkpoint=CHECKPOINT, fault_cell_valid=True, duration="end", late=0,
               mode=None, steps=None, diverged=False, executed_onset=None,
               policy_seed=7):
    # The frozen cell id spells an empty camera set "none".
    faulted = "+".join(cameras) if cameras else "none"
    mode = mode or ("hard_missing" if cameras else "healthy")
    cell_id = (f"{task}|ep{episode}|{mode}|miss={faulted}|"
               f"onset={onset:g}|dur={duration}")
    # Runner-shaped: a healthy cell carries its whole chain and no onset; a
    # fault cell carries the digest of its prefix before the onset, which
    # equals the healthy chain at that step unless the policy diverged.
    target = (onset_step(onset) if executed_onset is None else executed_onset) if cameras else None
    steps = steps if steps is not None else (T_HEALTHY if success else HORIZON)
    chain = prefix_chain(steps, salt="x" if diverged else "")
    result = {
        "cell_id": cell_id,
        "method_id": "policy/under-test",
        "checkpoint_hash": checkpoint,
        "runner_config_hash": "r" * 64,
        "success_or_valid": success,
        "clean_or_fault": "clean" if not cameras else "fault",
        "fault_cell_valid": fault_cell_valid,
        "faulted_observations_consumed": 3 if fault_cell_valid else 0,
        "onset_step": target,
        "realized_onset_step": (target + late) if cameras else None,
        "action_prefix_hash_chain": chain,
        "pre_fault_action_hash": (
            None if not cameras else
            (chain[target - 1] if target > 0 else EMPTY_PREFIX_HASH)),
        "subset_ranking_eligible": True,
        "ranking_eligible": True,
        "onset_basis": "healthy_reference" if cameras else "not_applicable",
        "onset_fallback": False,
        "healthy_reference_success": bool(cameras),
        "action_execution_count": steps,
        "environment_step_count": steps,
        "method_config_hash": "f" * 64,
        "dataset_authorization_hash": "a" * 64,
        "official_metrics": {"task": task, "episode_index": episode,
                             "success": success, "official_horizon": HORIZON},
        "policy_seed": policy_seed,
    }
    (root / f"{name}.json").write_text(
        json.dumps({"manifest_hash": "m" * 64, "result": result}), encoding="utf-8")


def remove_cell(root, task, episode, state, onset):
    """Delete the one cell of this condition, and refuse if there is not exactly one.

    Exactly one cell must match, so the removal cannot depend on directory
    order.
    """
    faulted = "+".join(state_cameras(PLATFORM, state))
    prefix = f"{task}|ep{episode}|hard_missing|miss={faulted}|onset={onset:g}|"
    matches = [p for p in sorted((root / "cells").glob("*.json"))
               if json.loads(p.read_text(encoding="utf-8"))["result"]["cell_id"].startswith(prefix)]
    assert len(matches) == 1, f"expected one cell for {prefix}, found {len(matches)}"
    matches[0].unlink()


def build(root, *, checkpoint_for=lambda task: CHECKPOINT,
          fault_success=lambda task, state, onset, episode: episode < 2,
          skip_condition=None, tasks=TASKS):
    cells = root / "cells"
    cells.mkdir(parents=True)
    n = 0
    for task in tasks:
        checkpoint = checkpoint_for(task)
        for episode in range(SCENES):
            write_cell(cells, f"h{n}", task=task, episode=episode, cameras=(),
                       onset=0.0, success=episode < HEALTHY_OK, checkpoint=checkpoint)
            n += 1
        # Faults only where healthy succeeded, as the protocol requires.
        for state, onset in MISSING_CONDITIONS:
            if skip_condition is not None and skip_condition(task, state, onset):
                continue
            for episode in range(HEALTHY_OK):
                write_cell(cells, f"f{n}", task=task, episode=episode,
                           cameras=state_cameras(PLATFORM, state), onset=onset,
                           success=fault_success(task, state, onset, episode),
                           checkpoint=checkpoint)
                n += 1
    return root


def test_the_chain_holds_from_cells_to_a_benchmark_score(tmp_path):
    report = mail_bench_report(build(tmp_path), scenes_per_task=SCENES,
                               expected_tasks=TASKS, require_official=False)
    # A synthetic eighteen-task suite of six scenes exercises the chain; it is
    # deliberately not the frozen suite, so it is not official and the
    # benchmark's own name is withheld.
    assert report["official"] is False
    assert report["official_shape"] is False
    assert report["mail_bench_score"] is None
    assert report["tasks_evaluated"] == 18
    assert report["policy_configuration"] == CHECKPOINT
    assert report["cells_outside_main_ranking"] == 0

    task = report["tasks"][0]
    assert task["healthy_score"] == pytest.approx(HEALTHY_OK / SCENES)
    # Ten components, not nine and not seven.
    assert len(task["conditions"]) == RANKING_COMPONENTS - 1
    # Two of the four healthy-solved scenes survive every fault.
    for key, condition in task["conditions"].items():
        assert condition["written"] == HEALTHY_OK
        assert condition["owed"] == HEALTHY_OK
        assert condition["complete"] is True
        assert condition["score"] == pytest.approx(2 / HEALTHY_OK)
    expected = (HEALTHY_OK / SCENES + 9 * (2 / HEALTHY_OK)) / RANKING_COMPONENTS
    assert task["task_score"] == pytest.approx(expected)
    assert report["score"] == pytest.approx(expected)


def test_total_visual_loss_is_ranked_all_the_way_through(tmp_path):
    """Total visual loss is three of the ten conditions and must arrive as a
    scored condition, or the task score averages over the wrong number of
    things."""
    report = mail_bench_report(build(tmp_path), scenes_per_task=SCENES,
                               expected_tasks=TASKS,
                               require_official=False)
    conditions = report["tasks"][0]["conditions"]
    total_loss = [key for key in conditions if key.startswith("all_vision_missing")]
    assert len(total_loss) == 3
    for key in total_loss:
        assert conditions[key]["score"] is not None
        assert conditions[key]["written"] == HEALTHY_OK


def test_a_missing_condition_lowers_the_score_rather_than_vanishing(tmp_path):
    """A cell that was owed and never produced must cost something.

    Dividing by the rollouts that exist would let a condition whose cells are
    missing score on the ones that ran, and nothing in the number would show it.
    """
    def skip(task, state, onset):
        return task == TASKS[0] and state == "wrist_missing" and onset == 0.30

    report = mail_bench_report(build(tmp_path, skip_condition=skip),
                            scenes_per_task=SCENES, expected_tasks=TASKS,
                            require_official=False)
    starved = report["tasks"][0]["conditions"]["wrist_missing@0.30"]
    assert starved["written"] == 0 and starved["owed"] == HEALTHY_OK
    assert starved["complete"] is False
    assert starved["score"] == 0.0
    # And a task that lost a condition scores below one that kept it.
    assert report["tasks"][0]["task_score"] < report["tasks"][1]["task_score"]


def test_faults_are_only_scored_where_the_policy_solved_the_scene(tmp_path):
    report = mail_bench_report(build(tmp_path), scenes_per_task=SCENES,
                               expected_tasks=TASKS,
                               require_official=False)
    for task in report["tasks"]:
        assert task["healthy_successes"] == HEALTHY_OK
        for condition in task["conditions"].values():
            # Never the six scenes of the task; always the four it solved.
            assert condition["owed"] == HEALTHY_OK
            assert condition["owed"] != SCENES


def test_per_task_checkpoint_switching_is_not_a_submission(tmp_path):
    report = build(tmp_path, checkpoint_for=lambda task: "a" * 64 if task == TASKS[0]
                   else "b" * 64)
    with pytest.raises(Exception, match="measurement identities"):
        mail_bench_report(report, scenes_per_task=SCENES, expected_tasks=TASKS,
                               require_official=False)


def test_a_partial_task_set_cannot_be_an_official_score(tmp_path):
    root = build(tmp_path, tasks=TASKS[:5])
    with pytest.raises(AggregationError, match="experiment rather than an official"):
        mail_bench_report(root, scenes_per_task=SCENES, expected_tasks=TASKS)
    record = mail_bench_report(root, scenes_per_task=SCENES,
                               expected_tasks=TASKS, require_official=False)
    assert record["official"] is False
    assert record["mail_bench_score"] is None


def test_a_diagnostic_camera_combination_never_reaches_the_ranking(tmp_path):
    # One agentview alone has no counterpart on a platform whose role is served
    # by one camera, so it is counted apart rather than folded into the
    # agentview condition it resembles.
    root = build(tmp_path)
    write_cell(root / "cells", "diagnostic", task=TASKS[0], episode=0,
               cameras=("robot0_agentview_left",), onset=0.45, success=True)
    report = mail_bench_report(root, scenes_per_task=SCENES, expected_tasks=TASKS,
                               require_official=False)
    assert report["cells_outside_main_ranking"] == 1
    assert report["tasks"][0]["conditions"]["agentview_missing@0.45"]["written"] == HEALTHY_OK


def test_the_realized_onset_is_observed_rather_than_predicted():
    """Two different things, and only one of them is a measurement.

    The onset manifest is written from the healthy phase, before any fault has
    run, so the boundary it gives is a prediction from a declared replanning
    interval. A policy queries when its action buffer empties, which need not
    land on a multiple of that interval, so the step at which the fault actually
    reached the policy has to come from the rollout.
    """
    import inspect

    from mail_bench.onset import onset_manifest_row
    from mail_bench.runner import RolloutAudit

    row = onset_manifest_row(
        model_id="m", checkpoint_identity="a" * 12, scene_identity="s",
        healthy_success=True, healthy_completion_step=200,
        official_horizon=500, query_interval=8,
    )
    # The manifest says what it is.
    assert "onset_30_realized_predicted" in row
    assert "onset_30_realized" not in row
    # And the rollout carries the observed one.
    assert "realized_onset_step" in inspect.signature(RolloutAudit).parameters or \
        hasattr(RolloutAudit(), "realized_onset_step")


def test_a_success_the_fault_never_reached_is_not_a_success(tmp_path):
    """A success the fault never reached is not a success.

    A scene solved in one environment step leaves a 60% onset at step 1, the
    episode is already over, and the policy never consumes a faulted
    observation. The runner records that -- fault_cell_valid=false,
    faulted_observations_consumed=0 -- and the scorer must respect it, because a
    task finished before the intervention arrived says nothing about robustness.

    The denominator does not shrink to hide it. The cell is written, invalid,
    and scores zero out of the task's healthy-success count.
    """
    root = build(tmp_path)
    cells = root / "cells"
    # Replace one condition's cells for one task with a single zero-exposure
    # success, as the real scene produced.
    for path in list(cells.glob("*.json")):
        result = json.loads(path.read_text(encoding="utf-8"))["result"]
        if (result["official_metrics"]["task"] == TASKS[0]
                and "miss=robot0_eye_in_hand|onset=0.3|" in result["cell_id"]):
            path.unlink()
    write_cell(cells, "zero_exposure", task=TASKS[0], episode=0,
               cameras=state_cameras(PLATFORM, "wrist_missing"), onset=0.30,
               success=True, fault_cell_valid=False)

    report = mail_bench_report(root, scenes_per_task=SCENES, expected_tasks=TASKS,
                               require_official=False)
    condition = report["tasks"][0]["conditions"]["wrist_missing@0.30"]
    assert condition["written"] == 1
    assert condition["valid"] == 0
    assert condition["invalid"] == 1
    assert condition["owed"] == HEALTHY_OK
    assert condition["successes"] == 0
    assert condition["score"] == 0.0
    assert report["zero_exposure_fault_cells"] == 1


def test_a_zero_exposure_cell_still_counts_as_executed(tmp_path):
    """Completeness is coverage of execution, not of exposure.

    A scene whose healthy rollout finishes in one step leaves a late onset with
    no episode to act on, however many times it is run. Requiring exposure for
    completeness would make the benchmark permanently unfinishable for a reason
    no submission can fix, so the invalid count is reported instead.
    """
    from mail_bench.scoring import condition_scores

    rows = [{"state": "wrist_missing", "onset_fraction": 0.30,
             "success": True, "valid": False}]
    condition = condition_scores(rows, healthy_successes=1)[0]
    assert condition.written == 1 and condition.valid == 0
    assert condition.complete is True          # it ran
    assert condition.score == 0.0              # it proved nothing


def test_a_mechanism_cell_cannot_reach_the_main_ranking(tmp_path):
    """Blackout at 60% shares a state and an onset with a ranking condition.

    Matching on state and onset alone would let a Failure Form study move a
    leaderboard score. The ranking is hard missing, until the episode ends.
    """
    root = build(tmp_path)
    baseline = mail_bench_report(root, scenes_per_task=SCENES, expected_tasks=TASKS,
                               require_official=False)
    for name, kwargs in (
        ("blackout", {"mode": "blackout"}),
        ("shortdur", {"duration": "0.25R"}),
    ):
        write_cell(root / "cells", name, task=TASKS[0], episode=0,
                   cameras=state_cameras(PLATFORM, "wrist_missing"), onset=0.60,
                   success=True, **kwargs)
    after = mail_bench_report(root, scenes_per_task=SCENES, expected_tasks=TASKS,
                               require_official=False)
    assert after["cells_outside_main_ranking"] == 2
    assert after["score"] == baseline["score"]
    assert (after["tasks"][0]["conditions"]["wrist_missing@0.60"]["written"]
            == baseline["tasks"][0]["conditions"]["wrist_missing@0.60"]["written"])


def test_one_checkpoint_under_two_runner_configs_is_two_measurements(tmp_path):
    """Same weights, different replanning profile, different evaluated policy.

    Checking the checkpoint alone would pass this.
    """
    root = build(tmp_path)
    path = sorted((root / "cells").glob("*.json"))[0]
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["result"]["runner_config_hash"] = "9" * 64
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(AggregationError, match="measurement identities"):
        mail_bench_report(root, scenes_per_task=SCENES, expected_tasks=TASKS,
                               require_official=False)


def test_eighteen_task_names_are_not_the_suite(tmp_path):
    """A task count is not a task set."""
    wrong = tuple(f"NotATask{i:02d}" for i in range(18))
    root = build(tmp_path, tasks=wrong)
    with pytest.raises(AggregationError, match="exact task set"):
        mail_bench_report(root, scenes_per_task=SCENES, expected_tasks=TASKS)
    record = mail_bench_report(root, scenes_per_task=SCENES, expected_tasks=TASKS,
                               require_official=False)
    assert record["task_set_exact"] is False
    assert record["official"] is False and record["mail_bench_score"] is None


def test_the_frozen_suite_cannot_be_redefined_by_argument(tmp_path):
    """Official means the frozen suite, not a set of parameters.

    Naming five tasks of six scenes and having them complete would otherwise
    produce official=True, which would let anyone redefine the benchmark by
    argument.
    """
    from mail_bench.suite import scenes_per_task, task_ids

    record = mail_bench_report(build(tmp_path), scenes_per_task=SCENES,
                               expected_tasks=TASKS, require_official=False)
    assert record["official_shape"] is False       # synthetic tasks, six scenes
    assert record["official"] is False
    assert len(task_ids()) == 18 and scenes_per_task() == 50


def test_an_unverifiable_task_set_is_not_a_verified_one(tmp_path):
    # Passing no expected tasks must not make task_set_exact true by default: a
    # run that cannot check the suite must not report that it did.
    record = mail_bench_report(build(tmp_path), scenes_per_task=SCENES,
                               require_official=False)
    assert record["task_set_verified"] is False
    assert record["task_set_exact"] is False
    assert record["official"] is False


def test_the_fault_set_must_be_the_healthy_success_set(tmp_path):
    """Same size is not the same scenes.

    A task solved on episodes {1,3,5,7} whose faults ran on {1,2,4,7} has four
    written against four owed and would have counted as complete, though every
    fault but two is paired with a scene the policy never solved.
    """
    from mail_bench.scoring import condition_scores

    rows = [{"state": "wrist_missing", "onset_fraction": 0.30, "success": True,
             "valid": True, "episode_index": e} for e in (1, 2, 4, 7)]
    wrong = condition_scores(rows, healthy_successes=4,
                             healthy_episodes=[1, 3, 5, 7])[0]
    # Only the owed scenes are counted; the strays are named, not scored.
    assert wrong.written == 2 and wrong.healthy_successes == 4
    assert wrong.missing_episodes == (3, 5) and wrong.extra_episodes == (2, 4)
    assert wrong.episodes_match_healthy is False
    assert wrong.complete is False
    right = condition_scores(rows, healthy_successes=4,
                             healthy_episodes=[1, 2, 4, 7])[0]
    assert right.complete is True


def test_the_reporting_command_runs(tmp_path):
    """The command-line entry point is exercised, not only the library function."""
    import subprocess
    import sys

    root = build(tmp_path)
    finished = subprocess.run(
        [sys.executable,
         str(Path(__file__).resolve().parents[1] / "scripts"
             / "report_mail_bench_score.py"),
         "--run-root", str(root), "--scenes-per-task", str(SCENES),
         "--expected-tasks", *TASKS, "--allow-experiment"],
        capture_output=True, text=True,
    )
    assert finished.returncode == 0, finished.stderr[-800:]
    assert "MAIL-Bench score" in finished.stdout
    assert "task set              exact" in finished.stdout
    # A synthetic suite is an experiment, and the command says so rather than
    # printing a benchmark score.
    assert "official shape        NO" in finished.stdout


def test_pre_fault_divergence_is_reported_as_provenance(tmp_path):
    """The pairing rule tolerates a stochastic policy; the report says how much.

    The comparison is the fault cell's pre-onset digest against the healthy
    cell's chain *at that onset* -- a healthy cell has no onset of its own and
    carries no pre-fault digest, so comparing the two same-named fields would
    call every real cell divergent.
    """
    root = build(tmp_path)
    cells = root / "cells"
    # One fault cell whose policy took a different path before the onset.
    remove_cell(root, TASKS[0], 0, "wrist_missing", 0.30)
    write_cell(cells, "diverged", task=TASKS[0], episode=0,
               cameras=state_cameras(PLATFORM, "wrist_missing"), onset=0.30,
               success=True, diverged=True)
    report = mail_bench_report(root, scenes_per_task=SCENES, expected_tasks=TASKS,
                               require_official=False)
    pft = report["pre_fault_trajectory"]
    assert pft["not_comparable"] == 0
    assert pft["fault_cells_with_reference"] == 18 * HEALTHY_OK * 9
    assert pft["diverged_before_fault"] == 1
    assert pft["divergence_rate"] == pytest.approx(1 / (18 * HEALTHY_OK * 9))
    # Divergence is provenance, not invalidity: the score is unaffected by it.
    assert report["score"] is not None


def test_a_healthy_cell_carries_no_pre_fault_digest_and_still_pairs(tmp_path):
    """Runner-shaped healthy cells have pre_fault_action_hash = null."""
    root = build(tmp_path)
    for path in (root / "cells").glob("*.json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload["result"]["clean_or_fault"] == "clean":
            assert payload["result"]["pre_fault_action_hash"] is None
    report = mail_bench_report(root, scenes_per_task=SCENES, expected_tasks=TASKS,
                               require_official=False)
    assert report["pre_fault_trajectory"]["diverged_before_fault"] == 0
    assert report["pre_fault_trajectory"]["identical_to_healthy"] == 18 * HEALTHY_OK * 9


def test_the_executed_onset_is_checked_against_the_healthy_reference(tmp_path):
    """half_up(f * T_healthy) is recomputed from the paired healthy cell; a cell
    that executed another step is a manifest/driver disagreement, not official."""
    root = build(tmp_path)
    report = mail_bench_report(root, scenes_per_task=SCENES, expected_tasks=TASKS,
                               require_official=False)
    assert report["onset_check"]["fault_cells_checked"] == 18 * HEALTHY_OK * 9
    assert report["onset_check"]["mismatched"] == 0
    # Remove first: writing the replacement first would leave two cells of this
    # condition and the removal would pick one of them by directory order.
    remove_cell(root, TASKS[1], 1, "agentview_missing", 0.45)
    write_cell(root / "cells", "drifted", task=TASKS[1], episode=1,
               cameras=state_cameras(PLATFORM, "agentview_missing"), onset=0.45,
               success=True, executed_onset=onset_step(0.45) + 1, policy_seed=8)
    report = mail_bench_report(root, scenes_per_task=SCENES, expected_tasks=TASKS,
                               require_official=False)
    assert report["onset_check"]["mismatched"] == 1
    assert report["official"] is False


def test_completion_efficiency_is_paired_and_never_over_failures(tmp_path):
    """dT and S_slow over scenes where both rollouts succeeded, as medians of the
    per-scene pairs; a failed rollout is never given the horizon as a time."""
    def slow(task, state, onset, episode):
        return episode < 2
    root = tmp_path / "root"
    cells = root / "cells"; cells.mkdir(parents=True)
    n = 0
    for task in TASKS:
        for episode in range(SCENES):
            write_cell(cells, f"h{n}", task=task, episode=episode, cameras=(),
                       onset=0.0, success=episode < HEALTHY_OK); n += 1
        for state, onset in MISSING_CONDITIONS:
            for episode in range(HEALTHY_OK):
                ok = episode < 2
                # The surviving scenes finish 20 and 40 steps later than healthy.
                write_cell(cells, f"f{n}", task=task, episode=episode,
                           cameras=state_cameras(PLATFORM, state), onset=onset,
                           success=ok, steps=(T_HEALTHY + 20 * (episode + 1)) if ok else HORIZON)
                n += 1
    report = mail_bench_report(root, scenes_per_task=SCENES, expected_tasks=TASKS,
                               require_official=False)
    eff = report["completion_efficiency"]
    assert eff["pairs"] == 18 * 9 * 2
    assert eff["median_delta_steps"] == pytest.approx(30.0)
    assert eff["median_relative_slowdown"] == pytest.approx(0.30)
    condition = report["tasks"][0]["conditions"]["wrist_missing@0.30"]["completion_efficiency"]
    assert condition["pairs"] == 2 and condition["median_delta_steps"] == pytest.approx(30.0)


def test_a_fault_success_on_an_unsolved_scene_cannot_raise_the_score(tmp_path):
    """M_c is over the owed scenes only; a stray success elsewhere is named, not counted."""
    root = build(tmp_path)
    write_cell(root / "cells", "stray", task=TASKS[0], episode=HEALTHY_OK + 1,
               cameras=state_cameras(PLATFORM, "wrist_missing"), onset=0.30, success=True)
    report = mail_bench_report(root, scenes_per_task=SCENES, expected_tasks=TASKS,
                               require_official=False)
    condition = report["tasks"][0]["conditions"]["wrist_missing@0.30"]
    assert condition["score"] == pytest.approx(2 / HEALTHY_OK)
    assert condition["extra_episodes"] == [HEALTHY_OK + 1]
    assert condition["complete"] is False
    assert report["fault_coverage_complete"] is False


def test_two_fault_cells_for_one_owed_scene_are_refused(tmp_path):
    root = build(tmp_path)
    # A second cell for one owed scene under another policy seed: the reader's
    # semantic-key dedup lets it through, the scorer must not count it.
    write_cell(root / "cells", "twin", task=TASKS[0], episode=0,
               cameras=state_cameras(PLATFORM, "wrist_missing"), onset=0.30, success=True,
               diverged=True, policy_seed=99)
    with pytest.raises(ScoringError, match="two fault cells"):
        mail_bench_report(root, scenes_per_task=SCENES, expected_tasks=TASKS,
                          require_official=False)


def test_a_cell_without_an_exposure_verdict_is_an_audit_failure(tmp_path):
    root = build(tmp_path)
    path = next((root / "cells").glob("f*.json"))
    payload = json.loads(path.read_text(encoding="utf-8"))
    del payload["result"]["fault_cell_valid"]
    path.write_text(json.dumps(payload), encoding="utf-8")
    report = mail_bench_report(root, scenes_per_task=SCENES, expected_tasks=TASKS,
                               require_official=False)
    assert "fault_cell_valid_missing" in report["audit_failures"]
    assert report["zero_exposure_fault_cells"] == 0
    assert report["official"] is False


def certificates(root, *, eligible=True, deviations=()):
    (root / "certificate_healthy_worker0.json").write_text(json.dumps(
        {"phase": "healthy", "healthy_phase_complete_for_worker": True}), encoding="utf-8")
    (root / "certificate_fault_worker0.json").write_text(json.dumps(
        {"phase": "fault", "leaderboard_eligible": eligible,
         "deviations": list(deviations)}), encoding="utf-8")


def test_the_official_verdict_reads_the_runs_own_certificates(tmp_path):
    """Cells cannot tell an official run from a custom one; the certificate can."""
    root = build(tmp_path)
    report = mail_bench_report(root, scenes_per_task=SCENES, expected_tasks=TASKS,
                               require_official=False)
    assert report["certificates"]["leaderboard_eligible"] is False
    assert "no certificate under the run root" in report["certificates"]["problems"]
    certificates(root, eligible=False, deviations=["no_official_profile_declared"])
    report = mail_bench_report(root, scenes_per_task=SCENES, expected_tasks=TASKS,
                               require_official=False)
    assert report["certificates"]["deviations"] == ["no_official_profile_declared"]
    assert report["certificates"]["leaderboard_eligible"] is False
    assert report["official"] is False
    certificates(root)
    report = mail_bench_report(root, scenes_per_task=SCENES, expected_tasks=TASKS,
                               require_official=False)
    assert report["certificates"]["leaderboard_eligible"] is True


def test_the_realized_onset_is_reported_beside_the_target(tmp_path):
    """The fault is scheduled for a step and observed at the first query that
    carried it. The report counts how often those differed; it never corrects
    a cell for it."""
    root = build(tmp_path)
    report = mail_bench_report(root, scenes_per_task=SCENES, expected_tasks=TASKS,
                               require_official=False)
    realization = report["onset_realization"]
    assert realization["fault_cells"] > 0
    assert realization["realized_after_target"] == 0
    # One cell whose first faulted query came four steps after the target.
    import json
    from mail_bench.aggregate import load_cells
    rows, _ = load_cells(root)
    late = Path(next(row.source for row in rows if not row.is_healthy))
    payload = json.loads(late.read_text())
    payload["result"]["realized_onset_step"] = payload["result"]["onset_step"] + 4
    late.write_text(json.dumps(payload))
    report = mail_bench_report(root, scenes_per_task=SCENES, expected_tasks=TASKS,
                               require_official=False)
    assert report["onset_realization"]["realized_after_target"] == 1



# ---------------------------------------------------------------------------
# The official path, once, end to end
#
# A scorer that refused everything would pass the other tests; this one
# requires that a conforming run is accepted: a run shaped exactly like the
# frozen suite, with certificates, scores as official and carries the
# benchmark's own name.
# ---------------------------------------------------------------------------

OFFICIAL_SOLVED = 3          # healthy successes per task; the rest are failures


def build_official(root, *, phase_layout=("healthy", "fault"), eligible=True,
                   deviations=(), solved=OFFICIAL_SOLVED):
    """A run with the frozen suite's shape: every task, every scene, every owed cell."""
    from mail_bench.suite import scenes_per_task, task_ids

    tasks, scenes = list(task_ids()), scenes_per_task()
    cells = root / "cells"
    cells.mkdir(parents=True)
    n = 0
    for task in tasks:
        for episode in range(scenes):
            write_cell(cells, f"h{n}", task=task, episode=episode, cameras=(),
                       onset=0.0, success=episode < solved)
            n += 1
        for state, onset in MISSING_CONDITIONS:
            for episode in range(solved):
                write_cell(cells, f"f{n}", task=task, episode=episode,
                           cameras=state_cameras(PLATFORM, state), onset=onset,
                           success=episode == 0)
                n += 1
    for index, phase in enumerate(phase_layout):
        certificate = {"phase": phase}
        if phase in ("healthy", "all"):
            certificate["healthy_phase_complete_for_worker"] = True
        if phase in ("fault", "all"):
            certificate["leaderboard_eligible"] = eligible
            certificate["deviations"] = list(deviations)
        (root / f"certificate_{phase}_worker{index}.json").write_text(
            json.dumps(certificate), encoding="utf-8")
    return root, tasks, scenes


def test_a_conforming_run_is_official_and_carries_the_benchmarks_name(tmp_path):
    root, tasks, scenes = build_official(tmp_path)
    report = mail_bench_report(root, expected_tasks=tasks, scenes_per_task=scenes)
    assert report["official"] is True
    assert report["official_shape"] is True
    assert report["healthy_phase_complete"] is True
    assert report["fault_coverage_complete"] is True
    assert report["audit_failures"] == []
    assert report["certificates"]["leaderboard_eligible"] is True
    assert report["onset_check"]["mismatched"] == 0
    assert report["pre_fault_trajectory"]["diverged_before_fault"] == 0
    # One of three solved scenes survives every condition, so M_c = 1/3 for all
    # nine and the task score is the equal average of H and those nine.
    expected_task = (OFFICIAL_SOLVED / scenes
                     + 9 * (1 / OFFICIAL_SOLVED)) / RANKING_COMPONENTS
    assert report["mail_bench_score"] == pytest.approx(expected_task)
    assert report["score"] == report["mail_bench_score"]



def test_one_invocation_that_ran_both_phases_is_still_official(tmp_path):
    """``--phase all`` writes a single certificate covering both phases.

    It is the script's default, so a scorer that demanded a separate healthy
    certificate would refuse the most ordinary conforming run there is.
    """
    root, tasks, scenes = build_official(tmp_path, phase_layout=("all",))
    report = mail_bench_report(root, expected_tasks=tasks, scenes_per_task=scenes)
    assert report["certificates"]["healthy_certificates"] == 1
    assert report["certificates"]["fault_certificates"] == 1
    assert report["official"] is True


def test_an_official_shaped_run_is_refused_when_its_certificate_says_so(tmp_path):
    root, tasks, scenes = build_official(
        tmp_path, eligible=False, deviations=["no_official_profile_declared"])
    with pytest.raises(AggregationError, match="leaderboard-eligible certificates"):
        mail_bench_report(root, expected_tasks=tasks, scenes_per_task=scenes)
    report = mail_bench_report(root, expected_tasks=tasks, scenes_per_task=scenes,
                               require_official=False)
    assert report["official"] is False
    assert report["certificates"]["deviations"] == ["no_official_profile_declared"]
    assert report["mail_bench_score"] is None


def test_a_run_scored_before_its_fault_phase_is_not_official(tmp_path):
    root, tasks, scenes = build_official(tmp_path, phase_layout=("healthy",))
    report = mail_bench_report(root, expected_tasks=tasks, scenes_per_task=scenes,
                               require_official=False)
    assert report["official"] is False
    assert any("fault phase" in problem for problem in report["certificates"]["problems"])


def test_a_certificate_this_reader_does_not_understand_is_a_problem(tmp_path):
    root, tasks, scenes = build_official(tmp_path)
    (root / "certificate_healthy_worker0.json").write_text(
        json.dumps({"phase": "probe", "leaderboard_eligible": True}), encoding="utf-8")
    report = mail_bench_report(root, expected_tasks=tasks, scenes_per_task=scenes,
                               require_official=False)
    assert report["official"] is False
    assert any("unknown phase" in problem for problem in report["certificates"]["problems"])


def test_the_paper_row_decomposes_the_score_it_reports(tmp_path):
    """The table's columns are a projection of the record, not a second reading.

    A row is only honest if H and the nine conditions average back into the
    score beside them; the script refuses to print one that does not, and this
    pins the arithmetic against a real record.
    """
    import subprocess
    import sys

    root, tasks, scenes = build_official(tmp_path)
    record = mail_bench_report(root, expected_tasks=tasks, scenes_per_task=scenes)
    path = tmp_path / "record.json"
    path.write_text(json.dumps(record), encoding="utf-8")
    finished = subprocess.run(
        [sys.executable,
         str(Path(__file__).resolve().parents[1] / "scripts" / "paper_results_table.py"),
         "--record", f"TestPolicy={path}", "--reported", "TestPolicy=0.157"],
        capture_output=True, text=True,
    )
    assert finished.returncode == 0, finished.stderr[-800:]
    cells = [c.strip() for c in finished.stdout.strip().rstrip("\\").split("&")]
    assert cells[0] == "TestPolicy" and cells[1] == "0.157"
    assert cells[2] == f"{OFFICIAL_SOLVED / scenes:.3f}"          # H
    assert cells[3:12] == [f"{1 / OFFICIAL_SOLVED:.3f}"] * 9      # the nine M_c
    assert cells[12] == f"{record['mail_bench_score']:.3f}"
    # The script enforces the decomposition on the unrounded values and exits
    # non-zero otherwise, so returning 0 above is the real assertion; the
    # printed cells reconstruct it to their own display precision.
    assert (float(cells[2]) + sum(float(c) for c in cells[3:12])) / RANKING_COMPONENTS \
        == pytest.approx(record["mail_bench_score"], abs=1e-3)


def test_a_row_that_does_not_decompose_its_score_is_refused(tmp_path):
    """A record whose score disagrees with its own tasks must not become a row."""
    import subprocess
    import sys

    root, tasks, scenes = build_official(tmp_path)
    record = mail_bench_report(root, expected_tasks=tasks, scenes_per_task=scenes)
    record["score"] = record["mail_bench_score"] + 0.05
    path = tmp_path / "tampered.json"
    path.write_text(json.dumps(record), encoding="utf-8")
    finished = subprocess.run(
        [sys.executable,
         str(Path(__file__).resolve().parents[1] / "scripts" / "paper_results_table.py"),
         "--record", f"TestPolicy={path}"],
        capture_output=True, text=True,
    )
    assert finished.returncode != 0
    assert "does not decompose" in finished.stderr
    assert finished.stdout.strip() == ""


def _run_script(name, *args):
    import subprocess
    import sys

    return subprocess.run(
        [sys.executable, str(Path(__file__).resolve().parents[1] / "scripts" / name), *args],
        capture_output=True, text=True,
    )


def test_the_healthy_report_command_runs_on_a_conforming_run(tmp_path):
    """Step 3 of the official pipeline, exercised as the command a user runs."""
    root, tasks, scenes = build_official(tmp_path / "run")
    out = tmp_path / "healthy_report.json"
    finished = _run_script("report_healthy_phase.py", "--run-root", str(root), "--out", str(out))
    assert finished.returncode == 0, finished.stdout[-800:] + finished.stderr[-800:]
    assert "HEALTHY_PHASE_COMPLETE" in finished.stdout
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["integrity"]["complete"] is True
    assert report["integrity"]["n_unique"] == len(tasks) * scenes
    assert report["healthy_successes"] == len(tasks) * OFFICIAL_SOLVED
    assert report["clean_success"] == pytest.approx(OFFICIAL_SOLVED / scenes)
    assert set(report["per_task"]) == set(tasks)
    assert report["per_task"][tasks[0]]["solved"] == OFFICIAL_SOLVED


def test_the_healthy_report_command_refuses_an_incomplete_phase(tmp_path):
    root, tasks, scenes = build_official(tmp_path / "run")
    remove_cell(root, tasks[0], 0, MISSING_CONDITIONS[0][0], MISSING_CONDITIONS[0][1])
    victim = next(p for p in sorted((root / "cells").glob("h*.json")))
    victim.unlink()
    finished = _run_script("report_healthy_phase.py", "--run-root", str(root))
    assert finished.returncode == 1
    assert "INCOMPLETE" in finished.stdout


def test_the_onset_manifest_command_runs_on_a_conforming_run(tmp_path):
    """Step 4 of the official pipeline: the manifest is derived from the cells
    and the certificate, never from what the operator asserts."""
    root, tasks, scenes = build_official(tmp_path / "run")
    certificate = root / "certificate_healthy_worker0.json"
    payload = json.loads(certificate.read_text(encoding="utf-8"))
    payload["cohort_specification"] = {"protocol_version": PROTOCOL_VERSION}
    payload["runtime"] = {"scene_namespace": "official"}
    payload["execution_profile"] = {"name": "native", "max_actions_per_query": 16}
    certificate.write_text(json.dumps(payload), encoding="utf-8")
    for cell in (root / "cells").glob("*.json"):
        record = json.loads(cell.read_text(encoding="utf-8"))
        record["result"]["official_metrics"]["scene_identity"] = "s" * 64
        cell.write_text(json.dumps(record), encoding="utf-8")
    out = tmp_path / "onsets.json"
    finished = _run_script("build_onset_manifest.py", "--run-root", str(root),
                           "--model-id", "policy-under-test",
                           "--checkpoint-identity", CHECKPOINT[:12], "--out", str(out))
    assert finished.returncode == 0, finished.stdout[-800:] + finished.stderr[-800:]
    assert "ONSET_MANIFEST_DONE" in finished.stdout
    manifest = json.loads(out.read_text(encoding="utf-8"))
    assert manifest["protocol_version"] == PROTOCOL_VERSION
    assert manifest["checkpoint_identity"] == CHECKPOINT
    assert manifest["query_interval"] == 16          # read from the certificate, not typed
    assert manifest["scenes"] == len(tasks) * scenes
    assert manifest["fault_evaluable_scenes"] == len(tasks) * OFFICIAL_SOLVED
    assert manifest["skipped"] == {"healthy_unsolved": len(tasks) * (scenes - OFFICIAL_SOLVED)}
    solved_row = next(r for r in manifest["rows"] if r["healthy_success"])
    assert solved_row["healthy_completion_step"] == T_HEALTHY
    assert solved_row["fault_evaluable"] is True
    assert solved_row["onset_45_target"] == round(0.45 * T_HEALTHY)

    # A flag that contradicts the certificate is refused, as is the wrong checkpoint
    # prefix and a certificate under another protocol.
    refused = _run_script("build_onset_manifest.py", "--run-root", str(root),
                          "--model-id", "policy-under-test", "--replan", "5", "--out", str(out))
    assert refused.returncode != 0 and "actions per query" in refused.stderr
    refused = _run_script("build_onset_manifest.py", "--run-root", str(root),
                          "--model-id", "policy-under-test",
                          "--checkpoint-identity", "deadbeef", "--out", str(out))
    assert refused.returncode != 0 and "checkpoint-identity" in refused.stderr
    payload["cohort_specification"] = {"protocol_version": "another_protocol"}
    certificate.write_text(json.dumps(payload), encoding="utf-8")
    refused = _run_script("build_onset_manifest.py", "--run-root", str(root),
                          "--model-id", "policy-under-test", "--out", str(out))
    assert refused.returncode != 0 and "protocol" in refused.stderr
