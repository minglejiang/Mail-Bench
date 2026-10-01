#!/usr/bin/env python3
"""The MAIL-Bench score for one run, from its standard cells.

The last step of the pipeline, as a command rather than something typed at a
server: healthy cells give each task its healthy score and the scenes a fault
may be placed in, fault cells are grouped into the nine conditions, ten make a
task score and eighteen make the benchmark score.

It refuses to hand back an official score for a partial task set or for a run
that used more than one policy configuration. Both are legitimate experiments
and --allow-experiment prints the record for them; neither is a submission.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mail_bench.aggregate import mail_bench_report                 # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-root", type=Path, required=True,
                        help="the run root, not its cells/ directory")
    parser.add_argument("--platform", default="robocasa365")
    parser.add_argument("--expected-tasks", nargs="+",
                        help="the exact task names an official run must cover; "
                             "defaults to the frozen suite")
    parser.add_argument("--scenes-per-task", type=int, default=None,
                        help="testing only; an official score requires the frozen "
                             "suite's fifty")
    parser.add_argument("--allow-experiment", action="store_true",
                        help="print the record for a partial run instead of refusing")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    # The suite is frozen in this repository, not read from whatever RoboCasa is
    # installed: a benchmark whose task identity depends on a package version is
    # immutable only until somebody upgrades.
    from mail_bench.suite import scenes_per_task as frozen_scenes, task_ids

    expected = args.expected_tasks or list(task_ids())
    scenes = args.scenes_per_task if args.scenes_per_task is not None else frozen_scenes()

    record = mail_bench_report(
        args.run_root, platform=args.platform,
        expected_tasks=expected, scenes_per_task=scenes,
        require_official=not args.allow_experiment,
    )

    print(f"policy configuration  {record['policy_configuration'][:16]}")
    print(f"tasks                 {record['tasks_evaluated']}/{record['tasks_expected']}")
    print(f"healthy phase         "
          f"{'complete' if record['healthy_phase_complete'] else 'INCOMPLETE'}")
    print("task set              " + (
        "exact" if record["task_set_exact"]
        else "NOT VERIFIED" if not record["task_set_verified"]
        else "NOT THE SUITE"))
    if not record["official_shape"]:
        print("official shape        NO (platform, task set or scene count is "
              "not the frozen suite)")
    print(f"fault coverage        "
          f"{'complete' if record['fault_coverage_complete'] else 'INCOMPLETE'}")
    if record["cells_outside_main_ranking"]:
        print(f"outside main ranking  {record['cells_outside_main_ranking']} "
              "(mechanism or diagnostic cells, never scored here)")
    pft = record["pre_fault_trajectory"]
    if pft["fault_cells_with_reference"]:
        print(f"pre-fault divergence  {pft['diverged_before_fault']}/"
              f"{pft['fault_cells_with_reference']} fault cells took a different path "
              f"than their healthy run before the fault "
              f"({pft['divergence_rate']:.1%}; a stochastic policy, tolerated by the pairing rule)")
    if record["zero_exposure_fault_cells"]:
        print(f"zero-exposure cells   {record['zero_exposure_fault_cells']} "
              "(ran, but the fault never reached the policy; scored as failures)")
    if record["audit_failures"]:
        print(f"audit failures        {record['audit_failures']}")
    check = record["onset_check"]
    if check["fault_cells_checked"]:
        print(f"onset check           {check['fault_cells_checked'] - check['mismatched']}/"
              f"{check['fault_cells_checked']} fault cells executed the onset their "
              f"healthy reference implies"
              + (f"; MISMATCHED {check['mismatched_cells'][:3]}" if check["mismatched"] else ""))
    certs = record["certificates"]
    print(f"certificates          {certs['healthy_certificates']} healthy, "
          f"{certs['fault_certificates']} fault; "
          + ("leaderboard-eligible" if certs["leaderboard_eligible"]
             else "NOT ELIGIBLE: " + "; ".join(certs["problems"][:3])))
    if certs["deviations"]:
        print(f"deviations            {certs['deviations']}")
    efficiency = record["completion_efficiency"]
    if efficiency["pairs"]:
        print(f"completion efficiency {efficiency['pairs']} scene pairs where both rollouts "
              f"succeeded: median dT {efficiency['median_delta_steps']:+.0f} steps, "
              f"median S_slow {efficiency['median_relative_slowdown']:+.1%} "
              "(secondary; never over failed episodes)")

    # n is printed beside every task because it is the denominator of all nine
    # of its condition scores. A task the policy solved once has an M_c of
    # exactly 0.00 or 1.00, and without n a reader would take the 1.00 for
    # robustness rather than for a single rollout.
    if not record["tasks"]:
        raise SystemExit("no task has a healthy cell; there is nothing to score")
    print(f"\n{'task':30s} {'H':>6} {'n':>4}  " + "  ".join(
        f"{key.split('@')[0][:4]}{key.split('@')[1][2:]}" for key in
        sorted(record["tasks"][0]["conditions"])) + f"  {'S_t':>6}")
    thin = []
    for task in record["tasks"]:
        # A condition with no denominator has no score; it is not 0.00.
        scores = "  ".join(
            f"{task['conditions'][key]['score']:5.2f}"
            if task['conditions'][key]['score'] is not None else "  n/a"
            for key in sorted(task["conditions"]))
        n = task["healthy_successes"]
        print(f"{task['task']:30s} {task['healthy_score']:6.3f} {n:>4}  {scores}  "
              f"{task['task_score']:6.3f}")
        if 0 < n <= 3:
            thin.append((task["task"], n))
    if thin:
        print("\nthin denominators (every condition score of these tasks rests on "
              "n healthy successes):")
        for name, n in thin:
            print(f"  {name:30s} n={n}")

    score = record["mail_bench_score"]
    print(f"\nMAIL-Bench score      "
          f"{score:.4f}" if score is not None else
          "\nMAIL-Bench score      N/A (experiment, not an official submission)")

    if args.out:
        args.out.write_text(json.dumps(record, indent=2, sort_keys=True), encoding="utf-8")
        print(f"record={args.out}")


if __name__ == "__main__":
    main()
