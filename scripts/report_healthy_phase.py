#!/usr/bin/env python3
"""Completeness and clean capability of a finished healthy phase.

Four integrity numbers first, because everything below only means something if
the cohort is exactly what it claims to be: every official scene present, once.
A missing scene silently shrinks the denominator of H; a duplicated one silently
weights a scene twice. Neither is visible in the score itself.

Then the clean capability: H over every official scene, the same per task, and
the distribution of completion steps over the episodes that succeeded. No scene
and no task is filtered out -- a task the policy cannot do is a result about the
policy, and dropping it would make H a different quantity.

The cells are read with the aggregator's own reader rather than a second parser,
so this cannot disagree with the tables about what a cell says.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from statistics import median
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mail_bench.aggregate import load_cells                      # noqa: E402


def quantile(values, q):
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round(q * (len(ordered) - 1)))))
    return ordered[index]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--expected", type=int,
                        help="scenes expected; defaults to the frozen suite's 18 x 50")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    from mail_bench.aggregate import MeasurementKey, astuple_key
    from mail_bench.suite import scenes_per_task, task_ids

    if args.expected is None:
        args.expected = len(task_ids()) * scenes_per_task()

    rows, failures = load_cells(args.run_root)
    healthy = [row for row in rows if row.is_healthy]
    fault = [row for row in rows if not row.is_healthy]

    seen = Counter(row.unit for row in healthy)
    duplicates = sorted(unit for unit, n in seen.items() if n > 1)
    unique = len(seen)
    # The suite's shape, not a count: eighteen named tasks with fifty scenes
    # each, from one policy configuration. Nine hundred scenes of the wrong
    # tasks, or of two checkpoints, would pass a count.
    tasks_seen = Counter(unit[0] for unit in seen)
    wrong_tasks = sorted(set(tasks_seen) ^ set(task_ids()))
    short_tasks = sorted(task for task in task_ids()
                         if tasks_seen.get(task, 0) != scenes_per_task())
    measurements = {astuple_key(MeasurementKey.of(row.result)) for row in healthy}

    print(f"read {len(rows)} cells: {len(healthy)} healthy, {len(fault)} fault")
    if failures:
        print(f"  audit failures: {len(failures)}")
        for failure in failures[:5]:
            print(f"    {failure.kind}: {failure.detail}")

    print("\nintegrity")
    print(f"  N_expected   {args.expected}")
    print(f"  N_unique     {unique}")
    print(f"  N_missing    {max(args.expected - unique, 0)}")
    print(f"  N_duplicate  {len(duplicates)}")
    if duplicates:
        print(f"    e.g. {duplicates[:5]}")
    if wrong_tasks:
        print(f"  tasks not the frozen suite: {wrong_tasks[:6]}")
    if short_tasks:
        print(f"  tasks without {scenes_per_task()} scenes: {short_tasks[:6]}")
    if len(measurements) != 1:
        print(f"  measurement identities: {len(measurements)} (one policy configuration expected)")
    complete = (unique == args.expected and not duplicates and not failures
                and not wrong_tasks and not short_tasks and len(measurements) == 1)

    # One canonical rollout per scene: a unit is solved when its healthy cell is.
    solved, steps, per_task = {}, [], {}
    for row in healthy:
        result = row.result
        success = bool(result.get("success_or_valid"))
        solved[row.unit] = solved.get(row.unit, False) or success
        entry = per_task.setdefault(row.task, {"scenes": 0, "solved": 0, "steps": []})
        entry["scenes"] += 1
        if success:
            entry["solved"] += 1
            step = int(result["environment_step_count"])
            entry["steps"].append(step)
            steps.append(step)

    successes = sum(1 for ok in solved.values() if ok)
    H = successes / args.expected if args.expected else None
    print(f"\nclean success  H = {successes}/{args.expected} = "
          f"{H:.4f}" if H is not None else "\nH undefined")

    print("\nper task")
    for task in sorted(per_task):
        e = per_task[task]
        rate = e["solved"] / e["scenes"] if e["scenes"] else 0.0
        med = f"{median(e['steps']):.0f}" if e["steps"] else "-"
        print(f"  {task:30s} {e['solved']:>3}/{e['scenes']:<3} = {rate:5.2f}   "
              f"median T {med:>5}")

    print("\ncompletion steps over successful episodes")
    if steps:
        print(f"  n {len(steps)}  min {min(steps)}  p25 {quantile(steps,0.25)}  "
              f"median {median(steps):.0f}  p75 {quantile(steps,0.75)}  max {max(steps)}")
    else:
        print("  none")

    payload = {
        "run_root": str(args.run_root),
        "integrity": {"n_expected": args.expected, "n_unique": unique,
                      "n_missing": max(args.expected - unique, 0),
                      "n_duplicate": len(duplicates), "duplicates": duplicates,
                      "audit_failures": [f.kind for f in failures],
                      "complete": complete},
        "clean_success": H,
        "healthy_successes": successes,
        "per_task": {t: {"scenes": e["scenes"], "solved": e["solved"],
                         "median_completion_step": (median(e["steps"]) if e["steps"] else None)}
                     for t, e in sorted(per_task.items())},
        "completion_steps": {"n": len(steps), "min": min(steps) if steps else None,
                             "p25": quantile(steps, 0.25),
                             "median": median(steps) if steps else None,
                             "p75": quantile(steps, 0.75),
                             "max": max(steps) if steps else None},
    }
    if args.out:
        args.out.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        print(f"\nreport={args.out}")
    if not complete:
        print("\nINCOMPLETE: do not derive an onset manifest from this run")
        raise SystemExit(1)
    print("\nHEALTHY_PHASE_COMPLETE")


if __name__ == "__main__":
    main()
