#!/usr/bin/env python3
"""The main results table's rows, built from score records rather than typed.

Every number in the reference results table is a projection of one run's score
record, so this reads the records and prints the LaTeX rows. Typing them by
hand is how a table comes to disagree with the run it claims to summarise.

The benchmark-level figure for a condition is the equal-weight mean over the
eighteen tasks of that condition's ``ranked_value`` -- its ``M_c,t``, with a
task that has no healthy success contributing zero, exactly as the task score
uses it. With that definition the row decomposes the score:

    S_MAIL = (mean_t H_t + sum_c mean_t M_c,t) / 10

which this checks and refuses to print if it does not hold.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mail_bench.scoring import MISSING_CONDITIONS, RANKING_COMPONENTS   # noqa: E402


def row_values(record: dict) -> dict:
    """Benchmark-level H, the nine conditions and the score, from one record."""
    tasks = record["tasks"]
    if not tasks:
        raise SystemExit("the record has no task scores")
    healthy = sum(task["healthy_score"] for task in tasks) / len(tasks)
    conditions = {}
    for state, onset in MISSING_CONDITIONS:
        key = f"{state}@{onset:.2f}"
        values = []
        for task in tasks:
            if key not in task["conditions"]:
                raise SystemExit(f"task {task['task']} has no condition {key}")
            score = task["conditions"][key]["score"]
            # None means the task has no healthy success, which the task score
            # counts as zero; the table must decompose the same way.
            values.append(0.0 if score is None else float(score))
        conditions[key] = sum(values) / len(values)
    decomposed = (healthy + sum(conditions.values())) / RANKING_COMPONENTS
    score = record.get("score")
    if score is not None and abs(decomposed - float(score)) > 1e-9:
        raise SystemExit(
            f"the row does not decompose the score: {decomposed!r} vs {score!r}. "
            "The table would say something the record does not."
        )
    return {"healthy": healthy, "conditions": conditions, "score": score,
            "official": bool(record.get("official")), "tasks": len(tasks)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--record", action="append", required=True, metavar="LABEL=PATH",
                        help="a policy's LaTeX row label and its score record; repeatable")
    parser.add_argument("--reported", action="append", default=[], metavar="LABEL=VALUE",
                        help="RoboCasa365's own reported Atomic-Seen number for that policy, "
                             "printed as the sanity reference column")
    args = parser.parse_args()

    reported = dict(pair.split("=", 1) for pair in args.reported)
    for entry in args.record:
        label, path = entry.split("=", 1)
        values = row_values(json.loads(Path(path).read_text(encoding="utf-8")))
        cells = [reported.get(label, "--"), f"{values['healthy']:.3f}"]
        cells += [f"{values['conditions'][f'{s}@{o:.2f}']:.3f}" for s, o in MISSING_CONDITIONS]
        cells.append(f"{values['score']:.3f}" if values["score"] is not None else "--")
        print(f"{label} & " + " & ".join(cells) + r" \\")
        if not values["official"]:
            print(f"% {label}: NOT an official record ({values['tasks']} tasks); "
                  "the row may not carry the benchmark's name", file=sys.stderr)


if __name__ == "__main__":
    main()
