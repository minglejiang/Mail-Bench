#!/usr/bin/env python3
"""Derive one model's onset manifest from its own standard healthy cells.

MAIL-Bench publishes one of these per model rather than one shared reference
file per platform. A shared file would have to name some policy's trajectory as
the definition of "45% of the task" for every other policy, which turns the
sweep back into absolute-time perturbation for everyone but that one.

The input is the standard cells the driver wrote, not a separate record of the
same run: a manifest built from a second, parallel account of what happened
could disagree with the tables about which scenes were solved and how long they
took, and there would be no way to tell which was right. The cells are read with
the aggregator's own reader for the same reason.

A scene with a healthy success carries three onsets and is evaluable. A scene
without one carries none: a fraction names a task phase, and a phase exists only
relative to a trajectory that reached the end.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mail_bench.aggregate import load_cells                      # noqa: E402
from mail_bench.manifest import PROTOCOL_VERSION                 # noqa: E402
from mail_bench.onset import RANKING_ONSET_FRACTIONS, onset_manifest_row   # noqa: E402


def completion_step(result: dict) -> int:
    """T_healthy, with the runner's two counts of it checked against each other."""
    executed = int(result["action_execution_count"])
    stepped = int(result["environment_step_count"])
    if executed != stepped:
        raise SystemExit(
            f"cell {result.get('cell_id')} executed {executed} actions over {stepped} "
            "environment steps; T_healthy is undefined when the two differ"
        )
    return executed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-root", type=Path, required=True,
                        help="the healthy phase's run root, not its cells/ directory")
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--checkpoint-identity",
                        help="optional assertion; the published identity is "
                             "always the one the cells carry")
    parser.add_argument("--platform", default="robocasa365")
    parser.add_argument("--expected-scenes", type=int, default=None,
                        help="testing only; defaults to the frozen suite's task "
                             "count times its scenes per task")
    parser.add_argument("--replan", type=int, default=None,
                        help="actions executed per policy query; read from the run "
                             "certificate's execution profile, and refused if it disagrees")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    rows, failures = load_cells(args.run_root)
    if failures:
        raise SystemExit(f"{len(failures)} cells could not be read; fix the run first")
    healthy = [row for row in rows if row.is_healthy]
    if not healthy:
        raise SystemExit(f"no healthy cells under {args.run_root}")

    seen = Counter(row.unit for row in healthy)
    duplicated = sorted(unit for unit, n in seen.items() if n > 1)
    if duplicated:
        raise SystemExit(f"{len(duplicated)} scenes have more than one healthy cell "
                         f"(e.g. {duplicated[:3]}); the onsets would be ambiguous")
    from mail_bench.suite import scenes_per_task, task_ids

    expected_scenes = (args.expected_scenes if args.expected_scenes is not None
                       else len(task_ids()) * scenes_per_task())
    if len(seen) != expected_scenes:
        raise SystemExit(f"{len(seen)} scenes have a healthy cell, expected "
                         f"{expected_scenes}; an onset manifest may not be built "
                         "from an incomplete healthy phase")
    if args.expected_scenes is None:
        # The count alone cannot tell 900 scenes of the suite from 900 scenes of
        # the wrong tasks; the shape is the suite's or the manifest is not built.
        per_task = Counter(task for task, _ in seen)
        wrong = sorted(set(per_task) ^ set(task_ids()))
        short = sorted(t for t, n in per_task.items() if n != scenes_per_task())
        if wrong or short:
            raise SystemExit(f"the healthy phase is not the frozen suite: tasks not in "
                             f"the suite or missing {wrong[:5]}, tasks without "
                             f"{scenes_per_task()} scenes {short[:5]}")

    # The cells carry no protocol_version of their own -- it lives in the run
    # certificate -- so the certificate is read.
    certificates = sorted(Path(args.run_root).glob("certificate_*.json"))
    if not certificates:
        raise SystemExit(
            f"no run certificate under {args.run_root}; the protocol and the "
            "policy identity of a run are recorded there, and a manifest that "
            "cannot check them would be asserting them"
        )
    declared = set()
    namespaces = set()
    intervals = set()
    for path in certificates:
        payload = json.loads(path.read_text(encoding="utf-8"))
        version = (payload.get("cohort_specification", {}) or {}).get("protocol_version") \
            or payload.get("protocol_version")
        if version:
            declared.add(str(version))
        runtime = payload.get("runtime") or {}
        if runtime.get("scene_namespace"):
            namespaces.add(str(runtime["scene_namespace"]))
        profile = payload.get("execution_profile") or {}
        if profile.get("max_actions_per_query") is not None:
            intervals.add(int(profile["max_actions_per_query"]))
    if len(namespaces) != 1:
        raise SystemExit(f"the run's certificates declare scene namespaces "
                         f"{sorted(namespaces) or 'nothing'}; a manifest names one bank")
    namespace = namespaces.pop()
    # The query interval is what the run executed, not what the operator types:
    # the certificate records it, and a flag may only restate it.
    if len(intervals) > 1:
        raise SystemExit(f"the run's certificates declare several query intervals {sorted(intervals)}")
    if intervals and args.replan is not None and args.replan != next(iter(intervals)):
        raise SystemExit(f"--replan is {args.replan}, but the run executed {next(iter(intervals))} actions per query")
    if not intervals and args.replan is None:
        raise SystemExit("the run's certificates carry no execution profile; pass --replan")
    query_interval = next(iter(intervals)) if intervals else int(args.replan)
    if declared != {PROTOCOL_VERSION}:
        # Onsets derived under one protocol and applied under another would place
        # faults on a scene bank the run does not have.
        # An empty set means the certificates carry no protocol at all, which
        # is not the same as carrying the right one.
        raise SystemExit(f"the run declares protocol {sorted(declared) or 'nothing'}, "
                         f"this build is {PROTOCOL_VERSION}")

    # The checkpoint identity is derived, not accepted. A published manifest that
    # took the operator's word for which weights ran could name weights the cells
    # never saw.
    ran = {str(row.result.get("checkpoint_hash")) for row in healthy}
    if len(ran) != 1:
        raise SystemExit(f"the healthy cells span {len(ran)} checkpoints: {sorted(ran)}")
    observed = ran.pop()
    if args.checkpoint_identity and not observed.startswith(args.checkpoint_identity):
        raise SystemExit(
            f"--checkpoint-identity is {args.checkpoint_identity!r}, but the cells "
            f"were produced by {observed!r}"
        )
    # The manifest publishes the full identity the cells carry, never the prefix
    # somebody typed: an abbreviation on the command line is an assertion, not a
    # source of identity.
    checkpoint_identity = observed

    manifest_rows = []
    for row in sorted(healthy, key=lambda r: (r.task, r.episode_index)):
        result = row.result
        metrics = result.get("official_metrics", {})
        success = bool(result.get("success_or_valid"))
        entry = onset_manifest_row(
            model_id=args.model_id,
            checkpoint_identity=checkpoint_identity,
            scene_identity=str(metrics.get("scene_identity", "")),
            healthy_success=success,
            # The protocol defines T_healthy as the healthy completion step.
            # The runner counts it as executed actions and as environment
            # steps; the two are checked equal here rather than assumed, and
            # the executed-action count is the one the fault phase uses.
            healthy_completion_step=(completion_step(result) if success else None),
            official_horizon=int(metrics["official_horizon"]),
            query_interval=query_interval,
        )
        if not entry["scene_identity"]:
            # A row that cannot say which scene it belongs to is not provenance.
            raise SystemExit(
                f"{row.task}#{row.episode_index} has no scene identity; an onset "
                "manifest without one names no scene"
            )
        entry["task"] = row.task
        entry["episode_index"] = row.episode_index
        entry["state0_sha256"] = metrics.get("state0_sha256")
        manifest_rows.append(entry)

    evaluable = [r for r in manifest_rows if r["fault_evaluable"]]
    skipped = Counter(r["skip_reason"] for r in manifest_rows if not r["fault_evaluable"])
    per_task: dict[str, dict[str, int]] = {}
    for entry in manifest_rows:
        bucket = per_task.setdefault(entry["task"], {"scenes": 0, "healthy_success": 0})
        bucket["scenes"] += 1
        bucket["healthy_success"] += int(entry["healthy_success"])

    manifest = {
        "protocol_version": PROTOCOL_VERSION,
        "platform": args.platform,
        "namespace": namespace,
        "model_id": args.model_id,
        "checkpoint_identity": checkpoint_identity,
        "derived_from": str(args.run_root),
        "onset_fractions": list(RANKING_ONSET_FRACTIONS),
        "query_interval": query_interval,
        "scenes": len(manifest_rows),
        "fault_evaluable_scenes": len(evaluable),
        "skipped": dict(skipped),
        # Reported beside the onsets so the fault numbers are never read alone.
        "clean_success": len(evaluable) / len(manifest_rows),
        "per_task": per_task,
        "rows": manifest_rows,
    }
    args.out.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(f"scenes {len(manifest_rows)}  fault-evaluable {len(evaluable)}  "
          f"skipped {dict(skipped)}  H={manifest['clean_success']:.4f}")
    print(f"manifest={args.out}")
    print("ONSET_MANIFEST_DONE")


if __name__ == "__main__":
    main()
