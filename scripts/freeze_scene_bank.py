#!/usr/bin/env python3
"""Check a finished scene bank and freeze it under one identity.

After this runs, a single hash stands for MAIL-Bench's canonical starting
scenes. A run can then state which bank it evaluated on in one field, and any
later divergence -- a scene regenerated, an episode quietly added, a fixture
edited -- shows up as a different hash rather than as an unexplained change in
somebody's score.

The integrity check comes first and gates the freeze. A bank missing a scene has
900 in its name and 899 on disk, and nothing downstream would notice: the
healthy phase would simply produce one fewer cell and every denominator would
shrink by one.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mail_bench.manifest import PROTOCOL_VERSION, semantic_sha256      # noqa: E402
from mail_bench.scenes import (                                        # noqa: E402
    SCENE_BANK_ID,
    SCENE_SCHEMA_VERSION,
    SceneBank,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bank-root", type=Path, required=True)
    parser.add_argument("--namespace", default="official")
    parser.add_argument("--platform", default="robocasa365")
    parser.add_argument("--expected", type=int, default=900)
    parser.add_argument("--expected-tasks", type=int, default=18)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    bank = SceneBank(args.bank_root, namespace=args.namespace)
    root = Path(args.bank_root) / args.namespace / args.platform
    if not root.is_dir():
        raise SystemExit(f"no bank at {root}")
    from mail_bench.suite import scenes_per_task, task_ids

    tasks = sorted(p.name for p in root.iterdir() if p.is_dir())
    # Eighteen directories is not the eighteen tasks, and nine hundred scenes
    # is not fifty per task: the bank is held to the suite by name and count.
    not_the_suite = sorted(set(tasks) ^ set(task_ids()))
    short = sorted(task for task in task_ids()
                   if len(bank.episodes(args.platform, task)) != scenes_per_task())

    episodes: list[tuple[str, int]] = []
    identities: dict[tuple[str, int], str] = {}
    bank_ids, schemas, problems = Counter(), Counter(), []
    for task in tasks:
        for index in sorted(bank.episodes(args.platform, task)):
            try:
                episode, _, _, _ = bank.read(args.platform, task, index)
            except Exception as exc:                        # noqa: BLE001
                problems.append(f"{task}#{index}: {type(exc).__name__}: {exc}"[:160])
                continue
            episodes.append((task, index))
            identities[(task, index)] = episode.identity
            bank_ids[episode.scene_bank_id] += 1
            schemas[episode.schema_version] += 1

    unique = len(set(episodes))
    duplicate = len(episodes) - unique
    missing = max(args.expected - unique, 0)

    print(f"tasks {len(tasks)} (expected {args.expected_tasks})")
    print(f"  expected   {args.expected}")
    print(f"  unique     {unique}")
    print(f"  missing    {missing}")
    print(f"  duplicate  {duplicate}")
    print(f"  scene_bank_id  {dict(bank_ids)}")
    print(f"  schema_version {dict(schemas)}")
    if problems:
        print(f"  unreadable {len(problems)}")
        for line in problems[:5]:
            print(f"    {line}")

    if not_the_suite:
        print(f"  tasks not the frozen suite: {not_the_suite[:6]}")
    if short:
        print(f"  tasks without {scenes_per_task()} scenes: {short[:6]}")
    ok = (
        unique == args.expected
        and not not_the_suite
        and not short
        and not problems
        and len(tasks) == args.expected_tasks
        and set(bank_ids) == {SCENE_BANK_ID}
        and set(schemas) == {SCENE_SCHEMA_VERSION}
    )
    # The bank's identity is the ordered list of its scenes' identities, so it
    # changes if any scene changes, is added or is removed.
    root_hash = semantic_sha256({
        "scene_bank_id": SCENE_BANK_ID,
        "schema_version": SCENE_SCHEMA_VERSION,
        "namespace": args.namespace,
        "platform": args.platform,
        "scenes": [[task, index, identities[(task, index)]]
                   for task, index in sorted(identities)],
    })
    record = {
        "scene_bank_id": SCENE_BANK_ID,
        "schema_version": SCENE_SCHEMA_VERSION,
        "namespace": args.namespace,
        "platform": args.platform,
        "frozen_under_protocol": PROTOCOL_VERSION,
        "tasks": len(tasks),
        "expected": args.expected,
        "unique": unique,
        "missing": missing,
        "duplicate": duplicate,
        "unreadable": problems,
        "root_manifest_hash": root_hash,
        "frozen": ok,
        "scene_identities": {f"{task}#{index:04d}": identities[(task, index)]
                             for task, index in sorted(identities)},
    }
    args.out.write_text(json.dumps(record, indent=2, sort_keys=True), encoding="utf-8")
    print(f"\nroot_manifest_hash {root_hash}")
    print(f"record={args.out}")
    if not ok:
        print("\nBANK NOT FROZEN: the integrity check did not pass")
        raise SystemExit(1)
    print("\nSCENE_BANK_FROZEN")


if __name__ == "__main__":
    main()
