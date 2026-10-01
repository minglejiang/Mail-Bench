#!/usr/bin/env python3
"""Freeze the RoboCasa official scene bank for one protocol: 18 tasks x 50.

Scene identity derives from the bank id, the namespace, the platform, the task
and the episode index; the protocol version is recorded but does not enter a seed. Each
scene is frozen and restored once in place; every twentieth is also restored
into a freshly constructed environment as a cross-environment replay check.

Fail-closed on any drift: a scene whose replayed state hash differs from the one
just written is recorded as a failure rather than kept, and the asset inventory
is digested before and after the whole run.
"""
import argparse, json, os, shutil, sys, time
from pathlib import Path

_ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
_ap.add_argument("--robocasa-root", type=Path, default=os.environ.get("ROBOCASA_ROOT"), help="RoboCasa365 checkout (its robocasa/models/assets is digested)")
_ap.add_argument("--out", type=Path, default=Path("scene_bank_build"), help="directory that receives robocasa_scene_bank/ and the report")
_args = _ap.parse_args()
if _args.robocasa_root is None:
    raise SystemExit("--robocasa-root (or ROBOCASA_ROOT) is required")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
ASSETS = Path(_args.robocasa_root) / "robocasa" / "models" / "assets"
REV = "a07e365c958c4216cd6bbd5f30b47f09a65c6f00"
CAMERAS = ["robot0_agentview_left", "robot0_agentview_right", "robot0_eye_in_hand"]
SPLIT, EPISODES, FRESH_EVERY = "target", 50, 20
#: The protocol this bank is frozen under. It is recorded but does not enter a
#: scene seed: which scenes exist is decided by the bank id, so a change to the
#: scoring rule or the fault grid does not invalidate the 900 scenes. The
#: assertion makes a bank say which protocol produced it.
EXPECTED_PROTOCOL = "mail_bench_perturbation_v1"
#: Stop with a clear reason before the disk fills.
MIN_FREE_GB = 5.0

from mail_bench.artifacts import inventory_digest
from mail_bench.runtime import collect_runtime_fingerprint
from mail_bench.manifest import PROTOCOL_VERSION
from mail_bench.scenes import (
    SCENE_BANK_ID,
    SceneBank,
    capture_episode,
    episode_seed,
    restore_episode,
)

import robocasa
from robocasa.utils.dataset_registry import TARGET_TASKS
from robocasa.utils.env_utils import create_env

assert PROTOCOL_VERSION == EXPECTED_PROTOCOL, (PROTOCOL_VERSION, EXPECTED_PROTOCOL)
from mail_bench.suite import task_ids, verify_against_platform
from robocasa.utils.dataset_registry_utils import get_task_horizon

# The scenes a bank holds are decided by the task list, so the bank must be
# built from the frozen one. Reading it from the installed RoboCasa would make
# the bank's identity depend on a package version.
verify_against_platform(TARGET_TASKS["atomic_seen"], get_task_horizon)
TASKS = list(task_ids())
TOTAL = len(TASKS) * EPISODES


Path(_args.out).mkdir(parents=True, exist_ok=True)


def free_gb() -> float:
    return shutil.disk_usage(Path(_args.out)).free / 2**30


assets = inventory_digest(ASSETS, dataset_id="robocasa", revision=REV)
print(f"[assets] {assets}", flush=True)
print(f"[disk] {free_gb():.1f} GB free, floor {MIN_FREE_GB} GB", flush=True)
print(f"[plan] {len(TASKS)} tasks x {EPISODES} episodes = {TOTAL} scenes "
      f"| bank {SCENE_BANK_ID} | protocol {PROTOCOL_VERSION} | namespace official "
      f"| CUDA={os.environ.get('CUDA_VISIBLE_DEVICES')} "
      f"EGL={os.environ.get('MUJOCO_EGL_DEVICE_ID')}", flush=True)

bank = SceneBank(Path(_args.out) / "robocasa_scene_bank", namespace="official")
started, made, failures, halted = time.time(), 0, [], None

for task in TASKS:
    if halted:
        break
    for index in range(EPISODES):
        if free_gb() < MIN_FREE_GB:
            halted = f"disk below {MIN_FREE_GB} GB at {task}#{index}"
            print(f"  HALT {halted}", flush=True)
            break
        seed = episode_seed("robocasa365", task, index, namespace="official")
        try:
            env = create_env(env_name=task, split=SPLIT, seed=seed,
                             camera_names=CAMERAS, camera_widths=256, camera_heights=256)
            env.reset()
            episode, ep_meta, xml, state0 = capture_episode(
                env, namespace="official", platform="robocasa365", task=task,
                episode_index=index, split=SPLIT, episode_seed=seed,
                environment_revision=REV, asset_inventory_sha256=assets)
            # Replay first, keep second: a fixture that does not restore to
            # its own state is recorded as a failure, never written.
            in_place = restore_episode(env, episode, ep_meta, xml, state0)
            env.close()
            fresh = None
            if made % FRESH_EVERY == 0:
                other = create_env(env_name=task, split=SPLIT, seed=seed,
                                   camera_names=CAMERAS, camera_widths=256,
                                   camera_heights=256)
                other.reset()
                fresh = restore_episode(other, episode, ep_meta, xml, state0)
                other.close()
            made += 1
            ok = in_place == episode.state0_sha256 and (
                fresh is None or fresh == episode.state0_sha256)
            if ok:
                bank.write(episode, ep_meta, xml, state0)
            else:
                failures.append({"task": task, "episode": index, "error": "replay mismatch",
                                 "in_place": in_place, "fresh": fresh,
                                 "expected": episode.state0_sha256})
            if made % 25 == 0 or not ok:
                rate = (time.time() - started) / made
                print(f"  [{made:>3}/{TOTAL}] {task}#{index} "
                      f"{'ok' if ok else 'MISMATCH'} {episode.state0_sha256[:12]} "
                      f"({rate:.1f}s/scene, eta {(TOTAL - made) * rate / 60:.0f}m, "
                      f"{free_gb():.1f} GB free)", flush=True)
        except Exception as exc:                            # noqa: BLE001
            failures.append({"task": task, "episode": index,
                             "error": f"{type(exc).__name__}: {exc}"[:240]})
            print(f"  FAIL {task}#{index}: {exc}", flush=True)

post = inventory_digest(ASSETS, dataset_id="robocasa", revision=REV)
complete = made == TOTAL and not failures and assets == post and halted is None
report = {"namespace": "official", "scene_bank_id": SCENE_BANK_ID,
          "protocol_version": PROTOCOL_VERSION,
          "platform": "robocasa365", "split": SPLIT, "tasks": len(TASKS),
          "episodes_per_task": EPISODES, "scenes_expected": TOTAL,
          "scenes_written": made, "failures": failures,
          "assets_pre": assets, "assets_post": post,
          "assets_drift_free": assets == post,
          "fresh_env_replay_every": FRESH_EVERY,
          "halted": halted, "complete": complete,
          "render_device": {"cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                            "mujoco_egl_device_id": os.environ.get("MUJOCO_EGL_DEVICE_ID")},
          "benchmark_commit": collect_runtime_fingerprint(Path(__file__).resolve().parents[1]).get("benchmark_git_commit"),
          "elapsed_seconds": round(time.time() - started, 1)}
out = Path(_args.out) / f"robocasa_official_{EXPECTED_PROTOCOL}_scene_bank_report.json"
out.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
print(f"\nscenes {made}/{TOTAL}  failures {len(failures)}  "
      f"assets_drift_free {assets == post}  halted {halted}  "
      f"{report['elapsed_seconds'] / 60:.0f} min")
print(f"report={out}")
# The marker is earned by the product, never by reaching the last line.
if complete:
    print("OFFICIAL_BANK_DONE")
else:
    print("OFFICIAL_BANK_INCOMPLETE")
    sys.exit(1)
