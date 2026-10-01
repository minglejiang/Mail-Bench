#!/usr/bin/env python3
"""Is the frozen platform's rendering bit-deterministic on this host?

Same frozen scene, zero actions, two environments in one process, and the
frame hashes saved so a second process can be compared. For every frame that
differs, the number of differing pixels and the largest per-channel
difference are printed, so a renderer's rasterisation noise can be told from
a real divergence.

On the hardware the reference results were produced on (recorded in each
run's runtime fingerprint) physics and two of three cameras are identical
across processes; the wrist camera differs by rasterisation noise in a handful
of pixels. The report's pre-fault divergence rate carries this as provenance.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scene-bank", type=Path, required=True)
    parser.add_argument("--namespace", default="official")
    parser.add_argument("--registry", type=Path, default=ROOT / "configs" / "dataset_registry.yaml")
    parser.add_argument("--task", default="CloseBlenderLid")
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--save", type=Path, help="write this process's frame hashes here")
    parser.add_argument("--compare", type=Path, help="hashes saved by another process")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sys.path.insert(0, str(ROOT / "src"))
    import numpy as np
    from mail_bench.manifest import PROTOCOL_VERSION
    from mail_bench.platforms.robocasa import RoboCasaEnvironmentAdapter
    from mail_bench.registry import load_and_validate_dataset
    from mail_bench.scenes import SceneBank
    from mail_bench.seeds import episode_seed
    from mail_bench.suite import horizons

    auth = load_and_validate_dataset(args.registry, "robocasa", stage="stage_1")
    bank = SceneBank(args.scene_bank, namespace=args.namespace)
    seed = episode_seed("robocasa", args.task, args.episode, PROTOCOL_VERSION)

    def sha(a):
        return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()[:16]

    def rollout():
        env = RoboCasaEnvironmentAdapter.from_frozen_scene(
            bank=bank, task=args.task, episode_index=args.episode,
            dataset_revision=auth.revision or "", split="target", seed=seed,
            official_horizon=horizons()[args.task])
        obs = env.reset(seed)
        frames, hashes = [], []
        for _ in range(args.steps):
            frames.append({c: np.array(f, copy=True) for c, f in obs.cameras.items()})
            hashes.append({"step": obs.step, "state": sha(obs.robot_state),
                           **{c: sha(f) for c, f in obs.cameras.items()}})
            obs = env.step(np.zeros(12, np.float32)).observation
        env.close()
        return frames, hashes

    started = time.time()
    frames_a, hashes_a = rollout()
    frames_b, hashes_b = rollout()
    print(f"in-process: two environments, {args.steps} steps, {time.time() - started:.0f}s")
    for key in (k for k in hashes_a[0] if k != "step"):
        differing = [h["step"] for h, g in zip(hashes_a, hashes_b) if h[key] != g[key]]
        print(f"  {key}: {'identical' if not differing else f'differs at {len(differing)}/{args.steps} steps'}")
    for step, (fa, fb) in enumerate(zip(frames_a, frames_b)):
        for camera in fa:
            d = np.abs(fa[camera].astype(np.int16) - fb[camera].astype(np.int16))
            pixels = int((d.max(axis=-1) > 0).sum())
            if pixels:
                print(f"    step {step:3d} {camera}: {pixels} of {d.shape[0] * d.shape[1]} pixels "
                      f"differ, max |d| = {int(d.max())}")
    if args.save:
        args.save.write_text(json.dumps(hashes_a))
    if args.compare:
        other = json.loads(args.compare.read_text())
        print("cross-process vs saved:")
        for key in (k for k in hashes_a[0] if k != "step"):
            differing = [h["step"] for h, g in zip(hashes_a, other) if h[key] != g[key]]
            print(f"  {key}: {'identical' if not differing else f'differs at {len(differing)}/{args.steps} steps'}")


if __name__ == "__main__":
    main()
