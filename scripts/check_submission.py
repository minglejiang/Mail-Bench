#!/usr/bin/env python3
"""Ask a policy server, in a minute, what an official run would refuse.

Every condition MAIL-Bench places on a submission is checked rather than
declared, and each of those checks sits behind a run that takes hours: the
declaration is validated at the first connection, the action chunk at the first
step, the pinned weights only when ``--official`` is given, and the seeded
policy stream only much later, when a fault arm is compared against the healthy
rollout it is paired with.  A server with a mistake in any of them can burn a
whole cohort before anything says so.

This script asks the same questions the kernel asks, in the same order, against
a synthetic observation, and prints which of them would refuse the run.  It
proves nothing about how well a policy works and awards nothing: a submission
that passes here has a well-formed server, not a good score.

The action chunk is read through the kernel's own adapter rather than a copy of
it, so a chunk of the wrong shape, of a non-numeric dtype, or carrying a NaN is
refused here in the same words it would be refused mid-cohort -- and a NaN is
worth catching here, because it otherwise passes every shape check and reaches
the simulator hours later.

Cross-process determinism is the one condition that cannot be checked from a
single server: healthy and fault phases are served by separately started
processes, so the same seed must give the same actions in a *fresh* process.
Start a second instance of the same server on another port and pass
``--second-port`` to check it; without that the condition is reported as
unchecked rather than passed.

    python scripts/check_submission.py --policy-port 9000
    python scripts/check_submission.py --policy-port 9000 --second-port 9001 \
        --model-revision <revision> --inventory-sha256 <digest>
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
PLATFORM = "robocasa365"
PASS, FAIL, WARN, UNCHECKED = "pass", "fail", "warn", "unchecked"
MARK = {PASS: "  ok  ", FAIL: " FAIL ", WARN: " warn ", UNCHECKED: "  --  "}


@dataclass
class Check:
    name: str
    status: str
    detail: str

    def line(self) -> str:
        return f"[{MARK[self.status]}] {self.name}: {self.detail}"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--policy-host", default="127.0.0.1")
    parser.add_argument("--policy-port", type=int, required=True)
    parser.add_argument("--second-port", type=int,
                        help="a second instance of the SAME server, started as its own "
                             "process, for the cross-process determinism check")
    parser.add_argument("--policy-timeout", type=float, default=600.0)
    parser.add_argument("--profiles", type=Path,
                        default=ROOT / "configs" / "platform_profiles.yaml")
    parser.add_argument("--model-revision", help="what an --official run would pin")
    parser.add_argument("--inventory-sha256", help="what an --official run would pin")
    parser.add_argument("--checkpoint-sha256", help="what an --official run would pin")
    parser.add_argument("--seed", type=int, default=12345,
                        help="the policy seed used for the repeatability checks")
    parser.add_argument("--image-size", type=int, default=256,
                        help="side of the synthetic frames sent to the server")
    parser.add_argument("--json", type=Path, help="write the report here as well")
    return parser.parse_args(argv)


def platform_cameras(profiles_path: Path) -> tuple[str, ...]:
    from mail_bench.registry import load_yaml

    profiles = load_yaml(profiles_path)
    platforms = profiles.get("platforms", profiles)
    return tuple(platforms[PLATFORM]["grid_cameras"])


def synthetic_observation(cameras, *, step: int, size: int, absent=(), seed: int = 0):
    """One canonical observation, shaped exactly as the kernel produces them.

    An absent camera is ``None`` beside ``availability=false``; the kernel never
    substitutes anything, so a server that cannot cope with the ``None`` will
    fail here rather than at the first fault cell.
    """
    import numpy as np

    from mail_bench.interfaces import CanonicalObservation
    from mail_bench.platforms.robocasa import ROBOCASA_STATE_DIM

    rng = np.random.default_rng(seed)
    frames, availability = {}, {}
    for camera in cameras:
        missing = camera in absent
        frames[camera] = None if missing else rng.integers(
            0, 256, size=(size, size, 3), dtype=np.uint8)
        availability[camera] = not missing
    return CanonicalObservation(
        step=step,
        cameras=frames,
        availability=availability,
        robot_state=np.zeros(ROBOCASA_STATE_DIM, dtype=np.float32),
        language="close the blender lid",
        new_frame={camera: availability[camera] for camera in cameras},
    )


def chunk_of(output) -> Any:
    import numpy as np

    return np.asarray([np.asarray(row, dtype=np.float64) for row in output.actions])


def connected_adapter(args, cameras, port: int):
    from mail_bench.policy_api import GenericSocketPolicyAdapter

    adapter = GenericSocketPolicyAdapter(
        host=args.policy_host, port=port, timeout_seconds=args.policy_timeout,
        environment_cameras=cameras,
        expected_identity={
            "model_revision": args.model_revision,
            "inventory_sha256": args.inventory_sha256,
            "checkpoint_sha256": args.checkpoint_sha256,
        },
    )
    identity = adapter.connect()
    return adapter, identity


def act_once(adapter, cameras, *, seed: int, size: int, absent=(), step: int = 0):
    from mail_bench.interfaces import EpisodeContext

    adapter.announce_episode(EpisodeContext(
        task="CloseBlenderLid", episode_index=0, official_horizon=500,
        instruction="close the blender lid"))
    adapter.reset(seed)
    return chunk_of(adapter.act(synthetic_observation(
        cameras, step=step, size=size, absent=absent)))


def run_checks(args) -> list[Check]:
    import numpy as np

    from mail_bench.policy_api import ContractError, cameras_outside_declared_scope
    from mail_bench.semantic_states import MISSING_STATES, state_cameras, validate_platform

    checks: list[Check] = []
    cameras = platform_cameras(args.profiles)
    validate_platform(PLATFORM, cameras)

    # 1. The connection and the declaration: what the cohort resolves before a
    #    single scene is restored.
    try:
        adapter, identity = connected_adapter(args, cameras, args.policy_port)
    except ContractError as error:
        return [Check("declaration", FAIL, str(error))]
    except Exception as error:  # noqa: BLE001 - the reason is what the user needs
        return [Check("connect", FAIL, f"{type(error).__name__}: {error}")]

    declaration = adapter.declaration
    checks.append(Check("connect", PASS,
                        f"{args.policy_host}:{args.policy_port} answered identity"))
    checks.append(Check("declaration", PASS,
                        f"model_id={declaration.model_id}, {declaration.weight_digest_field}"
                        f"={declaration.checkpoint_sha256[:12]}, chunk="
                        f"{declaration.predicted_action_chunk}, horizon="
                        f"{declaration.native_execution_horizon}, cameras="
                        f"{list(declaration.policy_visible_cameras)}"))

    try:
        health = adapter.require_ready()
        checks.append(Check("health", PASS, f"ready ({health.get('status', 'ready')})"))
    except Exception as error:  # noqa: BLE001
        checks.append(Check("health", FAIL, str(error)))

    # 2. What --official pins. The runner requires the flags; the adapter then
    #    holds the server to them, so a mismatch here is a refusal there.
    pinned = {key: value for key, value in {
        "model_revision": args.model_revision,
        "inventory_sha256": args.inventory_sha256,
        "checkpoint_sha256": args.checkpoint_sha256}.items() if value}
    if not pinned:
        checks.append(Check("official pinning", UNCHECKED,
                            "--official requires --model-revision and one weight digest; "
                            "pass them here to check the server agrees"))
    else:
        checks.append(Check("official pinning", PASS,
                            f"the server's identity matches {sorted(pinned)}"))

    from mail_bench.platforms.robocasa import ROBOCASA_ACTION_DIM

    if int(declaration.action_dimension) == ROBOCASA_ACTION_DIM:
        checks.append(Check("action dimension", PASS,
                            f"declares {declaration.action_dimension}, which is what this "
                            "platform executes"))
    else:
        checks.append(Check("action dimension", FAIL,
                            f"declares action_dimension={declaration.action_dimension}; this "
                            f"platform executes {ROBOCASA_ACTION_DIM}-dimensional actions and "
                            "the cohort refuses the mismatch before the first scene"))

    blind = cameras_outside_declared_scope(declaration, cameras)
    if blind:
        checks.append(Check("camera scope", WARN,
                            f"declares it never reads {list(blind)}; those conditions are "
                            "still ranked, and the declaration is published beside the score"))

    # 3. The action chunk, checked by the same code that checks it mid-cohort.
    try:
        first = act_once(adapter, cameras, seed=args.seed, size=args.image_size)
        checks.append(Check("action chunk", PASS,
                            f"{first.shape[0]}x{first.shape[1]}, finite, as declared"))
    except Exception as error:  # noqa: BLE001
        checks.append(Check("action chunk", FAIL, f"{type(error).__name__}: {error}"))
        return checks

    # 4. One seed, one policy stream. The healthy rollout and every fault arm of
    #    a scene are separate episodes; if the same seed does not reproduce the
    #    same actions, the arms cannot be compared before the onset at all.
    again = act_once(adapter, cameras, seed=args.seed, size=args.image_size)
    if np.array_equal(first, again):
        checks.append(Check("seed repeatability", PASS,
                            "the same seed reproduced the chunk bit for bit"))
    else:
        worst = float(np.max(np.abs(first - again)))
        checks.append(Check("seed repeatability", FAIL,
                            f"the same seed gave a different chunk (max |difference| "
                            f"{worst:.3e}); seed every source of randomness in the "
                            "server, the model's own sampler included"))

    # Repeatability alone is not evidence that the seed arrives: a server that
    # ignores it entirely passes that check perfectly.  The two together are the
    # evidence.  A server that seeds `random` and NumPy but not the model's own
    # sampler (a JAX key, a torch generator) passes repeatability and still
    # drops the seed, so an identical chunk under a different seed is flagged
    # unless the server declares itself deterministic.
    other = act_once(adapter, cameras, seed=args.seed + 1, size=args.image_size)
    if not np.array_equal(first, other):
        checks.append(Check("seed reaches the policy", PASS,
                            "a different seed changed the chunk"))
    elif declaration.policy_determinism == "deterministic":
        checks.append(Check("seed reaches the policy", PASS,
                            "identical under a different seed, and declared deterministic"))
    else:
        checks.append(Check("seed reaches the policy", WARN,
                            "a different seed gave an identical chunk. Either the seed is "
                            "not reaching the model's own sampler -- seeding random and "
                            "NumPy is not enough, a JAX or torch generator inside the model "
                            "has to be re-keyed too -- or the policy is deterministic, in "
                            "which case declare policy_determinism so a reader knows. "
                            "Nothing refuses this run; if the seed is being dropped, the "
                            "fault arms will not share the healthy rollout's policy stream "
                            "and the pairing the score rests on is void"))

    # 5. Every ranked missing state, not one camera at a time.  The conditions
    #    the score is made of remove a ROLE: agentview_missing takes both
    #    third-person cameras together, and all_vision_missing takes every
    #    frame.  A server that survives losing one camera can still fail when
    #    all three arrive as None, and that is the cheapest real crash to find.
    for state in MISSING_STATES:
        absent = state_cameras(PLATFORM, state)
        try:
            missing_chunk = act_once(adapter, cameras, seed=args.seed,
                                     size=args.image_size, absent=absent)
            identical = np.array_equal(first, missing_chunk)
            if identical and declaration.availability_consumed_by_policy:
                checks.append(Check(state, WARN,
                                    "answered, but the chunk is identical to the healthy "
                                    "one although the server declares it consumes "
                                    "availability"))
            else:
                checks.append(Check(state, PASS,
                                    f"answered a valid chunk without {list(absent)}"))
        except Exception as error:  # noqa: BLE001
            checks.append(Check(state, FAIL,
                                f"cannot serve {state} ({list(absent)} arrive as None): "
                                f"{type(error).__name__}: {error}"))

    # 6. A stateful server is told when the kernel drops its queued actions.
    if declaration.stateful_policy:
        for name, call in (("invalidate", adapter.invalidate_action_chunk),
                           ("visual_memory_reset", adapter.reset_visual_memory)):
            try:
                call("preflight", 0)
                checks.append(Check(name, PASS, "answered"))
            except Exception as error:  # noqa: BLE001
                checks.append(Check(name, FAIL,
                                    f"a server declaring stateful_policy receives {name} "
                                    f"and must answer it: {type(error).__name__}: {error}"))
    else:
        checks.append(Check("stateful hooks", UNCHECKED,
                            "stateless server; invalidate and visual_memory_reset are "
                            "never sent"))

    # 7. Cross-process determinism, the condition a single process cannot show.
    if args.second_port is None:
        checks.append(Check("cross-process determinism", UNCHECKED,
                            "NOT checked, and a server that is deterministic inside one "
                            "process can still fail it: GEMM autotuning picks kernels per "
                            "process (set XLA_FLAGS=--xla_gpu_autotune_level=0 "
                            "--xla_gpu_deterministic_ops=true before importing jax, or the "
                            "torch equivalents). Start the same server again on another "
                            "port and pass --second-port to check it here"))
    else:
        try:
            second, _ = connected_adapter(args, cameras, args.second_port)
            fresh = act_once(second, cameras, seed=args.seed, size=args.image_size)
            second.close()
            if np.array_equal(first, fresh):
                checks.append(Check("cross-process determinism", PASS,
                                    "a second process reproduced the chunk bit for bit"))
            else:
                worst = float(np.max(np.abs(first - fresh)))
                checks.append(Check("cross-process determinism", FAIL,
                                    f"a second process gave a different chunk (max "
                                    f"|difference| {worst:.3e}); healthy and fault phases "
                                    "are served by separately started processes"))
        except Exception as error:  # noqa: BLE001
            checks.append(Check("cross-process determinism", FAIL,
                                f"{type(error).__name__}: {error}"))

    try:
        adapter.close()
        checks.append(Check("disconnect", PASS, "closed cleanly"))
    except Exception as error:  # noqa: BLE001
        checks.append(Check("disconnect", WARN, f"{type(error).__name__}: {error}"))
    return checks


def main(argv=None) -> int:
    args = parse_args(argv)
    sys.path.insert(0, str(ROOT / "src"))
    checks = run_checks(args)

    for check in checks:
        print(check.line())
    failures = [check for check in checks if check.status == FAIL]
    warnings = [check for check in checks if check.status == WARN]
    unchecked = [check for check in checks if check.status == UNCHECKED]
    print()
    if failures:
        print(f"REFUSED: {len(failures)} condition(s) would stop an official run:")
        for check in failures:
            print(f"  - {check.name}")
    else:
        print("Nothing checked here would stop an official run.")
    if warnings:
        # Not refusals. Each one is a way a run can finish and produce a number
        # that does not mean what the reader will take it to mean.
        print("Accepted, but read these before spending the compute:")
        for check in warnings:
            print(f"  - {check.name}")
    if unchecked:
        print("Not checked: " + ", ".join(check.name for check in unchecked))
    if not failures:
        print("This says the server is well formed. It says nothing about the score.")

    if args.json:
        args.json.write_text(json.dumps(
            {"checks": [vars(check) for check in checks],
             "would_be_refused": bool(failures)}, indent=2, sort_keys=True) + "\n",
            encoding="utf-8")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
