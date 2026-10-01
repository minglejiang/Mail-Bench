#!/usr/bin/env python3
"""Run a MAIL-Bench cohort on RoboCasa against a policy served over a socket.

No scientific rule is decided here.  Horizons come from RoboCasa's own dataset
registry, cameras from the verified platform profile, onsets and subsets from
the protocol, replicate counts and thresholds from the protocol code.  What this
script does is read the frozen configs, prove the preflights, resolve the policy
contract from the live server, build the RoboCasa factories and hand everything
to :class:`mail_bench.driver.CohortRunner`.

Scenes are never constructed here.  Every unit is read from the frozen official
scene bank and replayed, so the healthy rollout and each fault arm of a unit
begin from the same bytes; ``mail_bench.scenes.restore_episode`` refuses to
return if the simulator lands anywhere else.

Two usage modes, both first-class -- ``custom`` (default) runs whatever
experiment you want and records what differed from the official profile;
``--official`` evaluates against the frozen profile and may claim comparability
with the official ranking.
"""

from __future__ import annotations

import argparse
from collections import OrderedDict
import contextlib
import json
import os
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
PLATFORM = "robocasa365"
#: Reference servers keep their own ports; anything else speaks the public
#: protocol, whose default port is the one the connection contract documents.
DEFAULT_PORTS = {"pi05_robocasa": 7606, "groot_n1_5_robocasa": 7614}
PUBLIC_PROTOCOL_PORT = 9000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    experiment = parser.add_argument_group("experiment")
    experiment.add_argument("--tasks", nargs="+",
                            help="RoboCasa task names; defaults to Atomic-Seen")
    experiment.add_argument("--episodes", type=int, nargs="+",
                            help="episode indices; defaults to every scene in the bank")
    experiment.add_argument("--onsets", type=float, nargs="+",
                            help="defaults to the protocol's ranked onset fractions")
    experiment.add_argument("--execution-profile", default="native")
    experiment.add_argument("--official", action="store_true")
    mechanism = parser.add_argument_group("mechanism analyses (experiments, never --official)")
    mechanism.add_argument("--fault-mode", default="hard_missing",
                           choices=("hard_missing", "blackout", "freeze", "stale_k", "burst_dropout"),
                           help="Failure Form: how the lost role is delivered after the onset (the ranking uses hard_missing)")
    mechanism.add_argument("--duration-fraction", type=float, default=1.0,
                           help="Missing Duration: the loss ends at this fraction of the remaining episode (1.0 = to the end)")
    mechanism.add_argument("--recovery-fraction", type=float, default=None,
                           help="Visual Recovery: the role returns at this fraction of the healthy completion step")
    mechanism.add_argument("--stale-ms", type=float, default=500.0, help="stale_k delay in milliseconds (protocol: 500)")
    mechanism.add_argument("--burst-ms", type=float, default=500.0, help="burst_dropout block in milliseconds (protocol: 500)")
    mechanism.add_argument("--burst-available-rate", type=float, default=0.5, help="burst_dropout: probability a later block is delivered")
    experiment.add_argument("--enforce-admission", dest="enforce_admission",
                            action="store_true", default=None)
    experiment.add_argument("--no-enforce-admission", dest="enforce_admission",
                            action="store_false")
    experiment.add_argument("--stage", default="stage_1")
    experiment.add_argument("--split", default="target")
    experiment.add_argument("--healthy-replicates", type=int,
                            help="defaults to the protocol's official replicate count")

    execution = parser.add_argument_group("execution")
    execution.add_argument("--phase", choices=("healthy", "fault", "all"), default="all")
    execution.add_argument("--workers", type=int, default=1)
    execution.add_argument("--worker-index", type=int, default=0)
    execution.add_argument("--output-dir", type=Path, required=True)

    policy = parser.add_argument_group("policy connection (runtime only, never identity)")
    policy.add_argument("--policy", default="custom")
    policy.add_argument("--policy-host", default="127.0.0.1")
    policy.add_argument("--policy-port", type=int)
    policy.add_argument("--policy-timeout", type=float, default=600.0)
    policy.add_argument("--policy-config", type=Path)
    policy.add_argument("--model-id")
    policy.add_argument("--checkpoint-sha256")
    policy.add_argument("--model-revision")
    policy.add_argument("--inventory-sha256")

    platform = parser.add_argument_group("platform")
    platform.add_argument("--scene-bank", type=Path, required=True,
                          help="root of the frozen scene bank")
    platform.add_argument("--namespace", default="official")
    platform.add_argument("--scene-bank-identity", type=Path,
                          default=ROOT / "configs" / "scene_bank_identity.json",
                          help="the published identity an --official run's bank must match")
    platform.add_argument("--robocasa-root", type=Path)
    platform.add_argument("--registry", type=Path,
                          default=ROOT / "configs" / "dataset_registry.yaml")
    platform.add_argument("--roster", type=Path,
                          default=ROOT / "configs" / "mail_bench_roster.yaml")
    platform.add_argument("--protocol-config", type=Path,
                          default=ROOT / "configs" / "perturbation_protocol.yaml")
    platform.add_argument("--profiles", type=Path,
                          default=ROOT / "configs" / "platform_profiles.yaml")
    return parser.parse_args()


def policy_port(args) -> int:
    if args.policy_port:
        return int(args.policy_port)
    return DEFAULT_PORTS.get(args.policy, PUBLIC_PROTOCOL_PORT)


def main() -> None:
    args = parse_args()
    if args.policy_config:
        from mail_bench.policy_api.connection import PolicyConnection

        connection = PolicyConnection.load(args.policy_config)
        args.policy_host = connection.host
        args.policy_port = connection.port
        args.policy_timeout = connection.timeout_seconds
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    if args.robocasa_root:
        sys.path.insert(0, str(args.robocasa_root.resolve()))

    from mail_bench.cohort import (
        cohort_units,
        execution_profile_for,
        platform_preflight,
        reference_model_preflight,
        resolve_runtime_contract,
        shard_units,
    )
    from mail_bench.driver import CohortRunner, CohortSpecification, OfficialProfile
    from mail_bench.experiment import MethodArmExecution, PairedCellRunner
    from mail_bench.io import AtomicResultWriter
    from mail_bench.manifest import PROTOCOL_VERSION, FaultManifest, semantic_sha256
    from mail_bench.platforms.robocasa import ROBOCASA_ACTION_DIM, RoboCasaEnvironmentAdapter
    from mail_bench.policy_api import GenericSocketPolicyAdapter
    from mail_bench.registry import load_and_validate_dataset, load_yaml
    from mail_bench.runner import RunnerConfig
    from mail_bench.runtime import collect_runtime_fingerprint
    from mail_bench.scenes import SceneBank, SceneBankError
    from mail_bench.seeds import episode_seed, fault_seed, policy_seed

    protocol = load_yaml(args.protocol_config)
    profiles = load_yaml(args.profiles)
    platforms = profiles.get("platforms", profiles)
    # One authority for camera ids, verified against MuJoCo rather than restated.
    cameras = tuple(platforms[PLATFORM]["grid_cameras"])
    onsets = tuple(args.onsets or protocol["onset_definition"]["fractions"])
    healthy_arm = protocol["pairing"]["healthy_arm"]
    replicates = args.healthy_replicates or int(
        healthy_arm.get("official_replicates_per_seed") or healthy_arm["replicates_per_seed"]
    )

    authorization = load_and_validate_dataset(args.registry, "robocasa", stage=args.stage)
    revision = authorization.revision or ""

    from robocasa.utils.dataset_registry import TARGET_TASKS
    from robocasa.utils.dataset_registry_utils import get_task_horizon

    from mail_bench.suite import horizons as frozen_horizons
    from mail_bench.suite import task_ids, verify_against_platform

    bank = SceneBank(args.scene_bank, namespace=args.namespace)
    if args.official:
        # An official run is a claim about a named checkpoint on the whole
        # suite. Without a pinned weight identity the server's word would be
        # the only record of what ran; with a task subset the eighteen task
        # scores would not be this benchmark's.
        if not (args.inventory_sha256 or args.checkpoint_sha256) or not args.model_revision:
            raise SystemExit("--official requires --model-revision and --inventory-sha256 "
                             "(or --checkpoint-sha256): the weights an official run "
                             "evaluates are pinned, not reported by the server")
        if args.tasks or args.episodes:
            raise SystemExit("--official evaluates the frozen suite; drop --tasks/--episodes "
                             "or drop --official and run a declared experiment")
        if args.fault_mode != "hard_missing" or args.duration_fraction != 1.0 or args.recovery_fraction is not None:
            raise SystemExit("--official is the main ranking (hard_missing, to the end of the episode); "
                             "the mechanism analyses run as declared experiments without --official")
        # The frozen scenes are what make two runs the same experiment. The
        # bank ships with the repository; a local rebuild is not guaranteed to
        # reproduce the reference bytes, so an official run is checked scene by
        # scene against the published identity before a single rollout starts.
        identity_path = args.scene_bank_identity
        if not identity_path.exists():
            raise SystemExit(f"--official needs the published bank identity at {identity_path}")
        try:
            bank.verify_against(json.loads(identity_path.read_text(encoding="utf-8")))
        except SceneBankError as exc:
            raise SystemExit(f"--official refused: {exc}")
    if args.recovery_fraction is not None and args.duration_fraction != 1.0:
        raise SystemExit("a cell has one end: give --recovery-fraction or --duration-fraction, not both")
    from mail_bench.operators import control_steps_for
    control_hz = float(platforms[PLATFORM]["control_hz"])
    stale_k = control_steps_for(args.stale_ms, control_hz) if args.fault_mode == "stale_k" else None
    burst_frames = control_steps_for(args.burst_ms, control_hz) if args.fault_mode == "burst_dropout" else None
    if args.official:
        # An official run is the frozen suite, taken from this repository rather
        # than from whichever RoboCasa is installed, and checked against the
        # installed one before anything executes: an upstream rename, a dropped
        # task or a changed horizon would otherwise be evaluated silently as if
        # it were the suite, and the run would produce eighteen task scores that
        # are not this benchmark's.
        #
        # A custom experiment is not making that claim, so it keeps the
        # platform's own task list and is not held to the frozen one.
        verify_against_platform(TARGET_TASKS["atomic_seen"], get_task_horizon)
        tasks = task_ids()
    else:
        tasks = tuple(args.tasks or TARGET_TASKS["atomic_seen"])
    # Horizons are per task here, not per suite. The kernel's cap has
    # to be one figure for the whole cohort -- it enters the cell identity, and a
    # per-task one would split the cohort into per-task measurements -- so the
    # specification carries the largest and no rollout is cut short by another
    # task's budget. Each adapter then terminates at its own task's official
    # horizon, which is where the per-task knowledge lives.
    # Pinned for the frozen tasks, the platform's own for anything else.
    pinned = frozen_horizons()
    horizons = {task: int(pinned.get(task) or get_task_horizon(task)) for task in tasks}
    # Over the whole frozen suite, not over the tasks selected: the figure
    # enters every cell's runner_config_hash, so a two-task experiment must
    # hash exactly as the full run does or its cells can never pair with it.
    horizon = max([*pinned.values(), *horizons.values()])

    all_units = []
    for task in tasks:
        available = bank.episodes(PLATFORM, task)
        chosen = [e for e in available if args.episodes is None or e in set(args.episodes)]
        if not chosen:
            raise SystemExit(f"the scene bank holds no episodes for {task!r}")
        all_units.extend(cohort_units([task], sorted(chosen)))
    all_units = tuple(all_units)
    my_units = shard_units(all_units, workers=args.workers, worker_index=args.worker_index)

    platform = platform_preflight(
        benchmark="robocasa",
        benchmark_root=ROOT,
        registry_path=args.registry,
        dataset_id="robocasa",
        units=all_units,
        stage=args.stage,
        platform_camera_ids=cameras,
    )
    entry = None
    if args.model_id:
        entry = reference_model_preflight(args.roster, args.model_id, suite="atomic_seen")

    expected = {k: v for k, v in {
        "checkpoint_sha256": args.checkpoint_sha256,
        "model_revision": args.model_revision,
        "inventory_sha256": args.inventory_sha256,
    }.items() if v}
    policy_adapter = GenericSocketPolicyAdapter(
        host=args.policy_host, port=policy_port(args),
        timeout_seconds=args.policy_timeout, environment_cameras=cameras,
        expected_identity=expected or None,
    )
    identity = policy_adapter.connect()
    declaration = resolve_runtime_contract(
        identity, platform_camera_ids=cameras, suite="atomic_seen", entry=entry,
        platform_action_dimension=ROBOCASA_ACTION_DIM,
    )
    profile = execution_profile_for(args.execution_profile, declaration)

    specification = CohortSpecification(
        benchmark="robocasa",
        suite="atomic_seen",
        dataset_id="robocasa",
        dataset_revision=revision,
        tasks=tasks,
        episode_indices=tuple(sorted({unit[1] for unit in all_units})),
        platform_camera_ids=cameras,
        onset_fractions=onsets,
        official_horizon=horizon,
        execution_profile=profile.name,
        protocol_version=PROTOCOL_VERSION,
    )
    official = None
    if args.official:
        from mail_bench.suite import scenes_per_task

        official = OfficialProfile(
            suite="atomic_seen",
            tasks=tuple(task_ids()),
            episode_indices=tuple(range(scenes_per_task())),
            onset_fractions=tuple(protocol["onset_definition"]["fractions"]),
            platform_camera_ids=cameras,
            official_horizon=horizon,
            dataset_id="robocasa",
            dataset_revision=revision,
            # The protocol's own figure, read from the protocol and not from
            # the command line: a run that asks for another count is then a
            # recorded deviation rather than an official run by definition.
            healthy_replicates=int(healthy_arm.get("official_replicates_per_seed")
                                   or healthy_arm["replicates_per_seed"]),
        )

    class EnvironmentLease:
        """Hand the runner an adapter it may close without destroying the cache.

        Building a RoboCasa scene costs seconds and every cell of a unit replays
        the same one, so adapters are cached per unit. The rollout kernel closes
        the adapter it is given, which is right in general and wrong for a cached
        one, so the lease forwards everything except the teardown.
        """

        def __init__(self, adapter):
            self._adapter = adapter

        def __getattr__(self, name):
            return getattr(self._adapter, name)

        def close(self) -> None:
            return None

    environments: OrderedDict[tuple[str, int], RoboCasaEnvironmentAdapter] = OrderedDict()

    def environment_for(manifest: FaultManifest):
        # Cached per unit, not per task. Every cell of a unit replays the same
        # scene, so one environment serves the healthy rollout and all nine fault
        # cells; a different unit gets a new one, because RoboCasa resamples
        # placements on reset and a long-lived environment can reach a state
        # where a layout cannot be placed.
        key = (str(manifest.scene_or_task), int(manifest.episode_or_sequence))
        if key not in environments:
            task, index = key
            with contextlib.redirect_stdout(sys.stderr):
                environments[key] = RoboCasaEnvironmentAdapter.from_frozen_scene(
                    bank=bank, task=task, episode_index=index,
                    dataset_revision=revision, split=args.split,
                    seed=episode_seed("robocasa", task, index, PROTOCOL_VERSION),
                    official_horizon=horizons[task],
                )
            # The driver runs a unit's cells together and then moves on, so
            # anything older than the last couple of units is finished with.
            # Without eviction the healthy phase would hold one live MuJoCo
            # environment per scene -- hundreds of them.
            while len(environments) > 2:
                _, stale = environments.popitem(last=False)
                with contextlib.suppress(Exception):
                    stale.close()
        environments.move_to_end(key)
        return EnvironmentLease(environments[key])

    def template(task: str, episode_index: int, faulted, onset_fraction: float) -> FaultManifest:
        return FaultManifest(
            dataset_version=revision,
            split=args.split,
            scene_or_task=task,
            episode_or_sequence=episode_index,
            camera_ids=list(cameras),
            faulted_camera_ids=list(faulted),
            perturbation_family="availability",
            fault_mode=args.fault_mode,
            onset_fraction=onset_fraction,
            duration_fraction=1.0,
            # Only the temporal modes derive a step count from the control rate; the
            # other manifests do not carry it, so their identity is unchanged.
            nominal_control_hz=(control_hz if args.fault_mode in ("stale_k", "burst_dropout") else None),
            stale_k=stale_k,
            burst_frames=burst_frames,
            available_rate=args.burst_available_rate if args.fault_mode == "burst_dropout" else None,
            environment_seed=episode_seed("robocasa", task, episode_index, PROTOCOL_VERSION),
            policy_seed=policy_seed("robocasa", task, episode_index, PROTOCOL_VERSION),
            fault_seed=fault_seed(
                "robocasa", task, episode_index, PROTOCOL_VERSION,
                args.fault_mode, tuple(faulted), onset_fraction,
            ),
            official_horizon=horizons[task],
            adapter_version="robocasa-adapter-v1",
            protocol_version=PROTOCOL_VERSION,
            # A template, not a frozen manifest: the driver replaces both from
            # this model's own healthy reference before any fault cell runs, and
            # there is no fallback if it cannot.
            onset_basis="not_applicable",
            healthy_reference_success=False,
            onset_reference_hash=semantic_sha256({"onset_reference": "pending"}),
            onset_fallback=False,
            # Every semantic state is ranked, All Vision Missing included: it is
            # three of the ten conditions, not a floor beside them.
            subset_ranking_eligible=True,
        )

    paired = PairedCellRunner(
        environment_factory=environment_for,
        policy_factory=lambda arm, manifest: policy_adapter,
        config=RunnerConfig(
            max_environment_steps=horizon,
            max_actions_per_query=profile.max_actions_per_query,
        ),
        execution=authorization,
        writer=AtomicResultWriter(args.output_dir / "cells"),
    )
    arm = MethodArmExecution.from_config(
        args.model_id or declaration.model_id,
        declaration.checkpoint_sha256,
        {
            "model_id": declaration.model_id,
            "suite": "atomic_seen",
            "policy_visible_cameras": list(declaration.policy_visible_cameras),
            "availability_consumed_by_policy": declaration.availability_consumed_by_policy,
            "predicted_action_chunk": declaration.predicted_action_chunk,
            "native_execution_horizon": declaration.native_execution_horizon,
            "reset_semantics": declaration.reset_semantics,
            "training_regime": declaration.training_regime,
            "training_disclosure": dict(declaration.training_disclosure),
            "execution_profile": profile.name,
        },
    )
    runner = CohortRunner(
        specification=specification,
        platform=platform,
        declaration=declaration,
        profile=profile,
        paired=paired,
        arm=arm,
        manifest_template=template,
        healthy_replicates=replicates,
        official_profile=official,
        enforce_admission=args.enforce_admission,
        worker_index=args.worker_index,
        workers=args.workers,
        semantic_platform=PLATFORM,
        fault_duration_fraction=args.duration_fraction,
        fault_recovery_fraction=args.recovery_fraction,
        runtime_metadata={
            # Connection and placement are provenance, never identity.
            "policy_endpoint": f"{args.policy_host}:{policy_port(args)}",
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "mujoco_egl_device_id": os.environ.get("MUJOCO_EGL_DEVICE_ID"),
            "phase": args.phase,
            "scene_bank": str(args.scene_bank),
            "scene_namespace": args.namespace,
            "runtime_fingerprint": collect_runtime_fingerprint(ROOT),
            "policy_server_identity": dict(identity),
        },
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    try:
        if args.phase == "healthy":
            outcomes = runner.run_healthy_phase(my_units)
            # The same identity block as the fault certificate: the onset
            # manifest is built from this phase's certificate alone.
            report = runner.healthy_certificate(my_units, outcomes)
        elif args.phase == "fault":
            admission = runner.resolve_admission(all_units)
            outcomes = runner.run_fault_phase(my_units, admission)
            cohort = runner._report(my_units, all_units, outcomes, admission)
            report = {"phase": args.phase, **cohort.certificate}
        else:
            cohort = runner.run(my_units, all_units)
            report = {"phase": args.phase, **cohort.certificate}
    finally:
        with contextlib.suppress(Exception):
            policy_adapter.close()
        for environment in environments.values():
            with contextlib.suppress(Exception):
                environment.close()

    path = args.output_dir / f"certificate_{args.phase}_worker{args.worker_index}.json"
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True, default=str) + "\n",
                         encoding="utf-8")
    temporary.replace(path)
    print(f"certificate={path}", flush=True)


if __name__ == "__main__":
    main()
