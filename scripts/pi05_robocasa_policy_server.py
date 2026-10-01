#!/usr/bin/env python3
"""Serve the frozen RoboCasa PI0.5 checkpoint over the public MAIL-Bench protocol.

This speaks ``mcr-policy-v1``, the same contract any third party implements, so
it needs no change to the execution kernel.  The canonical observation
arrives untranslated and everything model-specific happens here: the resize to
224, the openpi transform chain, the substitution for an absent camera, and the
denormalisation back to RoboCasa's twelve-dimensional action.

**Naming the action slots is not done here.**  A flat action vector leaves this
server exactly as it would leave a policy for any other platform, and the
RoboCasa adapter names the slots on the way into the gym environment.  Putting the naming
here would make one platform's action dict part of what a submission has to
know.

A missing camera arrives as ``None`` with ``availability=false``.  PI0.5 has a
fixed input shape and no notion of an absent slot, so the server substitutes a
frozen zero tensor.  That is this model's preprocessing, not the protocol's: an
availability-aware policy is free to see the absence and act on it, which is why
the kernel does not fill it in on anyone's behalf.
"""

from __future__ import annotations

import argparse
import dataclasses
import os
from pathlib import Path
import random
import socket
import time
import traceback


ROOT = Path(__file__).resolve().parents[1]
#: The canonical camera ids RoboCasa publishes, mapped to the openpi element
#: keys the RoboCasa PI0.5 checkpoint was trained against.
CAMERA_TO_ELEMENT = {
    "robot0_agentview_left": "observation/image",
    "robot0_eye_in_hand": "observation/wrist_image",
    "robot0_agentview_right": "observation/right_image",
}
RESIZE = 224
STATE_DIM = 16
ACTION_DIM = 12
#: Actions executed before the policy is queried again: the interval
#: RoboCasa365's own evaluation uses, not the predicted chunk length.
NATIVE_EXECUTION_HORIZON = 5
#: Only if the loaded policy does not expose its own horizon.
ACTION_HORIZON_FALLBACK = 50


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, required=True,
                        help="openpi checkpoint directory (params only)")
    parser.add_argument("--model-id", default="pi05_robocasa")
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--inventory-sha256", required=True,
                        help="digest of the verified checkpoint snapshot")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7606)
    parser.add_argument("--benchmark-root", type=Path, default=ROOT)
    parser.add_argument("--profile-latency", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint = args.checkpoint.resolve()
    if not checkpoint.is_dir():
        raise SystemExit(f"missing checkpoint directory: {checkpoint}")

    # Pairing has to survive a server restart: the healthy and fault phases
    # start their own processes, and so does any crash recovery. Left to
    # itself XLA autotunes GEMM kernels per process and two fresh servers give
    # different actions for the same input, so autotuning is off and
    # deterministic ops on -- before anything imports jax. Recorded in the
    # runtime fingerprint, and verified by the cross-process determinism check.
    xla_flags = "--xla_gpu_autotune_level=0 --xla_gpu_deterministic_ops=true"
    inherited = os.environ.get("XLA_FLAGS", "")
    for flag in ("--xla_gpu_autotune_level", "--xla_gpu_deterministic_ops"):
        if flag in inherited and flag + "=" + xla_flags.split(flag + "=")[1].split()[0] not in inherited:
            # XLA takes the last occurrence; an environment that set the
            # opposite would win quietly and the fingerprint would record both.
            raise SystemExit(f"XLA_FLAGS in the environment sets {flag} against the "
                             f"determinism this server requires: {inherited!r}")
    os.environ["XLA_FLAGS"] = (inherited + " " + xla_flags).strip()
    import jax
    import numpy as np
    import openpi.transforms as _transforms
    import openpi.policies.robocasa_policy as robocasa_policy
    from openpi.models import pi0_config
    from openpi.policies import policy_config as _policy_config
    from openpi.training.config import (
        DataConfig,
        DataConfigFactory,
        ModelTransformFactory,
        TrainConfig,
    )
    from openpi_client import image_tools

    from mail_bench.artifacts import inventory_digest
    from mail_bench.manifest import semantic_sha256
    from mail_bench.net import recv_message, send_message
    from mail_bench.platforms.robocasa import ROBOCASA_ACTION_DIM, convert_action
    from mail_bench.runtime import collect_runtime_fingerprint

    # The transcribed action interface must agree with the RoboCasa that is
    # actually installed, or a rollout would be naming the wrong slots.
    from robocasa.utils.env_utils import convert_action as upstream_convert_action

    probe = np.arange(ROBOCASA_ACTION_DIM, dtype=np.float32)
    ours, theirs = convert_action(probe), upstream_convert_action(probe)
    if set(ours) != set(theirs) or any(
        not np.array_equal(ours[key], theirs[key]) for key in ours
    ):
        raise SystemExit(
            "the transcribed RoboCasa action interface disagrees with the installed "
            f"robocasa: {sorted(ours)} vs {sorted(theirs)}"
        )

    @dataclasses.dataclass(frozen=True)
    class InferenceRobocasaDataConfig(DataConfigFactory):
        def create(self, assets_dirs, model_config) -> DataConfig:
            return DataConfig(
                asset_id=None,
                data_transforms=_transforms.Group(
                    inputs=[robocasa_policy.RobocasaInputs(
                        action_dim=model_config.action_dim,
                        model_type=model_config.model_type)],
                    outputs=[robocasa_policy.RobocasaOutputs()]),
                model_transforms=ModelTransformFactory()(model_config))

    # Before the load, not only after: a mistake in the arguments should not
    # cost a model load to discover.
    #
    # accessed_on is left at its default: it does not enter inventory_sha256, so
    # supplying a value here would suggest a comparison that is not being made.
    before = inventory_digest(checkpoint, dataset_id=args.model_id,
                              revision=args.model_revision)
    if before != args.inventory_sha256:
        raise SystemExit(
            f"checkpoint inventory is {before[:12]}... on disk, but "
            f"{args.inventory_sha256[:12]}... was verified"
        )

    started = time.perf_counter()
    policy = _policy_config.create_trained_policy(
        TrainConfig(name="pi05_inference",
                    model=pi0_config.Pi0Config(pi05=True, max_token_len=200),
                    data=InferenceRobocasaDataConfig()),
        str(checkpoint),
    )
    if not hasattr(policy, "_rng"):
        raise SystemExit(
            "the loaded openpi Policy has no _rng attribute to re-key on reset; "
            "seeding the flow head's sampling would silently not happen"
        )
    # Loading may rewrite a checkpoint directory, so the same digest is taken
    # again: what is served has to be what was verified, not merely what was on
    # disk before an upstream loader touched it.
    served = inventory_digest(checkpoint, dataset_id=args.model_id,
                              revision=args.model_revision)
    if served != before:
        raise SystemExit(
            f"loading rewrote the checkpoint: inventory is {served[:12]}... "
            f"after the load and was {before[:12]}... before it"
        )

    runtime = collect_runtime_fingerprint(args.benchmark_root)
    runtime.update({
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "xla_flags": os.environ["XLA_FLAGS"],
        "model_load_seconds": time.perf_counter() - started,
    })
    runtime["fingerprint_hash"] = semantic_sha256(
        {key: value for key, value in runtime.items() if key != "fingerprint_hash"}
    )

    identity = {
        "ok": True,
        "protocol": "mcr-policy-v1",
        "model_id": args.model_id,
        "model_revision": args.model_revision,
        "inventory_sha256": args.inventory_sha256,
        "policy_visible_cameras": sorted(CAMERA_TO_ELEMENT),
        # PI0.5 receives a zero tensor for an absent camera and is never told
        # that it is absent, so it cannot act on availability.
        "availability_consumed_by_policy": False,
        # Measured from the model rather than declared: a copied chunk length
        # would be a guess shaped like a measurement.
        "predicted_action_chunk": int(
            getattr(getattr(policy, "_model", None), "action_horizon", 0)
            or ACTION_HORIZON_FALLBACK),
        "native_execution_horizon": NATIVE_EXECUTION_HORIZON,
        "action_dimension": ACTION_DIM,
        "stateful_policy": False,
        "reset_semantics": "stateless",
        # Every source of randomness in the policy, the flow head's sampling
        # key included, is re-keyed from the reset seed.
        "policy_sampling_seeded": True,
        # Same actions from a fresh process: no per-process kernel autotuning.
        "deterministic_across_processes": True,
        "training_regime": "clean_trained",
        "category": "diffusion_chunked_vla",
        "runtime_fingerprint": runtime,
    }

    #: Frozen once, so every absent frame in every cell is the same tensor.
    ABSENT = np.zeros((RESIZE, RESIZE, 3), dtype=np.uint8)

    def prepared(frame):
        if frame is None:
            return ABSENT
        return image_tools.convert_to_uint8(
            image_tools.resize_with_pad(np.ascontiguousarray(frame), RESIZE, RESIZE)
        )

    def element_from(message):
        cameras = message["cameras"]
        availability = message["availability"]
        if set(cameras) != set(CAMERA_TO_ELEMENT):
            raise ValueError(
                f"expected cameras {sorted(CAMERA_TO_ELEMENT)}, got {sorted(cameras)}"
            )
        state = np.asarray(message["robot_state"], dtype=np.float32)
        if state.shape != (STATE_DIM,):
            raise ValueError(f"RoboCasa state must have shape ({STATE_DIM},)")
        element = {"observation/state": state, "prompt": str(message["language"])}
        for camera, key in CAMERA_TO_ELEMENT.items():
            frame = cameras[camera]
            if frame is None and availability[camera]:
                raise ValueError(f"{camera} is marked available but carries no frame")
            element[key] = prepared(frame)
        return element

    def predict(message):
        actions = np.asarray(policy.infer(element_from(message))["actions"])
        actions = actions.reshape(-1, ACTION_DIM).astype(np.float32)
        if actions.shape[0] != identity["predicted_action_chunk"]:
            raise ValueError(
                f"the model returned {actions.shape[0]} actions, the identity declares "
                f"{identity['predicted_action_chunk']}"
            )
        return actions

    def timed(message):
        if not args.profile_latency:
            return {"ok": True, "actions": predict(message)}
        began = time.perf_counter_ns()
        actions = predict(message)
        # Observability only: never enters an action, a hash or a cell identity.
        return {"ok": True, "actions": actions,
                "server_inference_ms": (time.perf_counter_ns() - began) / 1e6}

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((args.host, args.port))
    server.listen(1)
    print(f"PI0.5 RoboCasa server ready on {args.host}:{args.port}; "
          f"inventory={args.inventory_sha256[:12]}", flush=True)
    try:
        while True:
            connection, _ = server.accept()
            connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            with connection:
                while True:
                    try:
                        message = recv_message(connection)
                    except ConnectionError:
                        break
                    kind = message.get("type")
                    if kind == "health":
                        send_message(connection, {"ok": True, "ready": True,
                                                  "status": "ready"})
                    elif kind == "identity":
                        send_message(connection, identity)
                    elif kind == "reset":
                        # A failed reset fails the cell, never the server.
                        try:
                            seed = int(message["seed"])
                            random.seed(seed)
                            np.random.seed(seed % (1 << 32))
                            # The flow head samples its noise from the
                            # policy's own JAX key, which the two lines above
                            # never touch. Re-keying it is what makes the
                            # healthy rollout and every fault arm of a scene
                            # draw the same policy stream. The benchmark's
                            # seeds are 64-bit and jax.random.key takes a C
                            # long, so the same 32-bit reduction as numpy's is
                            # applied; pairing needs equal seeds, not wide ones.
                            policy._rng = jax.random.key(seed % (1 << 32))
                            send_message(connection, {"ok": True})
                        except Exception:                   # noqa: BLE001
                            traceback.print_exc()
                            send_message(connection, {
                                "ok": False, "error": traceback.format_exc()[-4000:]})
                    elif kind == "act":
                        try:
                            send_message(connection, timed(message))
                        except Exception:                   # noqa: BLE001
                            traceback.print_exc()
                            send_message(connection, {
                                "ok": False, "error": traceback.format_exc()[-4000:]})
                    elif kind == "disconnect":
                        send_message(connection, {"ok": True})
                        break
                    elif kind == "shutdown":
                        send_message(connection, {"ok": True})
                        return
                    else:
                        send_message(connection,
                                     {"ok": False, "error": f"unknown type {kind!r}"})
    finally:
        server.close()


if __name__ == "__main__":
    main()
