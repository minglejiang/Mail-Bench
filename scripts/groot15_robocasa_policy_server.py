#!/usr/bin/env python3
"""Serve the RoboCasa365 team's GR00T N1.5 checkpoint over the public MAIL-Bench protocol.

This speaks ``mcr-policy-v1``, the same contract any third party implements, so
it needs no change to the execution kernel.  The canonical observation
arrives untranslated and everything model-specific happens here: naming the
three cameras into the checkpoint's slots, slicing the flat proprioceptive
vector into the named keys the processor normalises, the substitution for an
absent camera, and flattening the five action groups back into RoboCasa's
twelve-dimensional action.

Every name below was read off the checkpoint's own ``experiment_cfg/metadata.json``
and off the robocasa-benchmark fork of Isaac-GR00T (``PandaOmronDataConfig`` at
9d7d7a9e), which is the code that produced and evaluated this checkpoint, and
the server checks them against the checkpoint it loads.  The state keys the embodiment
consumes are the five relative keys RoboCasa's own gym wrapper publishes --
the same ones this repository's adapter concatenates -- so no raw robosuite
proprioception is needed.  The order differs (the processor lists the gripper
third, the platform lists it last); that is why the vector is sliced by name
here rather than passed through.

**Naming the action slots is not done here.**  A flat action vector leaves this
server exactly as it would leave any other policy, and the RoboCasa adapter
names the slots on the way into the gym environment.  Flattening GR00T's five
groups in the adapter's slot order is the inverse of that naming, and the
adapter's own table is used so the two cannot drift apart.

A missing camera arrives as ``None`` with ``availability=false``.  GR00T has a
fixed set of image slots and no notion of an absent one, so the server
substitutes a frozen zero tensor.  That is this model's preprocessing, not the
protocol's: an availability-aware policy is free to see the absence and act on
it, which is why the kernel does not fill it in on anyone's behalf.

Images: NVIDIA's evaluator renders 512x512 and downsamples to 256 with
INTER_AREA; the frozen environment renders 256 natively, so the same function
is applied and is the identity here.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import random
import socket
import time
import traceback


ROOT = Path(__file__).resolve().parents[1]
#: The canonical camera ids RoboCasa publishes, mapped to the checkpoint's
#: image slots. From PandaOmronKeyConverter.get_camera_config in the fork
#: NVIDIA evaluates with: side_0 is the left agent view, side_1 the right,
#: wrist_0 the eye-in-hand camera.
CAMERA_TO_SLOT = {
    "robot0_agentview_left": "robot0_agentview_left",
    "robot0_agentview_right": "robot0_agentview_right",
    "robot0_eye_in_hand": "robot0_eye_in_hand",
}
IMAGE_RES = 256
ACTION_DIM = 12
#: Actions executed before the policy is queried again: the fork's own
#: evaluation client default (the fork's own scripts/run_eval.py, --n_action_steps
#: at 9d7d7a9e; that file is upstream's, not part of this repository).
NATIVE_EXECUTION_HORIZON = 16
EMBODIMENT = "new_embodiment"
DATA_CONFIG = "panda_omron"
LANGUAGE_KEY = "annotation.human.action.task_description"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, required=True,
                        help="GR00T checkpoint directory (the verified snapshot)")
    parser.add_argument("--model-id", default="groot_n1_5_robocasa")
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--inventory-sha256", required=True,
                        help="digest of the verified checkpoint snapshot")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7614)
    parser.add_argument("--benchmark-root", type=Path, default=ROOT)
    parser.add_argument("--profile-latency", action="store_true")
    return parser.parse_args()


def process_image(np, cv2, frame):
    """NVIDIA's ``GrootRoboCasaEnv.process_img``: pad to square, resize to 256."""
    image = np.ascontiguousarray(frame)
    height, width = image.shape[:2]
    if height != width:
        side = max(height, width)
        pad_y, pad_x = (side - height) // 2, (side - width) // 2
        image = np.pad(image, ((pad_y, pad_y), (pad_x, pad_x), (0, 0)))
        height = width = side
    if (height, width) != (IMAGE_RES, IMAGE_RES):
        image = cv2.resize(image, (IMAGE_RES, IMAGE_RES), interpolation=cv2.INTER_AREA)
    return np.ascontiguousarray(image, dtype=np.uint8)


def main() -> None:
    args = parse_args()
    checkpoint = args.checkpoint.resolve()
    if not checkpoint.is_dir():
        raise SystemExit(f"missing checkpoint directory: {checkpoint}")
    # The snapshot is what was verified; nothing may be fetched to complete it.
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    import cv2
    import numpy as np
    import torch

    # Pairing has to survive a server restart: deterministic kernels, no
    # cuDNN autotuning, recorded in the runtime fingerprint below.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    from gr00t.experiment.data_config import DATA_CONFIG_MAP
    from gr00t.model.policy import Gr00tPolicy

    from mail_bench.artifacts import inventory_digest
    from mail_bench.manifest import semantic_sha256
    from mail_bench.net import recv_message, send_message
    from mail_bench.platforms.robocasa import (
        ROBOCASA_ACTION_DIM,
        ROBOCASA_ACTION_SLOTS,
        ROBOCASA_STATE_DIM,
        ROBOCASA_STATE_LAYOUT,
        convert_action,
    )
    from mail_bench.runtime import collect_runtime_fingerprint

    # The transcribed action interface is checked against the installed RoboCasa
    # where one is installed. This runtime is GR00T's, not the simulator's, so
    # the check may be unavailable here; the rollout process, which does the
    # naming, carries its own copy of the check through the adapter.
    action_interface_verified = False
    try:
        from robocasa.utils.env_utils import convert_action as upstream_convert_action
    except ImportError:
        upstream_convert_action = None
    if upstream_convert_action is not None:
        probe = np.arange(ROBOCASA_ACTION_DIM, dtype=np.float32)
        ours, theirs = convert_action(probe), upstream_convert_action(probe)
        if set(ours) != set(theirs) or any(
            not np.array_equal(ours[key], theirs[key]) for key in ours
        ):
            raise SystemExit(
                "the transcribed RoboCasa action interface disagrees with the "
                f"installed robocasa: {sorted(ours)} vs {sorted(theirs)}"
            )
        action_interface_verified = True

    # Before the load, not only after: digesting the tree costs seconds and a
    # mistake in the arguments should not cost a model load to discover. The
    # digest covers the file tree only; dataset_id and revision label the
    # record and do not enter the comparison.
    before = inventory_digest(checkpoint, dataset_id=args.model_id,
                              revision=args.model_revision)
    if before != args.inventory_sha256:
        raise SystemExit(
            f"checkpoint inventory is {before[:12]}... on disk, but "
            f"{args.inventory_sha256[:12]}... was verified"
        )

    # Read the modality spec off the checkpoint before loading it, and refuse a
    # checkpoint whose embodiment is not the one this server is written for.
    metadata = json.loads((checkpoint / "experiment_cfg" / "metadata.json").read_text())
    if EMBODIMENT not in metadata:
        raise SystemExit(f"this checkpoint has no {EMBODIMENT!r} embodiment: {list(metadata)}")
    modalities = metadata[EMBODIMENT]["modalities"]
    statistics = metadata[EMBODIMENT]["statistics"]
    expected_slots = sorted(CAMERA_TO_SLOT.values())
    if sorted(modalities["video"]) != expected_slots:
        raise SystemExit(f"the checkpoint's video keys are {sorted(modalities['video'])}, "
                         f"this server maps cameras onto {expected_slots}")
    for slot, spec in modalities["video"].items():
        if list(spec.get("resolution", [])) != [IMAGE_RES, IMAGE_RES]:
            raise SystemExit(f"{slot} was recorded at {spec.get('resolution')}, not {IMAGE_RES}")
    state_layout = [(key.split(".", 1)[1], width) for key, width in ROBOCASA_STATE_LAYOUT]
    if sorted(k for k, _ in state_layout) != sorted(modalities["state"]):
        raise SystemExit(f"the checkpoint's state keys are {sorted(modalities['state'])}, "
                         f"the platform publishes {[k for k, _ in state_layout]}")
    for key, width in state_layout:
        recorded = list(modalities["state"][key]["shape"])[0]
        if recorded != width or len(statistics["state"][key]["mean"]) != width:
            raise SystemExit(f"the checkpoint records state.{key} with {recorded} values, "
                             f"the platform publishes {width}")
    action_groups = [name.split(".", 1)[1] for name, _, _ in ROBOCASA_ACTION_SLOTS]
    if sorted(action_groups) != sorted(modalities["action"]):
        raise SystemExit(f"the checkpoint's action groups are {sorted(modalities['action'])}, "
                         f"the platform's slots are {action_groups}")
    for name, start, stop in ROBOCASA_ACTION_SLOTS:
        recorded = list(modalities["action"][name.split(".", 1)[1]]["shape"])[0]
        if recorded != stop - start:
            raise SystemExit(f"the checkpoint predicts {name} with {recorded} values, "
                             f"the platform's slot is {stop - start} wide")
    data_config = DATA_CONFIG_MAP[DATA_CONFIG]
    if sorted(data_config.video_keys) != sorted(f"video.{s}" for s in CAMERA_TO_SLOT.values()):
        raise SystemExit(f"data config {DATA_CONFIG!r} reads {data_config.video_keys}")
    if [k.split(".", 1)[1] for k in data_config.action_keys] != action_groups:
        raise SystemExit(f"data config {DATA_CONFIG!r} orders actions {data_config.action_keys}, "
                         f"the platform's slots are {action_groups}")
    predicted_chunk = len(data_config.action_indices)
    LANGUAGE_KEY_LOCAL = data_config.language_keys[0] if getattr(data_config, "language_keys", None) else LANGUAGE_KEY

    started = time.perf_counter()
    policy = Gr00tPolicy(
        model_path=str(checkpoint), embodiment_tag=EMBODIMENT,
        modality_config=data_config.modality_config(),
        modality_transform=data_config.transform(), device=args.device,
    )
    # Loading may rewrite a checkpoint directory, so the same digest is taken
    # again: what is served has to be what was verified.
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
        "device": args.device,
        "torch_deterministic_algorithms": True,
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "model_load_seconds": time.perf_counter() - started,
        "action_interface_verified_against_installed_robocasa": action_interface_verified,
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
        "policy_visible_cameras": sorted(CAMERA_TO_SLOT),
        # GR00T receives a zero tensor for an absent camera and is never told
        # that it is absent, so it cannot act on availability.
        "availability_consumed_by_policy": False,
        # Measured from the processor's action delta indices for this
        # embodiment, not from config.json's action_horizon, which is the
        # model's maximum across embodiments.
        "predicted_action_chunk": predicted_chunk,
        "native_execution_horizon": NATIVE_EXECUTION_HORIZON,
        "action_dimension": ACTION_DIM,
        "stateful_policy": False,
        "reset_semantics": "stateless",
        "policy_sampling_seeded": True,
        # Deterministic torch kernels, so a fresh process gives the same actions.
        "deterministic_across_processes": True,
        "training_regime": "clean_trained",
        "category": "dual_system_vla",
        "embodiment_tag": EMBODIMENT,
        "data_config": DATA_CONFIG,
        "runtime_fingerprint": runtime,
    }

    #: Frozen once, so every absent frame in every cell is the same tensor.
    ABSENT = np.zeros((IMAGE_RES, IMAGE_RES, 3), dtype=np.uint8)

    def observation_from(message):
        cameras = message["cameras"]
        availability = message["availability"]
        if set(cameras) != set(CAMERA_TO_SLOT):
            raise ValueError(
                f"expected cameras {sorted(CAMERA_TO_SLOT)}, got {sorted(cameras)}"
            )
        state = np.asarray(message["robot_state"], dtype=np.float32)
        if state.shape != (ROBOCASA_STATE_DIM,):
            raise ValueError(f"RoboCasa state must have shape ({ROBOCASA_STATE_DIM},)")
        observation = {}
        for camera, slot in CAMERA_TO_SLOT.items():
            frame = cameras[camera]
            if frame is None and availability[camera]:
                raise ValueError(f"{camera} is marked available but carries no frame")
            image = ABSENT if frame is None else process_image(np, cv2, frame)
            observation[f"video.{slot}"] = image[None]                 # (T=1, H, W, 3)
        offset = 0
        for key, width in state_layout:
            observation[f"state.{key}"] = state[offset:offset + width][None]   # (T=1, D)
            offset += width
        # A 0-d string array, and nothing else: after Gr00tPolicy's batch axis
        # and the transform's x[0], it is the only form that renders as the bare
        # instruction, which is the training text and what RoboCasa's own
        # evaluator produces; a list or 1-d array renders with brackets and
        # quotes, and a bare str fails at x[0].
        observation[LANGUAGE_KEY_LOCAL] = np.array(str(message["language"]))
        return observation

    def flatten(actions):
        columns = []
        for name, start, stop in ROBOCASA_ACTION_SLOTS:
            group = name.split(".", 1)[1]
            if group in actions:
                value = actions[group]
            elif name in actions:
                value = actions[name]
            else:
                raise ValueError(f"policy returned no {group!r}; keys {sorted(actions)}")
            value = np.asarray(value, dtype=np.float32)
            if value.ndim == 3:                                  # (B=1, T, D)
                value = value[0]
            if value.shape != (predicted_chunk, stop - start):
                raise ValueError(
                    f"{group} came back as {value.shape}, expected "
                    f"({predicted_chunk}, {stop - start})"
                )
            columns.append(value)
        return np.concatenate(columns, axis=1).reshape(-1, ACTION_DIM)

    def predict(message):
        actions = policy.get_action(observation_from(message))
        return flatten(actions)

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
    print(f"GR00T N1.5 RoboCasa server ready on {args.host}:{args.port}; "
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
                            seed = int(message["seed"]) % (1 << 32)
                            random.seed(seed)
                            np.random.seed(seed)
                            # The flow head samples its noise through torch.
                            torch.manual_seed(seed)
                            torch.cuda.manual_seed_all(seed)
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
