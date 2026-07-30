#!/usr/bin/env python3
"""Record real MolmoAct2 rollouts on a SO101-Nexus env to mp4.

Mirrors the env/policy setup of scripts/smoke_molmoact_import.py and the
stepping semantics of so101_nexus RolloutRecorder (degree-unit policy actions
converted to radians and clipped before env.step), but renders every frame so
the rollout can be watched.

Usage:
    python scripts/record_molmoact_rollout.py --seed 0 --max-steps 100 \
        --dtype bfloat16 --device-map auto --disable-cuda-graph
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import imageio
import numpy as np

from scripts.smoke_molmoact_import import (
    apply_predict_action_compat,
    dtype_from_name,
    import_backend_for_env_id,
    make_env,
    to_jsonable,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env", default="MuJoCoTouch-v1")
    parser.add_argument("--repo-id", default="allenai/MolmoAct2-SO100_101")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--width", type=int, default=160)
    parser.add_argument("--height", type=int, default=120)
    parser.add_argument("--chunk-size", type=int, default=8)
    parser.add_argument("--num-steps", type=int, default=10)
    parser.add_argument("--disable-cuda-graph", action="store_true")
    parser.add_argument("--device-map", default=None)
    parser.add_argument(
        "--dtype",
        choices=("none", "float32", "float16", "bfloat16"),
        default="bfloat16",
    )
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument(
        "--output",
        default=None,
        help="Output mp4 path (default: outputs/videos/<env>_molmoact_seed<seed>.mp4)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_path = (
        Path(args.output)
        if args.output
        else Path("outputs/videos") / f"{args.env}_molmoact_seed{args.seed}.mp4"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    import_backend_for_env_id(args.env)
    from so101_nexus.policy_adapters import MolmoActPolicy

    print("loading model (this can take a while)...", flush=True)
    load_start = time.monotonic()
    policy = MolmoActPolicy.from_pretrained(
        args.repo_id,
        device=args.device_map,
        dtype=dtype_from_name(args.dtype),
        chunk_size=args.chunk_size,
        num_steps=args.num_steps,
        enable_cuda_graph=not args.disable_cuda_graph,
    )
    compat = apply_predict_action_compat(policy)
    print(f"model loaded in {time.monotonic() - load_start:.1f}s (compat shim: {compat})", flush=True)

    env = make_env(args.env, width=args.width, height=args.height)
    try:
        obs, _ = env.reset(seed=args.seed)
        policy.reset()
        task = getattr(env.unwrapped, "task_description", "")

        frames = [env.render()]
        total_reward = 0.0
        success = False
        info: dict = {}
        step_times: list[float] = []

        for step in range(args.max_steps):
            batch = {
                "observation.state": np.rad2deg(obs["state"]).astype(np.float32),
                "task": task,
                "observation.images.overhead": obs["overhead_camera"],
                "observation.images.wrist": obs["wrist_camera"],
            }
            step_start = time.monotonic()
            action_deg = np.asarray(policy.select_action(batch), dtype=np.float32)
            step_times.append(time.monotonic() - step_start)

            action_rad = np.clip(
                np.deg2rad(action_deg),
                env.action_space.low,
                env.action_space.high,
            ).astype(np.float32)
            obs, reward, terminated, truncated, info = env.step(action_rad)
            frames.append(env.render())
            total_reward += float(reward)
            success = success or bool(info.get("success", False))

            dist = info.get("tcp_to_obj_dist")
            print(
                f"step {step + 1}/{args.max_steps}"
                f" policy_time={step_times[-1]:.1f}s"
                f" dist={dist if dist is None else f'{dist:.4f}'}"
                f" success={success}",
                flush=True,
            )
            if terminated or truncated:
                break
    finally:
        env.close()

    imageio.mimwrite(output_path, frames, fps=args.fps)

    summary = {
        "env": args.env,
        "policy": "molmoact",
        "repo_id": args.repo_id,
        "seed": args.seed,
        "task": task,
        "steps": len(frames) - 1,
        "reward": round(total_reward, 4),
        "success": success,
        "final_info": to_jsonable(info),
        "video": str(output_path),
        "chunk_size": args.chunk_size,
        "mean_model_call_time_s": round(
            float(np.mean([t for t in step_times if t > 0.05])) if step_times else 0.0, 2
        ),
    }
    summary_path = output_path.with_suffix(".json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")

    for key in ("env", "seed", "steps", "reward", "success", "video"):
        print(f"{key}: {summary[key]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
