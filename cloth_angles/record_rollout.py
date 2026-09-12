"""Record a scripted ClothFoldEnv rollout to mp4, for visually inspecting the
still/fold/recovery episode kinds collected by collect_episodes.py.

Usage:
    python -m cloth_angles.record_rollout --kind still --seed 0
    python -m cloth_angles.record_rollout --kind fold --seed 0 --max-steps 200
    python -m cloth_angles.record_rollout --kind recovery --seed 0

Uses the exact same still_action/fold_action/recovery_action policies as
collect_episodes.py, so the recorded video matches what training data of
that kind actually looks like.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import imageio
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "mujuco"))

from cloth_angles.collect_episodes import EPISODE_KINDS, fold_action, recovery_action, still_action


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", required=True, choices=EPISODE_KINDS)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--fps", type=int, default=20, help="Matches ClothFoldEnv.metadata['render_fps']")
    parser.add_argument("--image-size", type=int, default=480,
                         help="Render resolution (square), for video quality only -- unrelated to training data")
    parser.add_argument("--output", default=None, help="Output mp4 path (default: outputs/cloth_angles/videos/<kind>.mp4)")
    return parser.parse_args()


def make_render_env(max_episode_steps: int, image_size: int):
    from sim_main import ClothFoldEnv  # noqa: PLC0415 -- mujuco/ only importable once sys.path is set above
    return ClothFoldEnv(max_episode_steps=max_episode_steps, observation_mode="state",
                         image_size=(image_size, image_size))


def main():
    args = parse_args()
    output_path = Path(args.output) if args.output else Path("outputs/cloth_angles/videos") / f"{args.kind}.mp4"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    env = make_render_env(args.max_steps, args.image_size)
    rng = np.random.default_rng(args.seed)
    env.reset(seed=args.seed)
    perturb_steps = int(rng.integers(1, max(2, args.max_steps // 3)))

    frames = [env._render_image()[0]]
    fold_score_final = 0.0
    success = False

    for t in range(args.max_steps):
        if args.kind == "still":
            action = still_action(env, rng, t)
        elif args.kind == "fold":
            action = fold_action(env, rng, t)
        else:
            action = recovery_action(env, rng, t, perturb_steps)

        obs, reward, terminated, truncated, info = env.step(action)
        frames.append(env._render_image()[0])
        fold_score_final = info["fold_score"]
        success = info["success"]
        if terminated or truncated:
            break

    imageio.mimwrite(output_path, frames, fps=args.fps)

    print(f"kind:        {args.kind}")
    print(f"steps:       {len(frames) - 1}")
    print(f"fold_score:  {fold_score_final:.4f}")
    print(f"success:     {success}")
    print(f"video:       {output_path}")


if __name__ == "__main__":
    main()
