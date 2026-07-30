"""Record a policy rollout to an mp4, for visually comparing policies.

Usage:
    python scripts/record_rollout.py --env MuJoCoTouch-v1 --policy random
    python scripts/record_rollout.py --env MuJoCoTouch-v1 --policy ppo \
        --checkpoint outputs/ppo/MuJoCoTouch-v1/model.zip
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import gymnasium as gym
import imageio

from policy_runner.loader import load_policy


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env", required=True, help="Gymnasium env id, e.g. MuJoCoTouch-v1")
    parser.add_argument("--policy", required=True, help="Registered policy name, e.g. random")
    parser.add_argument("--max-steps", type=int, default=500, help="Max steps to record")
    parser.add_argument("--seed", type=int, default=0, help="Seed for env reset")
    parser.add_argument(
        "--checkpoint", default=None, help="Checkpoint path, for policies that need one (e.g. ppo)"
    )
    parser.add_argument("--fps", type=int, default=30, help="Playback frames per second")
    parser.add_argument("--output", default=None, help="Output mp4 path (default: outputs/videos/<env>_<policy>.mp4)")
    return parser.parse_args()


def register_env_backends(env_id):
    """Some envs (e.g. so101-nexus) only register with Gymnasium as a side
    effect of importing their package, so import known backends here first."""
    if env_id.startswith("MuJoCo"):
        import so101_nexus.mujoco  # noqa: F401
    elif env_id.startswith("Warp"):
        import so101_nexus.warp  # noqa: F401


def main():
    args = parse_args()
    output_path = Path(args.output) if args.output else Path("outputs/videos") / f"{args.env}_{args.policy}.mp4"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    register_env_backends(args.env)
    env = gym.make(args.env, render_mode="rgb_array")
    policy_kwargs = {"checkpoint": args.checkpoint} if args.checkpoint else {}
    policy = load_policy(args.policy, env.action_space, env.observation_space, **policy_kwargs)

    obs, info = env.reset(seed=args.seed)
    policy.reset()

    frames = [env.render()]
    total_reward = 0.0
    success = None

    for _ in range(args.max_steps):
        action = policy.act(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        frames.append(env.render())
        total_reward += float(reward)

        if "success" in info:
            success = bool(info["success"]) or bool(success)

        if terminated or truncated:
            break

    env.close()

    imageio.mimwrite(output_path, frames, fps=args.fps)

    print(f"env:        {args.env}")
    print(f"policy:     {args.policy}")
    print(f"steps:      {len(frames) - 1}")
    print(f"reward:     {total_reward:.4f}")
    print(f"success:    {success}")
    print(f"video:      {output_path}")


if __name__ == "__main__":
    main()
