"""Run a policy against a Gymnasium environment 

Usage:
    python scripts/run_policy.py --env MuJoCoTouch-v1 --policy random --episodes 5

Swapping policies later (e.g. --policy ppo) requires no changes here — only
registering the new policy in policy_runner/loader.py.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import gymnasium as gym

from policy_runner.loader import load_policy


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env", required=True, help="Gymnasium env id, e.g. MuJoCoTouch-v1")
    parser.add_argument("--policy", required=True, help="Registered policy name, e.g. random")
    parser.add_argument("--episodes", type=int, default=5, help="Number of episodes to run")
    parser.add_argument("--max-steps", type=int, default=500, help="Max steps per episode")
    parser.add_argument("--seed", type=int, default=0, help="Seed for env reset")
    return parser.parse_args()


def run_episode(env, policy, seed, max_steps):
    obs, info = env.reset(seed=seed)
    policy.reset()

    total_reward = 0.0
    length = 0
    success = None

    for _ in range(max_steps):
        action = policy.act(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += float(reward)
        length += 1

        if "success" in info:
            success = bool(info["success"]) or bool(success)

        if terminated or truncated:
            break

    return total_reward, length, success


def register_env_backends(env_id):
    """Some envs (e.g. so101-nexus) only register with Gymnasium as a side
    effect of importing their package, so import known backends here first."""
    if env_id.startswith("MuJoCo"):
        import so101_nexus.mujoco  # noqa: F401
    elif env_id.startswith("Warp"):
        import so101_nexus.warp  # noqa: F401


def main():
    args = parse_args()

    register_env_backends(args.env)
    env = gym.make(args.env)
    policy = load_policy(args.policy, env.action_space, env.observation_space)

    rewards = []
    lengths = []
    successes = []

    for ep in range(args.episodes):
        reward, length, success = run_episode(env, policy, args.seed + ep, args.max_steps)
        rewards.append(reward)
        lengths.append(length)
        if success is not None:
            successes.append(success)
        print(f"episode {ep + 1}/{args.episodes}: reward={reward:.4f} length={length} success={success}")

    env.close()

    avg_reward = sum(rewards) / len(rewards)
    avg_length = sum(lengths) / len(lengths)

    print("\n=== summary ===")
    print(f"env:            {args.env}")
    print(f"policy:         {args.policy}")
    print(f"episodes:       {args.episodes}")
    print(f"avg reward:     {avg_reward:.4f}")
    print(f"avg length:     {avg_length:.2f}")
    if successes:
        print(f"success rate:   {sum(successes) / len(successes):.2%}")
    else:
        print("success rate:   n/a (env does not provide info['success'])")


if __name__ == "__main__":
    main()
