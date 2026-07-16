"""Train a small PPO baseline on an SO101-Nexus environment.

Usage:
    python scripts/train_ppo.py --env MuJoCoTouch-v1 --timesteps 50000

Produces a checkpoint (.zip) and tensorboard logs under --output-dir. Load the
checkpoint later with policy_runner's "ppo" policy:

    python scripts/run_policy.py --env MuJoCoTouch-v1 --policy ppo \
        --checkpoint outputs/ppo/MuJoCoTouch-v1/model.zip
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env", required=True, help="Gymnasium env id, e.g. MuJoCoTouch-v1")
    parser.add_argument("--timesteps", type=int, default=50_000, help="Total training timesteps")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Where to write checkpoint + logs. Defaults to outputs/ppo/<env>/",
    )
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
    output_dir = Path(args.output_dir) if args.output_dir else Path("outputs/ppo") / args.env
    output_dir.mkdir(parents=True, exist_ok=True)

    register_env_backends(args.env)
    env = Monitor(gym.make(args.env))

    model = PPO(
        "MlpPolicy",
        env,
        seed=args.seed,
        tensorboard_log=str(output_dir / "tensorboard"),
        verbose=1,
    )
    model.learn(total_timesteps=args.timesteps)

    checkpoint_path = output_dir / "model.zip"
    model.save(str(checkpoint_path))
    env.close()

    print("\n=== training complete ===")
    print(f"env:        {args.env}")
    print(f"timesteps:  {args.timesteps}")
    print(f"checkpoint: {checkpoint_path}")
    print(f"logs:       {output_dir / 'tensorboard'}")


if __name__ == "__main__":
    main()
