"""Train PPO on ClothFoldEnv while watching it live in a MuJoCo viewer.

PPO drives env.step() during rollout collection; we open a *passive* viewer on
the same model/data the env mutates in place, and sync() it every step. So the
window shows exactly the rollouts PPO is learning from, in real time.

macOS NOTE: the passive viewer must run under `mjpython`, not `python`:

    mjpython scripts/train_live.py --timesteps 200000

Every --save-every steps the checkpoint is written to
outputs/ppo/ClothFold-live/model.zip so you can reload it later.
"""

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "mujuco"))

import gymnasium as gym
import mujoco
import mujoco.viewer

from sim_main import ClothFoldEnv, StateOnlyWrapper


class LiveViewWrapper(gym.Wrapper):
    """Opens a passive viewer on the wrapped env's MuJoCo model/data and
    refreshes it once per env.step(). Purely a visualizer — it does not touch
    observations, rewards, or actions."""

    def __init__(self, env, sync_every=1):
        super().__init__(env)
        base = env.unwrapped
        self._viewer = mujoco.viewer.launch_passive(base.model, base.data)
        self._sync_every = sync_every
        self._i = 0

    def step(self, action):
        out = self.env.step(action)
        self._i += 1
        if self._viewer.is_running() and self._i % self._sync_every == 0:
            self._viewer.sync()
        return out

    def reset(self, **kwargs):
        out = self.env.reset(**kwargs)
        if self._viewer.is_running():
            self._viewer.sync()
        return out

    def close(self):
        self._viewer.close()
        return self.env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timesteps", type=int, default=200_000)
    parser.add_argument("--save-every", type=int, default=20_000)
    parser.add_argument("--sync-every", type=int, default=1,
                        help="Sync viewer every N env steps (raise to train faster / watch less)")
    parser.add_argument("--output-dir", default=str(REPO / "outputs/ppo/ClothFold-live"))
    args = parser.parse_args()

    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import CheckpointCallback

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # state obs -> flat Box -> plain MlpPolicy (no image encoder, CPU-friendly)
    env = LiveViewWrapper(
        StateOnlyWrapper(ClothFoldEnv(observation_mode="state")),
        sync_every=args.sync_every,
    )

    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=str(out / "tensorboard"))

    ckpt = CheckpointCallback(save_freq=args.save_every, save_path=str(out), name_prefix="model")
    try:
        model.learn(total_timesteps=args.timesteps, callback=ckpt)
    finally:
        model.save(str(out / "model.zip"))
        env.close()
        print(f"saved -> {out / 'model.zip'}")


if __name__ == "__main__":
    main()
