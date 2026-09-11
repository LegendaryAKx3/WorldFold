"""Iterative PPO training for the single-corner edge fold.

"Reiterating" is the point here: training runs in ROUNDS. Each round trains for
--steps-per-round, evaluates the current policy on held-out seeds, appends the
result to history.json, writes a checkpoint, and the next round resumes from it.
Kill the run at any time and re-launch with the same --run-dir to pick up where
it stopped -- nothing is lost but the partial round.

    python -m cloth_fold_rl.train --rounds 8 --steps-per-round 25000 --n-envs 8

Watch it:  tensorboard --logdir outputs/cloth_fold_rl/<run>/tb
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecMonitor

from cloth_fold_rl.fold_env import make_fold_env


def _make(max_episode_steps, physical=False):
    def _init():
        return make_fold_env(physical, max_episode_steps=max_episode_steps)
    return _init


def build_vec_env(n_envs, max_episode_steps, physical=False):
    fns = [_make(max_episode_steps, physical) for _ in range(n_envs)]
    # spawn: MuJoCo + macOS do not survive fork
    vec = SubprocVecEnv(fns, start_method="spawn") if n_envs > 1 else DummyVecEnv(fns)
    return VecMonitor(vec)


def evaluate(model, n_episodes=6, max_episode_steps=None, seed0=1000, physical=False):
    """Deterministic rollouts on seeds the policy never trained on."""
    env = make_fold_env(physical, max_episode_steps=max_episode_steps)
    max_episode_steps = env.unwrapped.max_episode_steps
    successes, scores, grasps, rewards = 0, [], 0, []
    for i in range(n_episodes):
        obs, info = env.reset(seed=seed0 + i)
        total, grasped = 0.0, False
        for _ in range(max_episode_steps):
            action, _ = model.predict(obs, deterministic=True)
            obs, r, term, trunc, info = env.step(action)
            total += r
            grasped = grasped or info["grasped"]
            if term or trunc:
                break
        successes += bool(info["success"])
        scores.append(info["fold_score"])
        grasps += bool(grasped)
        rewards.append(total)
    env.close()
    return {
        "success_rate": successes / n_episodes,
        "mean_fold_score": float(np.mean(scores)),
        "grasp_rate": grasps / n_episodes,
        "mean_reward": float(np.mean(rewards)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=8)
    ap.add_argument("--steps-per-round", type=int, default=25_000)
    ap.add_argument("--n-envs", type=int, default=8)
    ap.add_argument("--max-episode-steps", type=int, default=None,
                    help="default 200 (weld) / 250 (physical)")
    ap.add_argument("--eval-episodes", type=int, default=6)
    ap.add_argument("--run-dir", default="outputs/cloth_fold_rl/run1")
    ap.add_argument("--init-from", default=None,
                    help="Warm-start weights from another run's .zip (fresh "
                         "optimizer + history). Use to carry a learned skill "
                         "into a new task variant.")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--ent-coef", type=float, default=0.005)
    ap.add_argument("--physical", action="store_true", help="use the physical grabber (plates, no weld) -- see physical_env.py")
    args = ap.parse_args()

    run = Path(args.run_dir)
    run.mkdir(parents=True, exist_ok=True)
    latest = run / "latest.zip"
    best_path = run / "best.zip"
    hist_path = run / "history.json"

    history = json.loads(hist_path.read_text()) if hist_path.exists() else []
    best = max((h["eval"]["success_rate"] for h in history), default=-1.0)
    start_round = len(history)

    vec = build_vec_env(args.n_envs, args.max_episode_steps, args.physical)

    if latest.exists():
        print(f"resuming from {latest} (completed rounds: {start_round})")
        model = PPO.load(latest, env=vec, tensorboard_log=str(run / "tb"))
    elif args.init_from:
        print(f"warm-starting weights from {args.init_from}")
        model = PPO.load(args.init_from, env=vec, tensorboard_log=str(run / "tb"))
        model.num_timesteps = 0
    else:
        print("starting fresh")
        model = PPO(
            "MlpPolicy", vec, verbose=1, seed=0,
            learning_rate=args.lr, n_steps=512, batch_size=256, n_epochs=10,
            gamma=0.99, gae_lambda=0.95, clip_range=0.2, ent_coef=args.ent_coef,
            policy_kwargs=dict(net_arch=[256, 256]),
            tensorboard_log=str(run / "tb"),
        )

    for rnd in range(start_round, args.rounds):
        t0 = time.perf_counter()
        print(f"\n=== round {rnd + 1}/{args.rounds} "
              f"({args.steps_per_round:,} steps, {args.n_envs} envs) ===")
        model.learn(total_timesteps=args.steps_per_round,
                    reset_num_timesteps=False, tb_log_name="ppo")
        model.save(latest)

        ev = evaluate(model, args.eval_episodes, args.max_episode_steps,
                      physical=args.physical)
        dt = time.perf_counter() - t0
        print(f"round {rnd + 1}: success {ev['success_rate']:.0%}  "
              f"grasp {ev['grasp_rate']:.0%}  fold_score {ev['mean_fold_score']:.3f}  "
              f"reward {ev['mean_reward']:.1f}  ({dt/60:.1f} min)")

        history.append({"round": rnd + 1, "timesteps": int(model.num_timesteps),
                        "wall_minutes": dt / 60.0, "eval": ev})
        hist_path.write_text(json.dumps(history, indent=2))

        if ev["success_rate"] > best:
            best = ev["success_rate"]
            model.save(best_path)
            print(f"  new best ({best:.0%}) -> {best_path}")

    vec.close()
    print(f"\ndone. history: {hist_path}")
    print(f"best checkpoint: {best_path} (success {best:.0%})")


if __name__ == "__main__":
    main()
