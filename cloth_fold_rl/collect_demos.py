"""Collect scripted-expert demonstrations for behavior cloning.

The expert is closed-loop (it re-solves IK from the corner's current position at
each phase), so unlike a fixed trajectory it stays valid under the per-episode
cloth jitter -- which is exactly the generalization PPO-from-scratch failed to
learn. Only SUCCESSFUL episodes are kept.

    python -m cloth_fold_rl.collect_demos --episodes 200 --workers 8
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
from pathlib import Path

import numpy as np


def rollout(job):
    """One expert episode. Returns (obs, actions, success). Runs in a worker."""
    seed, physical, max_steps = job
    from cloth_fold_rl.fold_env import make_fold_env, make_expert

    env = make_fold_env(physical, max_episode_steps=max_steps)
    expert = make_expert(env, physical, seed=seed)
    obs, _ = env.reset(seed=seed)
    expert.reset()

    obs_buf, act_buf = [], []
    info = {}
    for _ in range(env.unwrapped.max_episode_steps):
        action = expert.act()
        obs_buf.append(np.asarray(obs, dtype=np.float32))   # obs BEFORE the action
        act_buf.append(np.asarray(action, dtype=np.float32))
        obs, _, term, trunc, info = env.step(action)
        if term or trunc:
            break
    env.close()
    return (np.array(obs_buf, dtype=np.float32),
            np.array(act_buf, dtype=np.float32),
            bool(info.get("success", False)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=200)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default=None,
                    help="default outputs/cloth_fold_rl[/physical]/demos.npz")
    ap.add_argument("--physical", action="store_true", help="use the physical grabber (plates, no weld) -- see physical_env.py")
    ap.add_argument("--max-episode-steps", type=int, default=None,
                    help="default 200 (weld) / 250 (physical)")
    args = ap.parse_args()
    if args.out is None:
        args.out = ("outputs/cloth_fold_rl/physical/demos.npz" if args.physical
                    else "outputs/cloth_fold_rl/demos.npz")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    print(f"collecting {args.episodes} expert episodes on {args.workers} workers...")
    ctx = mp.get_context("spawn")
    with ctx.Pool(args.workers) as pool:
        results = []
        for i, r in enumerate(pool.imap_unordered(
                rollout, [(s, args.physical, args.max_episode_steps) for s in range(args.episodes)])):
            results.append(r)
            if (i + 1) % 20 == 0:
                ok = sum(x[2] for x in results)
                print(f"  {i+1}/{args.episodes} episodes, {ok} successful")

    good = [(o, a) for o, a, ok in results if ok]
    n_ok = len(good)
    if not good:
        raise SystemExit("no successful expert episodes -- nothing to clone")

    obs = np.concatenate([o for o, _ in good])
    acts = np.concatenate([a for _, a in good])
    np.savez_compressed(out, obs=obs, actions=acts)

    print(f"\nkept {n_ok}/{args.episodes} successful episodes "
          f"({n_ok/args.episodes:.0%} expert success rate)")
    print(f"{len(obs):,} transitions -> {out}")
    print(f"obs {obs.shape} actions {acts.shape}")
    print(f"action range [{acts.min():.2f}, {acts.max():.2f}]")


if __name__ == "__main__":
    main()
