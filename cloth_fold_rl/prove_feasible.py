"""Phase 2 gate: can a scripted expert actually fold the corner?

If this does not clear the success threshold, the task is not learnable and
there is no point starting PPO. Run:

    python -m cloth_fold_rl.prove_feasible --episodes 3
"""

from __future__ import annotations

import argparse
import numpy as np

from cloth_fold_rl.fold_env import SingleCornerFoldEnv, SUCCESS_DIST
from cloth_fold_rl.expert import FoldExpert


def run_episode(env, expert, seed, verbose=True):
    obs, info = env.reset(seed=seed)
    expert.reset()
    total_r = 0.0
    best_score = 0.0
    grasped_at = None

    for t in range(env.unwrapped.max_episode_steps):
        action = expert.act()
        obs, r, terminated, truncated, info = env.step(action)
        total_r += r
        best_score = max(best_score, info["fold_score"])
        if info["newly_grasped"] and grasped_at is None:
            grasped_at = t
        if verbose and t % 25 == 0:
            print(f"  t{t:3d} {expert.PHASES[expert.phase]:<9} "
                  f"score {info['fold_score']:.3f}  d_goal {info['corner_to_goal']:.3f}  "
                  f"grasp {int(info['grasped'])}")
        if terminated or truncated:
            break

    return {
        "steps": t + 1,
        "reward": total_r,
        "final_score": info["fold_score"],
        "best_score": best_score,
        "corner_to_goal": info["corner_to_goal"],
        "grasped_at": grasped_at,
        "success": info["success"],
        "reason": info["termination_reason"] or ("truncated" if truncated else None),
        "anchor_drift": info["anchor_drift"],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    env = SingleCornerFoldEnv()
    expert = FoldExpert(env)

    rows = []
    for ep in range(args.episodes):
        print(f"\n=== episode {ep} (seed {ep}) ===")
        rows.append(run_episode(env, expert, seed=ep, verbose=not args.quiet))
        r = rows[-1]
        print(f"  -> success={r['success']} score={r['final_score']:.3f} "
              f"d_goal={r['corner_to_goal']:.3f}m grasped_at={r['grasped_at']} "
              f"reason={r['reason']}")

    print("\n=== expert summary ===")
    print(f"{'ep':>3} {'success':>8} {'score':>7} {'d_goal':>7} {'grasp@':>7} {'reason':>14}")
    for i, r in enumerate(rows):
        print(f"{i:>3} {str(r['success']):>8} {r['final_score']:>7.3f} "
              f"{r['corner_to_goal']:>7.3f} {str(r['grasped_at']):>7} {str(r['reason']):>14}")
    n_ok = sum(r["success"] for r in rows)
    best = max(r["best_score"] for r in rows)
    grasp_rate = sum(r["grasped_at"] is not None for r in rows) / len(rows)
    print(f"\nsuccess {n_ok}/{len(rows)}   grasp rate {grasp_rate:.0%}   "
          f"best fold_score {best:.3f}   (success needs d_goal < {SUCCESS_DIST} m)")
    print("\nVERDICT:", "task is achievable -- PPO is worth running"
          if n_ok else "NOT achievable as configured -- fix before training")


if __name__ == "__main__":
    main()
