"""Scripted expert for the quarter fold: one FoldExpert per (stage, arm) move,
staged. Both stages now run the identical synchronized-pair pattern: two
FoldExperts act every step, both hold their placed corners until the other has
placed too, then both release and retreat together.

A released corner springs back: the fold's bend pulls it a few centimetres
toward the fold line and the flap narrows across it. Placed exactly on the
goal, corners settle away from it and the fold fails, so each move places past
its goal by an OVERSHOOT (the mean spring-back measured on the stock cloth).
In stage 1 both moves converge on the SAME target corner (cloth_110) at the
same time -- a genuine two-arm carry of the two-layer west stack -- so their
OVERSHOOT values also carry a small lateral offset (+-2 cm in y) to keep the
two physical grippers from trying to occupy the same point; GRASP_RADIUS
(4 cm) and SUCCESS_DIST (5 cm) both tolerate that spread. Feasibility check:

    python -m cloth_fold_rl.quarter_fold_expert --episodes 5
"""

from __future__ import annotations

import argparse

import numpy as np

from cloth_fold_rl.expert import FoldExpert
from cloth_fold_rl.quarter_fold_env import STAGES, QuarterFoldEnv

# per (stage, arm): metres past the goal to place the corner at (the mean
# spring-back measured on the stock cloth). Stage 1 folds the two south corners
# inward toward centre; a released flap springs back outward, so each arm places
# a few cm further in (+x for the west arm, -x for the east arm). The two goals
# sit either side of centre and the two arms approach from opposite sides, so
# their grippers stay clear of each other.
OVERSHOOT = {(0, "left_"): np.array([-0.04, -0.03, 0.0]), (0, "right_"): np.array([0.02, -0.03, 0.0]),
             (1, "left_"): np.array([0.0, -0.02, 0.0]), (1, "right_"): np.array([0.0, -0.02, 0.0])}


def corner_index(env, vertex):
    return env.unwrapped._corner_ids.index(env.unwrapped._cloth_body_ids[vertex])


class QuarterFoldExpert:
    def __init__(self, env, seed=0):
        self.env = env
        self.base = env.unwrapped
        self.experts = {}
        self.correction = {}
        for s, stage in enumerate(STAGES):
            for k, move in enumerate(stage.moves):
                goal = lambda m=move, s=s: env.goal(m) + OVERSHOOT[(s, m.prefix)] + self.correction[(s, m.prefix)]
                self.experts[(s, move.prefix)] = FoldExpert(env, seed=seed + 10 * s + k, prefix=move.prefix,
                                                            corner=[corner_index(env, c) for c in move.corners],
                                                            goal=goal, release=True)
        self.retries = 0   # kept for run_episode's row dict; no retry logic anymore
        self.reset()

    def reset(self):
        for key, expert in self.experts.items():
            expert.reset()
            expert.release_allowed = False
            self.correction[key] = np.zeros(3)

    def _arms(self):
        return {p: e for (s, p), e in self.experts.items() if s == self.env.stage}

    def phases(self):
        return {p: e.PHASES[e.phase] for p, e in self._arms().items()}

    def act(self):
        arms = self._arms()
        if all(e.PHASES[e.phase] == "hold" for e in arms.values()):
            for e in arms.values():
                e.release_allowed = True
        return np.concatenate([arms["left_"].act(), arms["right_"].act()])


def run_episode(env, expert, seed, verbose=True):
    obs, info = env.reset(seed=seed)
    expert.reset()
    total = 0.0
    for t in range(env.unwrapped.max_episode_steps):
        obs, reward, terminated, truncated, info = env.step(expert.act())
        total += reward
        if verbose and t % 25 == 0:
            phases = expert.phases()
            print(f"  t{t:3d} stage {info['stage']} L {phases['left_']:<8} R {phases['right_']:<8} "
                  f"score {info['fold_score']:.3f} d {'/'.join(f'{d:.3f}' for d in info['move_distance'])} "
                  f"grasp {int(info['grasped']['left_'])}{int(info['grasped']['right_'])} settle {info['settle_steps']}")
        if terminated or truncated:
            break
    return {"seed": seed, "steps": t + 1, "reward": round(total, 2), "fold_score": round(info["fold_score"], 3),
            "stage": info["stage"], "retries": expert.retries, "success": info["success"],
            "reason": info["termination_reason"] or "truncated", "anchor_drift": round(info["anchor_drift"], 3)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=5)
    ap.add_argument("--seed-base", type=int, default=0)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()
    env = QuarterFoldEnv()
    env.unwrapped.domain_randomization = True
    expert = QuarterFoldExpert(env)
    rows = []
    for i in range(args.episodes):
        seed = args.seed_base + i
        if not args.quiet:
            print(f"episode seed={seed}")
        row = run_episode(env, expert, seed, verbose=not args.quiet)
        rows.append(row)
        print(f"  -> {row}")
    print(f"success {sum(r['success'] for r in rows)}/{len(rows)}, reached stage 1 in "
          f"{sum(r['stage'] >= 1 for r in rows)}, mean steps {np.mean([r['steps'] for r in rows]):.0f}, "
          f"mean score {np.mean([r['fold_score'] for r in rows]):.3f}")


if __name__ == "__main__":
    main()
