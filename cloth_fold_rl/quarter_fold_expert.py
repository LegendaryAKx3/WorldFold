"""Scripted expert for the quarter fold: one FoldExpert per move, staged.

Stage 0 runs a FoldExpert on each arm; both hold their placed corners until
the other has placed too, then both open and retreat. When the env reports
stage 1 the left arm parks above its start corner, out of the way, and once it
is clear the right arm runs a FoldExpert on the stacked corners.

A released corner springs back: the fold's bend pulls it a few centimetres
toward the fold line and the flap narrows across it. Placed exactly on the
goal, corners settle 5 to 7 cm away and the fold fails, so each move places
past its goal by an OVERSHOOT (the mean spring-back measured on the stock
cloth). In stage 1 the spring-back of the two-layer stack varies from under
1 cm to over 8 cm with the randomized cloth, so after releasing the expert
waits for the cloth to settle and, if the stack is off target, picks it up
again with the measured error added to its goal, up to MAX_RETRIES times.
Feasibility check:

    python -m cloth_fold_rl.quarter_fold_expert --episodes 5
"""

from __future__ import annotations

import argparse

import numpy as np

from cloth_fold_rl.expert import JOINT_DELTA_SCALE, FoldExpert
from cloth_fold_rl.quarter_fold_env import CLOTH_10, STAGES, QuarterFoldEnv

# per (stage, arm): metres past the goal to place the corner at
OVERSHOOT = {(0, "left_"): np.array([-0.04, -0.05, 0.0]), (0, "right_"): np.array([0.02, -0.05, 0.0]),
             (1, "right_"): np.array([0.03, 0.0, 0.0])}
PARK_HEIGHT = 0.10       # the left arm waits this far above its start corner in stage 1
CLEAR_DISTANCE = 0.12    # the right arm starts stage 1 once the left gripper is this far from the stack
SETTLE_WAIT = 15         # steps after retreat before judging the placement
MAX_RETRIES = 2


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
        self.left = self.experts[(0, "left_")]
        self.reset()

    def reset(self):
        for key, expert in self.experts.items():
            expert.reset()
            expert.release_allowed = False
            self.correction[key] = np.zeros(3)
        self.park_q = None
        self.done_steps = 0
        self.retries = 0

    def phases(self):
        if self.env.stage == 0:
            return {p: e.PHASES[e.phase] for (s, p), e in self.experts.items() if s == 0}
        right = self.experts[(1, "right_")]
        return {"left_": "park", "right_": right.PHASES[right.phase] if self._left_clear() else "wait"}

    def _left_clear(self):
        gripper = self.base.data.site_xpos[self.base._site_id["left_"]]
        stack = self.base.data.xpos[self.base._cloth_body_ids[STAGES[1].moves[0].corners[0]]]
        return float(np.linalg.norm(gripper - stack)) > CLEAR_DISTANCE

    def _park_action(self):
        """Left arm to a point above where its corner started, gripper open."""
        if self.park_q is None:
            target = self.env._start[CLOTH_10] + np.array([0.0, 0.0, PARK_HEIGHT])
            self.park_q, _ = self.left._ik(target)
        q_now = np.array([self.base.data.qpos[a] for a in self.left.qpos_adr])
        action = np.zeros(6, dtype=np.float32)
        action[0:5] = np.clip((self.park_q - q_now) / JOINT_DELTA_SCALE, -1.0, 1.0)
        action[5] = 1.0
        return action

    def act(self):
        if self.env.stage == 0:
            arms = {p: e for (s, p), e in self.experts.items() if s == 0}
            if all(e.PHASES[e.phase] == "hold" for e in arms.values()):
                for e in arms.values():
                    e.release_allowed = True
            return np.concatenate([arms["left_"].act(), arms["right_"].act()])
        right = self.experts[(1, "right_")]
        right.release_allowed = True
        if right.PHASES[right.phase] == "done":
            self.done_steps += 1
            if self.done_steps >= SETTLE_WAIT and not self.env._stage_settled() and self.retries < MAX_RETRIES:
                move = STAGES[1].moves[0]
                stack = np.mean([self.env._vertex(c) for c in move.corners], axis=0)
                self.correction[(1, "right_")] += self.env.goal(move) - stack
                right.reset()
                self.retries += 1
                self.done_steps = 0
        right_action = right.act() if self._left_clear() else np.array([0, 0, 0, 0, 0, 1.0], dtype=np.float32)
        return np.concatenate([self._park_action(), right_action])


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
