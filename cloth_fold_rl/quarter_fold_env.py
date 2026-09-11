"""Quarter fold: two arms fold the cloth in half and release, then the same two
arms fold the halved cloth in half again and release. The fold has to hold on
its own.

A wrapper around ClothFoldEnv in the style of fold_env.SingleCornerFoldEnv,
with two stages. Corner positions at reset: cloth_0 (-h, -h), cloth_10
(-h, +h), cloth_110 (+h, -h), cloth_120 (+h, +h), h = 0.15 m. Both arms sit
side-by-side south of the cloth (see mujuco/sim_main.py's ARM_BASE_LEFT/RIGHT),
each reaching its own near south corner -- unlike the old diagonal placement,
where the two arms sat at opposite corners.

  stage 0  the left arm carries cloth_10 onto cloth_0 and the right arm carries
           cloth_120 onto cloth_110 (a fold about y = 0); both release and the
           corners must stay placed for SETTLE_STEPS. cloth_0 and cloth_110 must
           not be dragged. This leaves a half-height sheet along the south edge
           with a two-layer stack at each end: cloth_0+cloth_10 at the west
           corner, cloth_110+cloth_120 at the east corner.
  stage 1  a fold about x = 0: the whole WEST short edge flips onto the EAST
           short edge, both its corners carried at once by different arms
           (a real two-gripper edge fold, not each arm handling its own side).
           The right arm carries the two-layer south stack (cloth_0+cloth_10)
           onto cloth_110's start; the left arm carries the single-layer
           crease-end vertex (cloth_5) onto cloth_115's start. Both release and
           the two carried points must stay placed; the east edge's two
           corners (cloth_110, cloth_120) are the anchors that must not drag.

The base env welds one hard-wired vertex per gripper within GRASP_RADIUS. Here
each gripper's weld list holds its stage-0 corner and its stage-1 corner (see
GRASP_CORNERS). In stage 0 an arm's two weld vertices are 0.30 m apart so only
the near one welds; in stage 1 the arm grasps a stacked south corner, where
both its weld vertices sit within GRASP_RADIUS, so it lifts the whole two-layer
stack. GRASP_RADIUS is 6 cm (> SUCCESS_DIST 5 cm) so both layers still weld
when the stage-0 fold leaves them a few cm apart.

Reward: potential-based shaping summed over the stage's moves. Each move has
three regimes, free (reach the corner) / grasped (carry it) / released and
placed, joined so that grasping and letting go of a placed corner are steps up
and dropping an unplaced corner is a step down. STAGE_BONUS when stage 0
settles, SUCCESS_BONUS when stage 1 does. Drag and instability terminate.

Action: 12 values, left joint deltas(5) + left gripper, right joint deltas(5)
+ right gripper. Observation: the base env's flat state vector.
"""

from __future__ import annotations

from dataclasses import dataclass

import gymnasium as gym
import numpy as np

from cloth_fold_rl.fold_env import (
    CLOTH_JITTER, CTRL_COST, DRAG_LIMIT, GRASP_BONUS, SUCCESS_BONUS, SUCCESS_DIST, W_CARRY, W_REACH,
)
from sim_main import CLOTH_COUNT, ClothFoldEnv, StateOnlyWrapper  # noqa: E402 (path set by fold_env)

N = CLOTH_COUNT
CLOTH_0, CLOTH_10, CLOTH_110, CLOTH_120 = 0, N - 1, (N - 1) * N, N * N - 1
# Stage-1 goal vertices, the midpoints of the west/east SHORT edges after the
# stage-0 fold (row-major index = ix*N + iy). CLOTH_5 (-h, 0) is the crease-end
# of the west short edge; CLOTH_115 (+h, 0) is its mirror on the east edge.
CLOTH_5 = (N // 2)               # (-h, 0): west short edge, crease end
CLOTH_115 = (N - 1) * N + N // 2  # (+h, 0): east short edge, crease end
# Each arm welds its stage-0 corner and its stage-1 corner. In stage 0 the two
# weld vertices of an arm are 0.30 m apart (only the near one is in range), so
# one weld each. Stage 1 folds the whole WEST short edge onto the east short
# edge (both its corners: the two-layer south stack and the single-layer
# crease-end vertex) -- a real edge-to-edge fold, not each arm handling its own
# side. Reach is asymmetric though: both west-edge points sit at x=-h, so
# they're near the LEFT arm and a genuine cross-table reach for the RIGHT arm
# (measured via solve_ik: right_ -> west stack 0.5cm, right_ -> west midpoint
# 3.3cm, both well under GRASP_RADIUS). The right arm keeps doing the stack
# (its stage-0 role, and the easier of the two reaches); the left arm -- which
# is already home at the west edge -- carries the lone midpoint vertex across
# to CLOTH_115 (measured reach 4.3cm, also within tolerance).
GRASP_CORNERS = {"left_": (CLOTH_10, CLOTH_5), "right_": (CLOTH_120, CLOTH_0, CLOTH_10)}
GRASP_RADIUS = 0.06      # both layers of a stacked corner weld even if the
                         # stage-0 fold left them a few cm apart
RELEASE_BONUS = 3.0      # potential step for letting go of a placed corner
STAGE_BONUS = 10.0
SETTLE_STEPS = 20        # 1.0 s released and placed before a stage completes
MAX_STEPS = 400
ACTION_DIM = 12


@dataclass(frozen=True)
class Move:
    prefix: str        # the arm
    corners: tuple     # cloth vertices it carries (reached for at their mean position)
    goal: int          # cloth vertex whose start position they go to


@dataclass(frozen=True)
class Stage:
    moves: tuple
    anchors: tuple     # (vertex, reference vertex): vertex must stay within DRAG_LIMIT of the reference's start


STAGES = (
    Stage(moves=(Move("left_", (CLOTH_10,), CLOTH_0), Move("right_", (CLOTH_120,), CLOTH_110)),
          anchors=((CLOTH_0, CLOTH_0), (CLOTH_110, CLOTH_110))),
    Stage(moves=(Move("right_", (CLOTH_0, CLOTH_10), CLOTH_110), Move("left_", (CLOTH_5,), CLOTH_115)),
          anchors=((CLOTH_110, CLOTH_110), (CLOTH_120, CLOTH_120))),
)


class QuarterFoldEnv(gym.Wrapper):
    """Fold in half, release, fold in half again, release. See module docstring."""

    def __init__(self, max_episode_steps=MAX_STEPS, seed=None, cloth_jitter=CLOTH_JITTER):
        env = ClothFoldEnv(observation_mode="state", action_mode="joint_delta",
                           max_episode_steps=max_episode_steps, grasp_corners=GRASP_CORNERS,
                           grasp_radius=GRASP_RADIUS)
        super().__init__(env)
        self.cloth_jitter = cloth_jitter
        self._rng = np.random.default_rng(seed)
        self._flat = StateOnlyWrapper(env)
        self.observation_space = self._flat.observation_space
        self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(ACTION_DIM,), dtype=np.float32)
        self.stage = 0
        self._start = None          # cloth vertex positions at reset, [N*N, 3]
        self._stage_start = None    # cloth vertex positions at the current stage's start
        self._settle_steps = 0
        self._prev_potential = 0.0

    # ---- geometry helpers -------------------------------------------------

    def _vertex(self, index):
        return self.env.data.xpos[self.env._cloth_body_ids[index]]

    def goal(self, move):
        return self._start[move.goal]

    def _apply_weld_mask(self):
        # Each arm may weld only the corners it is meant to carry this stage.
        # CLOTH_10 lives in both arms' grasp_corners (left carries it in stage 0,
        # the right-arm stack includes it in stage 1); without this mask the left
        # arm welds CLOTH_10 along with CLOTH_5 at the stage-1 grasp and drags it
        # to the wrong goal, so the right arm's stack never places.
        mask = {p: set() for p in self.env.prefixes}
        for move in STAGES[self.stage].moves:
            mask[move.prefix].update(move.corners)
        self.env.weld_mask = mask

    def _grasped(self, prefix):
        return self.env.grasp_active(prefix)

    def _move_distance(self, move):
        """Mean distance of the move's corners to its goal."""
        return float(np.mean([np.linalg.norm(self._vertex(c) - self.goal(move)) for c in move.corners]))

    def _placed(self, move):
        return all(np.linalg.norm(self._vertex(c) - self.goal(move)) < SUCCESS_DIST for c in move.corners)

    def _start_distance(self, move):
        return max(float(np.mean([np.linalg.norm(self._start[c] - self.goal(move)) for c in move.corners])), 1e-6)

    def _anchor_drift(self):
        # reference is the vertex's position at the START of the current stage,
        # not at reset: an anchor corner may have been legitimately relocated by
        # an earlier stage (e.g. CLOTH_120 is folded onto CLOTH_110 in stage 0,
        # ~0.30 m from its reset pose). What must not move is the already-placed
        # cloth as THIS stage runs, so we measure drift from the stage's start.
        return max(float(np.linalg.norm(self._vertex(v) - self._stage_start[ref])) for v, ref in STAGES[self.stage].anchors)

    def fold_score(self):
        """Fraction of the two stages' total carry distance that has been covered."""
        done = sum(self._start_distance(m) for s in STAGES[:self.stage] for m in s.moves)
        current = sum(self._start_distance(m) - min(self._move_distance(m), self._start_distance(m))
                      for m in STAGES[self.stage].moves)
        total = sum(self._start_distance(m) for s in STAGES for m in s.moves)
        return float((done + current) / total)

    # ---- reward -----------------------------------------------------------

    def _move_potential(self, move):
        d = self._move_distance(move)
        if self._grasped(move.prefix):
            return GRASP_BONUS - W_CARRY * d
        if self._placed(move):
            return GRASP_BONUS + RELEASE_BONUS - W_CARRY * d
        gripper = self.env.data.site_xpos[self.env._site_id[move.prefix]]
        reach = float(np.linalg.norm(gripper - np.mean([self._vertex(c) for c in move.corners], axis=0)))
        return -W_REACH * reach - W_CARRY * self._start_distance(move)

    def _potential(self):
        return sum(self._move_potential(m) for m in STAGES[self.stage].moves)

    def _stage_settled(self):
        return all(not self._grasped(m.prefix) and self._placed(m) for m in STAGES[self.stage].moves)

    # ---- gym API ----------------------------------------------------------

    def reset(self, seed=None, options=None):
        opts = dict(options or {})
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        if self.cloth_jitter > 0 and "cloth_pose" not in opts:
            opts["cloth_pose"] = self._rng.uniform(-self.cloth_jitter, self.cloth_jitter, size=2)
        obs, info = self.env.reset(seed=seed, options=opts)
        self._start = self.env.data.xpos[self.env._cloth_body_ids].copy()
        self._stage_start = self._start.copy()
        self.stage = 0
        self._apply_weld_mask()
        self._settle_steps = 0
        self._prev_potential = self._potential()
        info = dict(info)
        info.update(self._info(None))
        return self._flat.observation(obs), info

    def _expand(self, action):
        a = np.zeros(14, dtype=np.float32)
        action = np.asarray(action, dtype=np.float32).ravel()
        a[0:5], a[6] = action[0:5], action[5]       # left joint deltas, left gripper
        a[7:12], a[13] = action[6:11], action[11]   # right joint deltas, right gripper
        return a

    def _info(self, reason):
        return {"stage": self.stage, "fold_score": self.fold_score(), "settle_steps": self._settle_steps,
                "grasped": {p: self._grasped(p) for p in self.env.prefixes},
                "move_distance": [self._move_distance(m) for m in STAGES[self.stage].moves],
                "anchor_drift": self._anchor_drift(), "success": reason == "success",
                "termination_reason": reason}

    def step(self, action):
        obs, _, _, _, info = self.env.step(self._expand(action))

        potential = self._potential()
        reward = potential - self._prev_potential - CTRL_COST * float(np.square(action).sum())
        self._settle_steps = self._settle_steps + 1 if self._stage_settled() else 0

        terminated, reason = False, None
        if self._settle_steps >= SETTLE_STEPS:
            if self.stage == len(STAGES) - 1:
                reward += SUCCESS_BONUS
                terminated, reason = True, "success"
            else:
                reward += STAGE_BONUS
                self.stage += 1
                self._stage_start = self.env.data.xpos[self.env._cloth_body_ids].copy()
                self._apply_weld_mask()
                self._settle_steps = 0
                potential = self._potential()
        elif self._anchor_drift() > DRAG_LIMIT:
            reward -= 5.0
            terminated, reason = True, "cloth_dragged"
        elif self.env._failed():
            reward -= 5.0
            terminated, reason = True, "unstable"
        self._prev_potential = potential
        truncated = (not terminated) and self.env._step_count >= self.env.max_episode_steps

        info = dict(info)
        info.update(self._info(reason))
        return self._flat.observation(obs), float(reward), terminated, truncated, info
