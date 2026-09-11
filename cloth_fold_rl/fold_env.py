"""Single-corner edge-fold task, built as a WRAPPER around ClothFoldEnv.

Why a wrapper and not an edit to sim_main.py: ClothFoldEnv's default goal sends
every corner to its diagonal-opposite start position, which is a 180-degree
in-plane rotation of a flat sheet, not a fold -- folding the cloth in half makes
that score go DOWN. cloth_angles/ collects data against the stock env, so the
fold objective lives out here instead of changing semantics underneath it.

What this wrapper changes:

  goal    left arm's welded corner (cloth_10) travels to cloth_120's start
          position: a 0.30 m edge fold about the x=0 line. cloth_110 is 0.083 m
          outside the arm's kinematic workspace, so the diagonal fold is
          impossible -- this is the fold that is actually reachable.
  action  left arm only (5 joint deltas + gripper). joint_delta mode bypasses
          the env's differential IK, which stalls 0.10 m short of poses that are
          kinematically valid. Halving the action space also halves exploration.
  reward  dense potential-based shaping (reach -> grasp -> carry -> place),
          because the stock reward is -mean(corner_dist) with a sparse +10 and
          nothing that rewards ever touching the cloth.
  success measured on the corner that actually moves, not averaged over four
          corners where three are supposed to stay put.
"""

from __future__ import annotations

import sys
from pathlib import Path

import gymnasium as gym
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "mujuco"))

from sim_main import ClothFoldEnv, StateOnlyWrapper, TABLE_TOP_Z  # noqa: E402

# corner ordering inside ClothFoldEnv._corner_ids
IDX_CLOTH_0, IDX_CLOTH_10, IDX_CLOTH_110, IDX_CLOTH_120 = 0, 1, 2, 3

MOVING_CORNER = IDX_CLOTH_10     # the corner welded to the left gripper
GOAL_CORNER = IDX_CLOTH_120      # its mirror across x=0 -- the fold target

SUCCESS_DIST = 0.05      # corner within 5 cm of the fold target
HOLD_STEPS = 10          # ...sustained this many control steps (0.5 s)
LIFT_TARGET_Z = TABLE_TOP_Z + 0.12

W_REACH = 4.0            # shaping weight, gripper -> corner (pre-grasp)
W_CARRY = 8.0            # shaping weight, corner -> goal (post-grasp)
GRASP_BONUS = 3.0
SUCCESS_BONUS = 20.0
CTRL_COST = 0.002
DRAG_LIMIT = 0.20        # anchor corners may not be dragged further than this
CLOTH_JITTER = 0.025     # +-2.5 cm random cloth offset per episode (see __init__)


class SingleCornerFoldEnv(gym.Wrapper):
    """One arm folds one corner across the cloth. See module docstring."""

    # the weld env can only hold the corner at the goal while welded, so success
    # requires an active grasp. physical_env.py relaxes this: a corner set down
    # on its target is a fold whether or not the jaw is still pinching it.
    SUCCESS_NEEDS_GRASP = True

    def __init__(self, max_episode_steps=200, single_arm=True, seed=None,
                 cloth_jitter=CLOTH_JITTER, base_env=None):
        env = base_env if base_env is not None else ClothFoldEnv(
            observation_mode="state",
            action_mode="joint_delta",
            max_episode_steps=max_episode_steps,
        )
        super().__init__(env)
        self.single_arm = single_arm
        # ClothFoldEnv.reset only consumes np_random when domain_randomization is
        # on, so seeding it otherwise is a no-op: every seed yields a byte-identical
        # start. Without this jitter all parallel envs run the SAME trajectory and
        # the policy overfits one episode. Drawn here and passed in as cloth_pose.
        self.cloth_jitter = cloth_jitter
        self._rng = np.random.default_rng(seed)
        self._flat = StateOnlyWrapper(env)
        self.observation_space = self._flat.observation_space
        # left arm: 5 joint deltas + 1 gripper
        n_act = 6 if single_arm else 14
        self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(n_act,), dtype=np.float32)

        self._goal = None
        self._anchors0 = None
        self._grasped = False
        self._success_steps = 0
        self._prev_potential = 0.0

    # ---- geometry helpers -------------------------------------------------

    def _corners(self):
        return self.env.data.xpos[self.env._corner_ids].copy()

    def _gripper_pos(self):
        return self.env.data.site_xpos[self.env._site_id["left_"]].copy()

    def _moving_corner(self):
        return self.env.data.xpos[self.env._corner_ids[MOVING_CORNER]].copy()

    def _corner_to_goal(self):
        return float(np.linalg.norm(self._moving_corner() - self._goal))

    def _grasp_active(self):
        return bool(self.env.data.eq_active[self.env._weld_id["left_"]])

    def _anchor_drift(self):
        """How far the corners that are supposed to stay put have been dragged."""
        now = self._corners()
        keep = [IDX_CLOTH_0, IDX_CLOTH_110]
        return float(np.linalg.norm(now[keep] - self._anchors0[keep], axis=1).max())

    def fold_score(self):
        """1.0 = corner is on its target, 0.0 = corner has not moved at all."""
        return float(np.clip(1.0 - self._corner_to_goal() / self._start_dist, 0.0, 1.0))

    def _goal_for(self, corners0):
        """Where the moving corner has to go: the mirror corner's start (edge fold)."""
        return corners0[GOAL_CORNER].copy()

    # ---- reward -----------------------------------------------------------

    def _potential(self):
        """Higher is better. Two regimes, joined so grasping is never a step down."""
        if self._grasp_active():
            return GRASP_BONUS - W_CARRY * self._corner_to_goal()
        reach = float(np.linalg.norm(self._gripper_pos() - self._moving_corner()))
        return -W_REACH * reach - W_CARRY * self._start_dist

    # ---- gym API ----------------------------------------------------------

    def reset(self, seed=None, options=None):
        opts = dict(options or {})
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        if self.cloth_jitter > 0 and "cloth_pose" not in opts:
            opts["cloth_pose"] = self._rng.uniform(-self.cloth_jitter,
                                                   self.cloth_jitter, size=2)
        obs, info = self.env.reset(seed=seed, options=opts)

        corners0 = self._corners()
        self._anchors0 = corners0
        self._goal = self._goal_for(corners0)
        self._start_dist = max(float(np.linalg.norm(corners0[MOVING_CORNER] - self._goal)), 1e-6)

        self._grasped = False
        self._success_steps = 0
        self._prev_potential = self._potential()

        info = dict(info)
        info.update({"fold_score": self.fold_score(), "grasped": False,
                     "corner_to_goal": self._corner_to_goal()})
        return self._flat.observation(obs), info

    def _expand(self, action):
        a = np.zeros(14, dtype=np.float32)
        action = np.asarray(action, dtype=np.float32).ravel()
        if self.single_arm:
            a[0:5] = action[0:5]     # left joint deltas
            a[6] = action[5]         # left gripper
        else:
            a[:] = action
        return a

    def step(self, action):
        obs, _, _, _, info = self.env.step(self._expand(action))

        grasp = self._grasp_active()
        newly_grasped = grasp and not self._grasped
        self._grasped = grasp

        potential = self._potential()
        reward = potential - self._prev_potential      # potential-based shaping
        self._prev_potential = potential
        reward -= CTRL_COST * float(np.square(action).sum())

        d = self._corner_to_goal()
        placed = d < SUCCESS_DIST and (grasp or not self.SUCCESS_NEEDS_GRASP)
        self._success_steps = self._success_steps + 1 if placed else 0

        terminated = False
        reason = None
        if self._success_steps >= HOLD_STEPS:
            reward += SUCCESS_BONUS
            terminated, reason = True, "success"
        elif self._anchor_drift() > DRAG_LIMIT:
            reward -= 5.0
            terminated, reason = True, "cloth_dragged"
        elif self.env._failed():
            reward -= 5.0
            terminated, reason = True, "unstable"

        truncated = (not terminated) and self.env._step_count >= self.env.max_episode_steps

        info = dict(info)
        info.update({
            "fold_score": self.fold_score(),
            "corner_to_goal": d,
            "grasped": grasp,
            "newly_grasped": newly_grasped,
            "anchor_drift": self._anchor_drift(),
            "success": reason == "success",
            "termination_reason": reason,
        })
        return self._flat.observation(obs), float(reward), terminated, truncated, info


def make_fold_env(physical=False, **kw):
    """The task env: weld-cheat grasp (default) or the physical grabber."""
    if kw.get("max_episode_steps") is None:
        kw.pop("max_episode_steps", None)        # let each env apply its own default
    if physical:
        from cloth_fold_rl.physical_env import SingleCornerPhysicalFoldEnv
        return SingleCornerPhysicalFoldEnv(**kw)
    return SingleCornerFoldEnv(**kw)


def make_expert(env, physical=False, seed=0):
    """The scripted expert matching make_fold_env(physical)."""
    if physical:
        from cloth_fold_rl.physical_expert import ScoopExpert
        return ScoopExpert(env, seed=seed)
    from cloth_fold_rl.expert import FoldExpert
    return FoldExpert(env, seed=seed)


def make_env(max_episode_steps=200, seed=None, physical=False):
    """Factory for SB3 vec envs."""
    def _init():
        env = make_fold_env(physical, max_episode_steps=max_episode_steps)
        if seed is not None:
            env.reset(seed=seed)
        return env
    return _init
