"""Scripted expert for the single-corner edge fold.

Purpose: feasibility proof. If a hand-written controller with a proper IK solver
cannot reach the success threshold, no amount of PPO will -- so this runs first.

It also doubles as the demo policy for the viewer and as a source of
demonstrations if we later want to warm-start / imitate.

The env's own ik_substep() is a damped-free differential solver that stalls
~0.10 m short of poses that are kinematically valid, so this module carries its
own damped-least-squares IK with random restarts, solved offline on a scratch
MjData and executed through joint_delta actions.
"""

from __future__ import annotations

import numpy as np
import mujoco

from cloth_fold_rl.fold_env import (
    MOVING_CORNER, LIFT_TARGET_Z, SUCCESS_DIST,
)

JOINT_DELTA_SCALE = 0.05     # matches sim_main.JOINT_DELTA_SCALE
GRASP_RADIUS = 0.03          # matches sim_main.GRASP_RADIUS


def solve_ik(model, data, site_id, qpos_adr, dof_adr, joint_range, target,
             iters=400, damping=0.08, restarts=12, tol=0.006, rng=None):
    """Damped least squares IK, position only (orientation free on a 5-DOF arm).

    Solves on a scratch MjData seeded from `data` so the live sim is untouched.
    Returns (q, err) for the best configuration found.
    """
    rng = rng or np.random.default_rng(0)
    scratch = mujoco.MjData(model)
    lo = np.array([r[0] for r in joint_range])
    hi = np.array([r[1] for r in joint_range])
    target = np.asarray(target, dtype=float)

    best_q, best_err = None, np.inf
    for r in range(restarts):
        scratch.qpos[:] = data.qpos
        scratch.qvel[:] = 0.0
        if r > 0:                                   # restart from a random pose
            for k, adr in enumerate(qpos_adr):
                scratch.qpos[adr] = rng.uniform(lo[k], hi[k])

        for _ in range(iters):
            mujoco.mj_kinematics(model, scratch)
            mujoco.mj_comPos(model, scratch)
            err = target - scratch.site_xpos[site_id]
            e = float(np.linalg.norm(err))
            if e < tol:
                break
            jacp = np.zeros((3, model.nv))
            mujoco.mj_jacSite(model, scratch, jacp, None, site_id)
            J = jacp[:, dof_adr]
            # damped least squares: dq = J^T (J J^T + lambda^2 I)^-1 e
            JJt = J @ J.T + (damping ** 2) * np.eye(3)
            dq = J.T @ np.linalg.solve(JJt, err)
            step = float(np.max(np.abs(dq)))
            if step > 0.1:
                dq *= 0.1 / step
            for k, adr in enumerate(qpos_adr):
                scratch.qpos[adr] = np.clip(scratch.qpos[adr] + dq[k], lo[k], hi[k])

        mujoco.mj_kinematics(model, scratch)
        e = float(np.linalg.norm(target - scratch.site_xpos[site_id]))
        if e < best_err:
            best_err = e
            best_q = np.array([scratch.qpos[a] for a in qpos_adr])
            if e < tol:
                break
    return best_q, best_err


class FoldExpert:
    """Phase machine: approach -> descend -> grasp -> lift -> carry -> place -> hold.

    prefix/corner/goal pick the arm, the corner index (or indices, reached for
    at their mean) in the env's corner list and a callable returning the goal
    (default: env._goal). With
    release=True the machine continues hold -> release -> retreat -> done once
    release_allowed is set (True by default; a coordinator can gate it).
    """

    PHASES = ("approach", "descend", "lift", "carry", "place", "hold")
    RELEASE_PHASES = ("release", "retreat", "done")
    OPEN_PHASES = ("approach", "release", "retreat", "done")
    RETREAT_HEIGHT = 0.05

    def __init__(self, env, seed=0, prefix="left_", corner=MOVING_CORNER, goal=None, release=False):
        self.env = env
        self.base = env.unwrapped
        self.rng = np.random.default_rng(seed)
        self.prefix = prefix
        self.corners = tuple(np.atleast_1d(corner))
        self.goal = goal if goal is not None else (lambda: self.env._goal)
        if release:
            self.PHASES = self.PHASES + self.RELEASE_PHASES
        self.release_allowed = True
        self.site_id = self.base._site_id[self.prefix]
        self.qpos_adr = self.base._arm_qpos_adr[self.prefix]
        self.dof_adr = self.base._arm_dof_adr[self.prefix]
        self.joint_range = [self.base.model.joint(f"{self.prefix}{n}").range
                            for n in ["shoulder_pan", "shoulder_lift", "elbow_flex",
                                      "wrist_flex", "wrist_roll"]]
        self.reset()

    def reset(self):
        self.phase = 0
        self.q_target = None
        self.phase_steps = 0
        self.retreat_target = None

    def _ik(self, target):
        q, err = solve_ik(self.base.model, self.base.data, self.site_id,
                          self.qpos_adr, self.dof_adr, self.joint_range,
                          target, rng=self.rng)
        return q, err

    def _corner(self):
        return np.mean([self.base.data.xpos[self.base._corner_ids[c]] for c in self.corners], axis=0)

    def _plan(self):
        corner = self._corner()
        goal = np.asarray(self.goal(), dtype=float).copy()
        name = self.PHASES[self.phase]
        if name == "approach":
            return corner + np.array([0.0, 0.0, 0.06])
        if name == "descend":
            return corner + np.array([0.0, 0.0, 0.005])
        if name == "lift":
            return np.array([corner[0], corner[1], LIFT_TARGET_Z])
        if name == "carry":
            return np.array([goal[0], goal[1], LIFT_TARGET_Z])
        if name == "place":
            return goal + np.array([0.0, 0.0, 0.02])
        if name == "retreat":
            if self.retreat_target is None:
                self.retreat_target = self.base.data.site_xpos[self.site_id] + np.array([0.0, 0.0, self.RETREAT_HEIGHT])
            return self.retreat_target
        return None      # hold, release, done: stop moving

    def act(self):
        action = np.zeros(6, dtype=np.float32)
        name = self.PHASES[self.phase]

        # gripper: open while approaching and after release, closed in between
        action[5] = 1.0 if name in self.OPEN_PHASES else -1.0

        target = self._plan()
        if target is not None:
            if self.q_target is None:
                self.q_target, _ = self._ik(target)
            q_now = np.array([self.base.data.qpos[a] for a in self.qpos_adr])
            dq = self.q_target - q_now
            action[0:5] = np.clip(dq / JOINT_DELTA_SCALE, -1.0, 1.0)

        self.phase_steps += 1
        self._maybe_advance(target)
        return action

    def _maybe_advance(self, target):
        name = self.PHASES[self.phase]
        if name == "done" or (name == "hold" and not (self.release_allowed and "release" in self.PHASES)):
            return
        site = self.base.data.site_xpos[self.site_id]
        corner = self._corner()
        reached = target is not None and float(np.linalg.norm(site - target)) < 0.02

        advance = False
        if name == "descend":
            # only move on once the weld has actually engaged
            advance = self.base.grasp_active(self.prefix)
        elif name == "place":
            advance = float(np.linalg.norm(corner - np.asarray(self.goal()))) < SUCCESS_DIST
        elif name == "hold":
            advance = True
        elif name == "release":
            # wait for the weld to let go before moving the arm away
            advance = not self.base.grasp_active(self.prefix) and self.phase_steps >= 2
        else:
            advance = reached

        # per-phase patience, so a stuck phase does not eat the whole episode
        if advance or self.phase_steps > 45:
            self.phase = min(self.phase + 1, len(self.PHASES) - 1)
            self.phase_steps = 0
            self.q_target = None
            self.retreat_target = None
