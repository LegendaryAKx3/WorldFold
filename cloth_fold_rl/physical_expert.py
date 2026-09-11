"""Scripted scoop-pinch-fold expert for the physical grabber (no weld).

The weld expert (expert.py) descends vertically onto the corner and waits for
the weld to engage. A real jaw cannot pick flat cloth off a table that way, so
this expert ports the routine that mujuco/prove_grabber.py showed to work:

  hover -> down -> slide -> close -> lift -> carry -> place -> hold

  * hover/down: park the gripper on the cloth's free side (+y of the corner)
    with the scoop ramp's leading edge skimming the table,
  * slide: creep in -y until the corner rides up the ramp (closed-loop on the
    corner's height, so it tolerates the per-episode cloth jitter),
  * close: pinch, wait for the physical grasp signal,
  * lift/carry/place: a rate-limited straight line to the fold goal.

Actions are joint deltas from a damped-least-squares IK on a scratch MjData,
like expert.py, with one difference that matters: the ramp is bolted to the
gripper, so the wrist has to hold the posture the plates were calibrated in
(shoulder_lift on its limit, arm stretched low -- Q_SCOOP below, measured from
the proof) or the ramp digs into the table / points into the air. The approach
phases therefore solve position-only IK with a nullspace pull toward Q_SCOOP;
the carry phases seed from the current joints and let orientation follow.

All approach offsets are the poses the proof actually REACHED (its own IK
stalled 2-3 cm short of its targets, and the plates were calibrated there).
"""

from __future__ import annotations

import numpy as np
import mujoco

from cloth_fold_rl.fold_env import MOVING_CORNER, SUCCESS_DIST
from cloth_fold_rl.expert import JOINT_DELTA_SCALE

# joint posture at the proof's scoop (pan, lift, elbow, wrist_flex, wrist_roll)
Q_SCOOP = np.array([-0.33, -1.745, 1.456, 1.28, 0.0])

DOWN_OFFSET  = np.array([0.006, 0.050])   # site xy relative to the corner before sliding
SLIDE_Z      = 0.0327                     # site height above the table while the ramp skims it
HOVER_Z      = 0.10
SLIDE_HOP    = 0.005                      # the proof's slide profile: hop the target 5 mm, then hold...
SLIDE_PERIOD = 4                          # ...for this many control steps (a smooth creep shoves the cloth)
SLIDE_Z_GAIN = 0.5                        # integral correction on the measured site height
ON_RAMP_LIFT = 0.022                      # corner this high above the table = riding the ramp
LIFT_Z       = 0.05                       # site height for the carry (higher costs reach)
PLACE_Z      = 0.03
CARRY_SPEED  = 0.005                      # m per control step for the carry target...
TRACK_TOL    = 0.02                       # ...which only advances while the site is this close
CARRY_ROT_W  = 0.3                        # hold the grasp-time jaw orientation while carrying (a 5-DOF
                                          # arm cannot reach the far goal with it held hard; see README)


def solve_ik_posture(model, data, site_id, qpos_adr, dof_adr, joint_range, target,
                     seed_q, q_ref=None, iters=200, damping=0.05, posture_gain=0.3,
                     tol=0.003):
    """Position-only DLS IK seeded from `seed_q`, with an optional nullspace pull
    toward `q_ref`. Deterministic: no random restarts, so the solution stays in
    the configuration branch of the seed."""
    scratch = mujoco.MjData(model)
    scratch.qpos[:] = data.qpos
    scratch.qvel[:] = 0.0
    lo = np.array([r[0] for r in joint_range]); hi = np.array([r[1] for r in joint_range])
    n = len(qpos_adr)
    q = np.clip(np.asarray(seed_q, dtype=float), lo, hi)
    target = np.asarray(target, dtype=float)
    err_norm = np.inf
    for _ in range(iters):
        for k, adr in enumerate(qpos_adr):
            scratch.qpos[adr] = q[k]
        mujoco.mj_kinematics(model, scratch)
        mujoco.mj_comPos(model, scratch)
        err = target - scratch.site_xpos[site_id]
        err_norm = float(np.linalg.norm(err))
        if err_norm < tol:
            break
        jacp = np.zeros((3, model.nv))
        mujoco.mj_jacSite(model, scratch, jacp, None, site_id)
        J = jacp[:, dof_adr]
        JJt = J @ J.T + (damping ** 2) * np.eye(3)
        J_pinv = J.T @ np.linalg.solve(JJt, np.eye(3))
        dq = J_pinv @ err
        if q_ref is not None:
            dq += posture_gain * (np.eye(n) - J_pinv @ J) @ (q_ref - q)
        step = float(np.max(np.abs(dq)))
        if step > 0.1:
            dq *= 0.1 / step
        q = np.clip(q + dq, lo, hi)
    return q, err_norm


class ScoopExpert:
    """Phase machine for the physical grabber. Same act()/reset() API as FoldExpert."""

    PHASES = ("hover", "down", "slide", "close", "lift", "carry", "place", "hold")
    PATIENCE = {"hover": 45, "down": 25, "slide": 60, "close": 25,
                "lift": 12, "carry": 70, "place": 25, "hold": 10 ** 9}

    def __init__(self, env, seed=0):
        self.env = env
        self.base = env.unwrapped
        self.rng = np.random.default_rng(seed)
        self.prefix = "left_"
        self.site_id = self.base._site_id[self.prefix]
        self.qpos_adr = self.base._arm_qpos_adr[self.prefix]
        self.dof_adr = self.base._arm_dof_adr[self.prefix]
        self.joint_range = [self.base.model.joint(f"{self.prefix}{n}").range
                            for n in ["shoulder_pan", "shoulder_lift", "elbow_flex",
                                      "wrist_flex", "wrist_roll"]]
        from sim_main import TABLE_TOP_Z
        self.table_z = TABLE_TOP_Z
        self.reset()

    def reset(self):
        self.phase = 0
        self.phase_steps = 0
        self.q_target = None
        self.slide_xy = None
        self.slide_z_corr = 0.0
        self.carry_target = None
        self.carry_quat = None
        self.grasp_steps = 0
        self.ik_err = 0.0

    # ---- helpers ------------------------------------------------------------

    def _site(self):
        return self.base.data.site_xpos[self.site_id].copy()

    def _q(self):
        return np.array([self.base.data.qpos[a] for a in self.qpos_adr])

    def _corner(self):
        return self.base.data.xpos[self.base._corner_ids[MOVING_CORNER]].copy()

    def _lift(self):
        return float(self._corner()[2] - self.table_z)

    def _grasped(self):
        return bool(self.env._grasp_active())

    def _ik(self, target, posture):
        if posture:
            seed, ref = Q_SCOOP, Q_SCOOP
        else:
            seed, ref = self._q(), None
        q, self.ik_err = solve_ik_posture(self.base.model, self.base.data, self.site_id,
                                          self.qpos_adr, self.dof_adr, self.joint_range,
                                          target, seed_q=seed, q_ref=ref, iters=400, tol=0.001)
        return q

    def _toward(self, goal_xyz):
        """Rate-limited carry target: creep CARRY_SPEED per step along a straight
        line, and only while the arm is actually tracking it (the proof's carry
        did the same) -- otherwise the target runs away and the IK yanks."""
        if self.carry_target is None:
            self.carry_target = self._site()
        if float(np.linalg.norm(self._site() - self.carry_target)) < TRACK_TOL:
            d = goal_xyz - self.carry_target
            n = float(np.linalg.norm(d))
            if n > CARRY_SPEED:
                d *= CARRY_SPEED / n
            self.carry_target = self.carry_target + d
        return self.carry_target.copy()

    def _track(self, target):
        """One joint-rate-limited 6-DoF step toward `target` while holding the jaw
        orientation captured at the grasp (a wish on a 5-DOF arm, weighted by
        CARRY_ROT_W). Seeded from the live joints so the arm stays in its branch.

        Note mju_subQuat gives the rotation error in the site's LOCAL frame while
        mj_jacSite's rotational Jacobian is in the WORLD frame; the error has to
        be rotated into world before the two are stacked."""
        m, d = self.base.model, self.base.data
        lo = np.array([r[0] for r in self.joint_range]); hi = np.array([r[1] for r in self.joint_range])
        scratch = mujoco.MjData(m)
        scratch.qpos[:] = d.qpos
        q_now = self._q(); q = q_now.copy()
        for _ in range(40):
            for k, adr in enumerate(self.qpos_adr):
                scratch.qpos[adr] = q[k]
            mujoco.mj_kinematics(m, scratch); mujoco.mj_comPos(m, scratch)
            err_pos = target - scratch.site_xpos[self.site_id]
            R = scratch.site_xmat[self.site_id].reshape(3, 3)
            sq = np.zeros(4); mujoco.mju_mat2Quat(sq, scratch.site_xmat[self.site_id])
            err_loc = np.zeros(3); mujoco.mju_subQuat(err_loc, self.carry_quat, sq)
            err_rot = R @ err_loc
            err_rot[2] = 0.0          # hold tilt only: yaw is free so the pan can steer
            if np.linalg.norm(err_pos) < 0.002 and np.linalg.norm(err_rot) < 0.02:
                break
            jacp = np.zeros((3, m.nv)); jacr = np.zeros((3, m.nv))
            mujoco.mj_jacSite(m, scratch, jacp, jacr, self.site_id)
            J = np.vstack([jacp[:, self.dof_adr], CARRY_ROT_W * jacr[:, self.dof_adr]])
            e = np.concatenate([err_pos, CARRY_ROT_W * err_rot])
            dq = J.T @ np.linalg.solve(J @ J.T + 0.05 ** 2 * np.eye(6), e)
            step = float(np.max(np.abs(dq)))
            if step > 0.05:
                dq *= 0.05 / step
            q = np.clip(q + dq, lo, hi)
        return q_now + np.clip(q - q_now, -JOINT_DELTA_SCALE, JOINT_DELTA_SCALE)

    # ---- plan ---------------------------------------------------------------

    def _plan(self):
        """(site target, posture-biased?) for the current phase; None = hold still."""
        name = self.PHASES[self.phase]
        c = self._corner()
        goal = self.env._goal
        tz = self.table_z
        if name == "hover":
            return np.array([c[0] + DOWN_OFFSET[0], c[1] + DOWN_OFFSET[1], tz + HOVER_Z]), True
        if name == "down":
            return np.array([c[0] + DOWN_OFFSET[0], c[1] + DOWN_OFFSET[1], tz + SLIDE_Z]), True
        if name == "slide":
            if self.slide_xy is None:
                self.slide_xy = np.array([c[0] + DOWN_OFFSET[0], self._site()[1]])
            if self.phase_steps % SLIDE_PERIOD == 0:
                self.slide_xy = self.slide_xy + np.array([0.0, -SLIDE_HOP])
            # the arm tracks a few mm high; integrate the measured error out
            self.slide_z_corr += SLIDE_Z_GAIN * (tz + SLIDE_Z - self._site()[2])
            self.slide_z_corr = float(np.clip(self.slide_z_corr, -0.01, 0.01))
            return np.array([self.slide_xy[0], self.slide_xy[1], tz + SLIDE_Z + self.slide_z_corr]), True
        if name == "close":
            return None, True
        s = self._site()
        off = s[:2] - c[:2]        # keep the jaw-to-corner offset so the CORNER lands on the goal
        if name == "lift":
            return self._toward(np.array([s[0], s[1], tz + LIFT_Z])), False
        if name == "carry":
            return self._toward(np.array([goal[0] + off[0], goal[1] + off[1], tz + LIFT_Z])), False
        if name == "place":
            return self._toward(np.array([goal[0] + off[0], goal[1] + off[1], tz + PLACE_Z])), False
        return None, False

    def act(self):
        name = self.PHASES[self.phase]
        action = np.zeros(6, dtype=np.float32)
        action[5] = 1.0 if name in ("hover", "down", "slide") else -1.0

        target, posture = self._plan()
        if target is not None:
            if name in ("lift", "carry", "place"):
                self.q_target = self._track(target)
            elif self.q_target is None or name == "slide":
                self.q_target = self._ik(target, posture)
            action[0:5] = np.clip((self.q_target - self._q()) / JOINT_DELTA_SCALE, -1.0, 1.0)

        self.phase_steps += 1
        self._maybe_advance(name, target)
        return action

    def _maybe_advance(self, name, target):
        if name == "hold":
            return
        reached = target is not None and float(np.linalg.norm(self._site() - target)) < 0.012
        if name == "slide":
            advance = self._lift() > ON_RAMP_LIFT
        elif name == "close":
            self.grasp_steps = self.grasp_steps + 1 if self._grasped() else 0
            advance = self.grasp_steps >= 3
        elif name == "carry":
            advance = float(np.linalg.norm(self._corner()[:2] - self.env._goal[:2])) < 0.03
        elif name == "place":
            advance = float(np.linalg.norm(self._corner() - self.env._goal)) < SUCCESS_DIST
        else:
            advance = reached
        if advance or self.phase_steps > self.PATIENCE[name]:
            self.phase = min(self.phase + 1, len(self.PHASES) - 1)
            self.phase_steps = 0
            self.q_target = None
            if name in ("close", "lift"):
                self.carry_target = None      # re-anchor the rate limiter on the live site
            if name == "close":
                q = np.zeros(4); mujoco.mju_mat2Quat(q, self.base.data.site_xmat[self.site_id])
                self.carry_quat = q           # jaw orientation to hold through the carry
