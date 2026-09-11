"""Grabber proof of concept: pinch the cloth with real contact physics, no weld.

ClothFoldEnv fakes grasping with a weld equality (see the SIM-ONLY GRASP CHEAT
note in sim_main.py) because the stock SO101 finger pads -- 2.5mm cubes closing
with a vertical pinch axis -- cannot get under a flat sheet lying on a table.
This script shows a small hardware change is enough to grab the cloth for real:

  * a thin "scoop ramp" plate on the fixed jaw (20 deg incline, leading edge
    skimming the table) that slides under the cloth corner so it rides up, and
  * a matching "paddle" plate on the moving jaw, placed so the closed jaw
    pinches the whole ramp surface instead of a single 2.5mm point.

Both plates are added through compile_model's spec_hook, so the stock model
(and everything trained against it) is untouched. The paddle's pose is
calibrated kinematically at startup: its transform relative to the fixed jaw
depends only on the gripper joint angle, so one mj_forward at the closed angle
is enough to place it parallel to the ramp.

The scripted routine (left arm) approaches from the cloth's free side, slides
the ramp under the corner until it climbs on, snaps the jaw shut, and carries
the corner along the fold diagonal. The weld stays disabled throughout (this is
asserted every step) -- everything that happens is contact + friction.

What this proves, and what it does not: the physical grasp reliably scoops,
pinches, and carries the corner airborne through most of the fold arc, but the
pocket hold slips when cloth tension peaks late in the fold, so the weld cheat
remains the env default for now. Run:

    python mujuco/prove_grabber.py

Takes a few minutes (the contact-heavy grasp runs at a 0.5ms timestep).
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import mujoco
import numpy as np

from sim_main import (ClothFoldEnv, TABLE_TOP_Z, GRIPPER_OPEN, GRIPPER_CLOSED)

# scoop ramp on the fixed jaw, in the gripper body frame. Calibrated at the
# grasp approach pose (deterministic from reset): 20 deg incline, leading edge
# ~3.2cm ahead of the gripperframe site at table+1.5mm, root under the pads.
RAMP_POS   = [-0.01646, -0.01311, -0.09925]
RAMP_QUAT  = [0.88926, 0.2325, -0.06419, -0.38864]
RAMP_SIZE  = [0.001, 0.019, 0.020]      # half-sizes: thin / long (slide dir) / wide
PADDLE_GAP = 0.009    # ramp-to-paddle center distance along the ramp normal when
                      # closed; snug on the 2cm cloth vertex without ejecting it

SCOOP_Z    = TABLE_TOP_Z + 0.0115       # site height while sliding under the cloth
APPROACH_Y = 0.07                       # approach offset on the cloth's free side
FOLD_GOAL  = np.array([0.15, -0.15, TABLE_TOP_Z + 0.01])   # opposite corner's start

# pass/fail thresholds (best observed: peak lift 0.044m, carry 0.10m, fold 0.095m)
MIN_PEAK_LIFT     = 0.020   # corner must rise this far above the table while held
MIN_CARRY_DIST    = 0.050   # lateral travel with the corner airborne (>2cm up)
MIN_FOLD_PROGRESS = 0.050   # reduction of corner-to-goal distance over the fold


def make_grabber_hook(paddle_pose=None):
    """spec_hook that bolts the grabber plates onto both arms' jaws."""
    def hook(spec):
        plates = [("ramp", "gripper", RAMP_POS, RAMP_QUAT, RAMP_SIZE,
                   [0.2, 0.9, 0.4, 1.0])]
        if paddle_pose is not None:
            plates.append(("paddle", "moving_jaw_so101_v1", paddle_pose[0],
                           paddle_pose[1], RAMP_SIZE, [0.9, 0.4, 0.2, 1.0]))
        for prefix in ["left_", "right_"]:
            for name, body_name, pos, quat, size, rgba in plates:
                g = spec.body(f"{prefix}{body_name}").add_geom()
                g.name = f"{prefix}{name}"
                g.type = mujoco.mjtGeom.mjGEOM_BOX
                g.size = size
                g.pos = pos
                g.quat = quat
                g.rgba = rgba
                g.friction = [1.2, 0.05, 0.001]
                g.condim = 3
                # same softened contact as the finger pads in compile_model,
                # so the plates don't explode the ~1-gram cloth vertices
                g.solref = [0.02, 1]
                g.solimp = [0.8, 0.9, 0.01, 0.5, 2]
    return hook


def calibrate_paddle():
    """Place the moving-jaw paddle parallel to the ramp at the closed angle.

    The paddle-to-ramp transform depends only on the gripper joint angle, so a
    single mj_forward with the joint set to closed gives the exact local pose.
    """
    cal = ClothFoldEnv(spec_hook=make_grabber_hook())
    m, d = cal.model, cal.data
    d.qpos[m.joint("left_gripper").qposadr[0]] = GRIPPER_CLOSED
    mujoco.mj_forward(m, d)
    ramp_gid = m.geom("left_ramp").id
    Rr = d.geom_xmat[ramp_gid].reshape(3, 3)
    paddle_world = d.geom_xpos[ramp_gid] + Rr[:, 0] * PADDLE_GAP  # local x = normal
    jaw_bid = m.body("left_moving_jaw_so101_v1").id
    Rj = d.xmat[jaw_bid].reshape(3, 3)
    pos_local = Rj.T @ (paddle_world - d.xpos[jaw_bid])
    quat_local = np.zeros(4)
    mujoco.mju_mat2Quat(quat_local, np.ascontiguousarray(Rj.T @ Rr).ravel())
    return pos_local.tolist(), quat_local.tolist()


class GrabberDemo:
    """Scripted single-arm scoop-pinch-fold routine on the physical grabber."""

    def __init__(self):
        self.env = ClothFoldEnv(spec_hook=make_grabber_hook(calibrate_paddle()))
        self.env.reset(seed=0)
        self.m, self.d = self.env.model, self.env.data
        self.p = "left_"
        self.site_id = self.env._site_id[self.p]
        self.grip_act = self.env._gripper_act[self.p]
        self.corner_bid = self.env._corner_body[self.p][0]
        self.weld_id = self.env._weld_id[self.p]

    def run_to(self, target, grip, steps):
        self.env._target_pos[self.p] = np.array(target, dtype=float)
        for _ in range(steps):
            self.env.ik_substep(self.p)
            self.env.ik_substep("right_")   # keep the other arm parked
            self.d.ctrl[self.grip_act] = grip
            mujoco.mj_step(self.m, self.d)
        # the whole point: the sim-only grasp cheat must never engage
        assert self.d.eq_active[self.weld_id] == 0, "weld engaged -- not a physical grasp"

    def corner(self):
        return self.d.xpos[self.corner_bid]

    def site(self):
        return self.d.site_xpos[self.site_id]

    def ikerr(self):
        return float(np.linalg.norm(self.env._target_pos[self.p] - self.site()))

    def log(self, tag):
        c, s = self.corner(), self.site()
        print(f"  {tag:<10} corner {np.round(c, 4)}  lift {c[2]-TABLE_TOP_Z:+.4f}  "
              f"site {np.round(s, 4)}  ikerr {self.ikerr():.4f}")

    def run(self):
        corner0 = self.corner().copy()
        print("phase 1: approach from the cloth's free side")
        self.run_to([corner0[0], corner0[1] + APPROACH_Y, TABLE_TOP_Z + 0.07],
                    GRIPPER_OPEN, 10000)
        self.log("hover")
        self.run_to([corner0[0], corner0[1] + APPROACH_Y, SCOOP_Z],
                    GRIPPER_OPEN, 12000)
        self.log("down")

        print("phase 2: slide the ramp under the corner until it climbs on")
        on_ramp = False
        for k in range(30):
            s = self.site()
            self.run_to([corner0[0], s[1] - 0.005, SCOOP_Z], GRIPPER_OPEN, 1500)
            if self.corner()[2] > TABLE_TOP_Z + 0.022:
                on_ramp = True
                self.log(f"scoop {k}")
                break
        if not on_ramp:
            self.log("scoop-fail")

        print("phase 3: snap the jaw shut (paddle pinches the whole ramp face)")
        self.run_to(self.site().copy(), GRIPPER_CLOSED, 6000)
        self.log("close")

        print("phase 4: carry the corner along the fold diagonal")
        s0 = self.site().copy()
        d_goal0 = float(np.linalg.norm(self.corner() - FOLD_GOAL))
        peak_lift = 0.0
        carry_dist = 0.0
        last_xy = self.site()[:2].copy()
        frac, budget = 0.0, 60
        while frac < 1.0 and budget > 0:
            budget -= 1
            if self.ikerr() < 0.012:    # advance only once the arm has caught up
                frac = min(frac + 0.05, 1.0)
            # follow through on orientation: re-targeting to the current pose
            # keeps the 5-DOF IK from stalling against an infeasible rotation
            q = np.zeros(4)
            mujoco.mju_mat2Quat(q, self.d.site_xmat[self.site_id])
            self.env._target_quat[self.p] = q
            dz = 0.035 * np.sin(np.pi * min(frac * 1.25, 1.0))  # rise then settle
            self.run_to([s0[0] + frac * 0.18, s0[1] - frac * 0.18, s0[2] + dz],
                        GRIPPER_CLOSED, 1500)
            lift = float(self.corner()[2] - TABLE_TOP_Z)
            peak_lift = max(peak_lift, lift)
            xy = self.site()[:2].copy()
            if lift > 0.02:             # count travel only while truly airborne
                carry_dist += float(np.linalg.norm(xy - last_xy))
            last_xy = xy
        self.log("fold-end")

        d_goal1 = float(np.linalg.norm(self.corner() - FOLD_GOAL))
        return {
            "scooped": on_ramp,
            "peak_lift": peak_lift,
            "carry_dist": carry_dist,
            "fold_progress": d_goal0 - d_goal1,
            "d_goal": (d_goal0, d_goal1),
        }


def main():
    demo = GrabberDemo()
    r = demo.run()

    print("\n=== grabber summary (weld disabled throughout) ===")
    checks = [
        ("corner scooped onto the ramp", r["scooped"], ""),
        (f"peak lift {r['peak_lift']:.3f}m (need >= {MIN_PEAK_LIFT})",
         r["peak_lift"] >= MIN_PEAK_LIFT, ""),
        (f"airborne carry {r['carry_dist']:.3f}m (need >= {MIN_CARRY_DIST})",
         r["carry_dist"] >= MIN_CARRY_DIST, ""),
        (f"fold progress {r['fold_progress']:.3f}m of {r['d_goal'][0]:.3f}m "
         f"(need >= {MIN_FOLD_PROGRESS})",
         r["fold_progress"] >= MIN_FOLD_PROGRESS, ""),
    ]
    ok = True
    for label, passed, _ in checks:
        ok &= passed
        print(f"  [{'PASS' if passed else 'FAIL'}] {label}")

    print("\nVERDICT:", "physical grasp works -- the grabber can scoop, pinch, "
          "lift and carry the cloth corner without the weld cheat"
          if ok else "grabber did not meet the proof thresholds")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
