"""Single-corner edge fold with a PHYSICAL grasp: grabber plates, no weld.

Same task, reward and observation layout as fold_env.SingleCornerFoldEnv, so
every script in this package (expert gate, demos, BC, PPO, video) runs on it
through the --physical flag. What changes:

  * the SO101 jaws carry the scoop ramp + paddle from mujuco/prove_grabber.py
    (added through compile_model's spec_hook -- the stock model is untouched),
  * the weld equality is never engaged; closing the gripper only closes the
    gripper, and holding the cloth is contact + friction,
  * the "grasp" signal, which feeds the reward potential, the info dict and the
    proprio observation slot the weld flag used to occupy, is now physical:
    jaw commanded closed AND the corner sitting inside the closed jaw's pocket,
  * success no longer requires the grasp flag: a corner set down on its target
    is a fold whether or not the jaw is still pinching it.

Run the same pipeline as the weld version, with --physical:

    python -m cloth_fold_rl.prove_feasible --physical --episodes 3
    python -m cloth_fold_rl.collect_demos --physical --episodes 200 --workers 8
    python -m cloth_fold_rl.bc --physical --epochs 30
    python -m cloth_fold_rl.train --physical --run-dir outputs/cloth_fold_rl/physical/run1 \
        --init-from outputs/cloth_fold_rl/physical/bc.zip
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "mujuco"))
from sim_main import ClothFoldEnv, GRIPPER_OPEN, GRIPPER_CLOSED  # noqa: E402
from prove_grabber import make_grabber_hook, calibrate_paddle  # noqa: E402

from cloth_fold_rl.fold_env import SingleCornerFoldEnv, MOVING_CORNER

PINCH_RADIUS = 0.030   # corner within this of the ramp/paddle midpoint = in the jaw...
PINCH_MIN_DZ = -0.020  # ...and not lying under the ramp (an ejected corner sits ~2.5 cm below it)
MAX_EPISODE_STEPS = 250   # the scoop approach needs ~50 more steps than the weld task

# Two task-setup differences from the weld version, both forced by the SO101's
# kinematics with the grabber plates (measured, see README "physical grasp"):
#  * the scoop posture self-collides (gripper body into shoulder body) when the
#    corner is closer than ~15 cm to the arm base, which the +-2.5 cm cloth jitter
#    reaches. The cloth is therefore placed 2 cm further from the left arm.
#  * with the jaw orientation held so the pocket keeps the corner, the arm cannot
#    track the carry past ~x=+0.10; the corner is dropped around x=0. The fold
#    goal is a fraction of the full edge fold (1.0 = the weld task's goal).
CLOTH_SHIFT = np.array([0.02, 0.0])
FOLD_FRACTION = 0.5
PROPRIO_PER_ARM = 25   # joint pos(5) + vel(5) + gripper(1) + ee pose(7) + ee vel(6) + grasp(1)

_PADDLE_POSE = None


def paddle_pose():
    """Kinematic paddle calibration is deterministic; do it once per process."""
    global _PADDLE_POSE
    if _PADDLE_POSE is None:
        _PADDLE_POSE = calibrate_paddle()
    return _PADDLE_POSE


class PhysicalGraspEnv(ClothFoldEnv):
    """ClothFoldEnv with grabber plates on both jaws and the weld cheat disabled."""

    def __init__(self, **kw):
        super().__init__(spec_hook=make_grabber_hook(paddle_pose()), **kw)
        self._ramp_gid = {p: self.model.geom(f"{p}ramp").id for p in self.prefixes}
        self._paddle_gid = {p: self.model.geom(f"{p}paddle").id for p in self.prefixes}

    def set_gripper(self, prefix, command):
        # same hysteresis as the stock env, minus the weld
        if command < -0.3:
            self._gripper_closed[prefix] = True
        elif command > 0.3:
            self._gripper_closed[prefix] = False
        closed = self._gripper_closed[prefix]
        self.data.ctrl[self._gripper_act[prefix]] = GRIPPER_CLOSED if closed else GRIPPER_OPEN
        self.data.eq_active[self._weld_id[prefix]] = 0

    def pinch_point(self, prefix):
        """World position of the pocket between ramp and paddle."""
        return 0.5 * (self.data.geom_xpos[self._ramp_gid[prefix]]
                      + self.data.geom_xpos[self._paddle_gid[prefix]])

    def physical_grasp(self, prefix):
        if not self._gripper_closed[prefix]:
            return False
        corner = self.data.xpos[self._corner_body[prefix]]
        pinch = self.pinch_point(prefix)
        return bool(np.linalg.norm(corner - pinch) < PINCH_RADIUS
                    and corner[2] - pinch[2] > PINCH_MIN_DZ)

    def _get_obs(self):
        obs = super()._get_obs()
        for i, p in enumerate(self.prefixes):
            obs["proprio"][(i + 1) * PROPRIO_PER_ARM - 1] = float(self.physical_grasp(p))
        return obs

    def _step_info(self, terms, reason=None):
        info = super()._step_info(terms, reason)
        info["left_grasp_active"] = self.physical_grasp("left_")
        info["right_grasp_active"] = self.physical_grasp("right_")
        return info


class SingleCornerPhysicalFoldEnv(SingleCornerFoldEnv):
    """fold_env's task on the physical grabber. See module docstring."""

    SUCCESS_NEEDS_GRASP = False

    def __init__(self, max_episode_steps=MAX_EPISODE_STEPS, fold_fraction=FOLD_FRACTION,
                 cloth_shift=CLOTH_SHIFT, **kw):
        base = PhysicalGraspEnv(observation_mode="state", action_mode="joint_delta",
                                max_episode_steps=max_episode_steps)
        self.fold_fraction = fold_fraction
        self.cloth_shift = np.asarray(cloth_shift, dtype=float)
        super().__init__(max_episode_steps=max_episode_steps, base_env=base, **kw)

    def reset(self, seed=None, options=None):
        opts = dict(options or {})
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        if "cloth_pose" not in opts:
            jitter = (self._rng.uniform(-self.cloth_jitter, self.cloth_jitter, size=2)
                      if self.cloth_jitter > 0 else np.zeros(2))
            opts["cloth_pose"] = self.cloth_shift + jitter
        return super().reset(seed=None, options=opts)

    def _goal_for(self, corners0):
        start = corners0[MOVING_CORNER]
        return start + self.fold_fraction * (super()._goal_for(corners0) - start)

    def _grasp_active(self):
        return self.env.physical_grasp("left_")

    def corner_lift(self):
        from sim_main import TABLE_TOP_Z
        return float(self._moving_corner()[2] - TABLE_TOP_Z)

    def step(self, action):
        obs, r, term, trunc, info = super().step(action)
        info["corner_lift"] = self.corner_lift()
        return obs, r, term, trunc, info
