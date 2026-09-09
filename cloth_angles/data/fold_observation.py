"""State vector of SingleCornerFoldEnv as recorded by the collector.

observe_state(base) returns float32[378]: 121 cloth vertices (x, y, z) then
the left arm's joint positions(5), joint velocities(5), gripper command(1),
end-effector position(3) and grasp weld flag(1).
"""

from __future__ import annotations

import numpy as np

PREFIX = "left_"


def robot_state(base) -> np.ndarray:
    site = base._site_id[PREFIX]
    return np.array([*(base.data.qpos[a] for a in base._arm_qpos_adr[PREFIX]),
                     *(base.data.qvel[a] for a in base._arm_dof_adr[PREFIX]),
                     base.data.ctrl[base._gripper_act[PREFIX]],
                     *base.data.site_xpos[site],
                     float(base.data.eq_active[base._weld_id[PREFIX]])], dtype=np.float32)


def cloth_vertices(base) -> np.ndarray:
    return base.data.xpos[base._cloth_body_ids].astype(np.float32)


def observe_state(base) -> np.ndarray:
    return np.concatenate([cloth_vertices(base).reshape(-1), robot_state(base)])
