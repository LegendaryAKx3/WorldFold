"""State vector of a fold env as recorded by the collector.

observe_state(base, task) returns float32[363 + 15 * n_arms]: 121 cloth
vertices (x, y, z) then, per arm, joint positions(5), joint velocities(5),
gripper command(1), end-effector position(3) and grasp weld flag(1).
"""

from __future__ import annotations

import numpy as np


def robot_state(base, prefix: str) -> np.ndarray:
    site = base._site_id[prefix]
    return np.array([*(base.data.qpos[a] for a in base._arm_qpos_adr[prefix]),
                     *(base.data.qvel[a] for a in base._arm_dof_adr[prefix]),
                     base.data.ctrl[base._gripper_act[prefix]],
                     *base.data.site_xpos[site],
                     float(base.grasp_active(prefix))], dtype=np.float32)


def cloth_vertices(base) -> np.ndarray:
    return base.data.xpos[base._cloth_body_ids].astype(np.float32)


def observe_state(base, task) -> np.ndarray:
    return np.concatenate([cloth_vertices(base).reshape(-1), *(robot_state(base, a.prefix) for a in task.arms)])
