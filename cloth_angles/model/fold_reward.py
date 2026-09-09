"""SingleCornerFoldEnv's reward and termination, computed from a state vector.

The world model predicts physical state (cloth vertices, left-arm state,
grasp flag), so the fold reward can be evaluated with the environment's own
formula instead of a learned head. Everything here mirrors
cloth_fold_rl/fold_env.py; the one term that cannot be reproduced is the
"unstable" termination, which reads joint accelerations the state omits.

State layout (see cloth_angles/data/state_episode.py): 363 vertex coordinates
then 15 robot values; the end-effector position sits at robot[11:14] and the
grasp flag at robot[14].
"""

from __future__ import annotations

import torch

from cloth_fold_rl.fold_env import (
    CTRL_COST, DRAG_LIMIT, GRASP_BONUS, HOLD_STEPS, SUCCESS_BONUS, SUCCESS_DIST, W_CARRY, W_REACH,
)

VERTEX_DIM = 363
MOVING_CORNER = 10           # cloth_10, welded to the left gripper
ANCHOR_VERTICES = (0, 110)   # cloth_0 and cloth_110 must not be dragged
EE_SLICE = slice(VERTEX_DIM + 11, VERTEX_DIM + 14)
GRASP_INDEX = VERTEX_DIM + 14
DRAG_PENALTY = 5.0


def _vertex(state: torch.Tensor, index: int) -> torch.Tensor:
    return state[..., 3 * index:3 * index + 3]


class FoldReward:
    """Per-episode constants (goal, anchors, start distance) plus batched reward rules.

    goal: [3] or [B, 3]; anchors0: [4, 3] or [B, 4, 3] (cloth_0, _10, _110, _120).
    A batched spec pairs each batch row of the states with its own episode.

    soft=True blends the grasped and free potentials by the clamped grasp
    value instead of thresholding it. On real states (flag exactly 0 or 1)
    this is identical; on predicted states it lets reward gradients reach the
    gripper action through the model's grasp prediction. Success and drag
    rules always use the hard threshold.
    """

    def __init__(self, goal, anchors0, soft: bool = False):
        self.soft = soft
        self.goal = torch.as_tensor(goal, dtype=torch.float32)
        self.anchors0 = torch.as_tensor(anchors0, dtype=torch.float32)
        self.start_dist = torch.clamp((self.anchors0[..., 1, :] - self.goal).norm(dim=-1), min=1e-6)

    def grasped(self, state: torch.Tensor) -> torch.Tensor:
        return state[..., GRASP_INDEX] > 0.5

    def corner_to_goal(self, state: torch.Tensor) -> torch.Tensor:
        return (_vertex(state, MOVING_CORNER) - self.goal).norm(dim=-1)

    def potential(self, state: torch.Tensor) -> torch.Tensor:
        reach = (state[..., EE_SLICE] - _vertex(state, MOVING_CORNER)).norm(dim=-1)
        carried = GRASP_BONUS - W_CARRY * self.corner_to_goal(state)
        free = -W_REACH * reach - W_CARRY * self.start_dist
        if self.soft:
            weight = state[..., GRASP_INDEX].clamp(0.0, 1.0)
            return weight * carried + (1.0 - weight) * free
        return torch.where(self.grasped(state), carried, free)

    def anchor_drift(self, state: torch.Tensor) -> torch.Tensor:
        drifts = [(_vertex(state, v) - self.anchors0[..., i, :]).norm(dim=-1) for v, i in zip(ANCHOR_VERTICES, (0, 2))]
        return torch.stack(drifts, -1).max(-1).values

    def shaping(self, state: torch.Tensor, next_state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.potential(next_state) - self.potential(state) - CTRL_COST * action.pow(2).sum(-1)

    def rollout(self, states: torch.Tensor, actions: torch.Tensor, hold: torch.Tensor | None = None):
        """states: [B, T+1, D], actions: [B, T, A] -> rewards [B, T], terminated [B, T].

        Applies the success bonus once the corner has been placed while
        grasped for HOLD_STEPS consecutive steps, and the drag penalty when an
        anchor corner moves further than DRAG_LIMIT. Rewards after a
        termination are still computed; callers mask them with `terminated`.
        """
        batch, time = actions.shape[:2]
        hold = torch.zeros(batch, dtype=torch.long) if hold is None else hold.clone()
        rewards, terminated = [], []
        for t in range(time):
            nxt = states[:, t + 1]
            reward = self.shaping(states[:, t], nxt, actions[:, t])
            placed = self.grasped(nxt) & (self.corner_to_goal(nxt) < SUCCESS_DIST)
            hold = torch.where(placed, hold + 1, torch.zeros_like(hold))
            success = hold >= HOLD_STEPS
            dragged = ~success & (self.anchor_drift(nxt) > DRAG_LIMIT)
            reward = reward + SUCCESS_BONUS * success.float() - DRAG_PENALTY * dragged.float()
            rewards.append(reward)
            terminated.append(success | dragged)
        return torch.stack(rewards, 1), torch.stack(terminated, 1)


CTRL_INDEX = VERTEX_DIM + 10
GRIPPER_HYSTERESIS = 0.3
GRASP_RADIUS = 0.03


def gripper_transition(state: torch.Tensor, action: torch.Tensor, soft: bool = False):
    """SingleCornerFoldEnv's gripper and weld rule, from state and action.

    Returns (closed, grasp) for the next state. The gripper closes below
    -0.3, opens above +0.3 and otherwise keeps its state; the weld engages
    when the gripper is closed and the end effector is within GRASP_RADIUS
    of the corner, stays while closed, and releases when it opens. soft=True
    replaces the thresholds with linear ramps so gradients reach the gripper
    command and the end-effector position in imagination.
    """
    cmd = action[..., 5]
    closed_now = state[..., CTRL_INDEX] < 0
    gap = (state[..., EE_SLICE] - _vertex(state, MOVING_CORNER)).norm(dim=-1)
    grasp_now = state[..., GRASP_INDEX]
    if soft:
        hold = ((GRIPPER_HYSTERESIS - cmd) / (2 * GRIPPER_HYSTERESIS)).clamp(0.0, 1.0)
        closed = torch.where(cmd < -GRIPPER_HYSTERESIS, torch.ones_like(cmd),
                             torch.where(cmd > GRIPPER_HYSTERESIS, torch.zeros_like(cmd),
                                         torch.where(closed_now, torch.ones_like(cmd), hold)))
        near = ((GRASP_RADIUS + 0.01 - gap) / 0.02).clamp(0.0, 1.0)
        grasp = closed * torch.maximum(grasp_now.clamp(0.0, 1.0), near)
        return closed, grasp
    closed = torch.where(cmd < -GRIPPER_HYSTERESIS, torch.ones_like(closed_now),
                         torch.where(cmd > GRIPPER_HYSTERESIS, torch.zeros_like(closed_now), closed_now))
    grasp = closed & ((grasp_now > 0.5) | (gap < GRASP_RADIUS))
    return closed.float(), grasp.float()
