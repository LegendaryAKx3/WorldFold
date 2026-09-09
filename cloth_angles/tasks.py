"""Fold tasks as seen by the world model: state layout, arms, staged moves, success rule.

The state vector is 363 cloth vertex coordinates followed by 15 robot values
per arm (joint pos 5, joint vel 5, gripper command, end-effector position 3,
grasp weld flag). An episode's goal vector holds the start positions of the
goal corners, 3 values per goal slot; anchors0 holds the start positions of
the four corners in CORNER_LIST order. Both come from the env at reset.

    single   the left arm carries cloth_10 to cloth_120's start and holds it
    quarter  both arms fold the y > 0 edge over and release, then the right arm
             carries the stacked cloth_0 + cloth_10 onto cloth_110 and releases
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

VERTEX_DIM = 363
ROBOT_DIM = 15
CORNER_LIST = (0, 10, 110, 120)


def corner_slot(vertex: int) -> int:
    return CORNER_LIST.index(vertex)


@dataclass(frozen=True)
class Arm:
    prefix: str
    robot: int          # offset of its robot block in the state vector
    gripper: int        # index of its gripper command in the action
    joints: slice       # its joint deltas in the action
    weld_corners: tuple # cloth vertices its gripper welds to when closed within grasp_radius

    @property
    def ctrl_index(self) -> int:
        return self.robot + 10

    @property
    def ee_slice(self) -> slice:
        return slice(self.robot + 11, self.robot + 14)

    @property
    def grasp_index(self) -> int:
        return self.robot + 14


@dataclass(frozen=True)
class Move:
    arm: int            # index into Task.arms
    corners: tuple      # cloth vertices carried, reached for at their mean
    goal: int           # goal slot in the goal vector


@dataclass(frozen=True)
class Stage:
    moves: tuple
    anchors: tuple      # (vertex, reference corner): vertex stays within DRAG_LIMIT of the reference's start


@dataclass(frozen=True)
class Task:
    name: str
    arms: tuple
    stages: tuple
    goal_corners: tuple   # corner per goal slot
    action_dim: int
    grasp_radius: float
    release: bool         # placed corners must be let go (else held) to settle
    settle_steps: int     # consecutive settled steps that complete a stage
    max_steps: int        # the env's episode length

    @property
    def state_dim(self) -> int:
        return VERTEX_DIM + ROBOT_DIM * len(self.arms)

    @property
    def goal_dim(self) -> int:
        return 3 * len(self.goal_corners)

    def make_env(self, max_steps=None, cloth_jitter=None):
        kwargs = {} if cloth_jitter is None else {"cloth_jitter": cloth_jitter}
        kwargs["max_episode_steps"] = self.max_steps if max_steps is None else max_steps
        if self.name == "single":
            from cloth_fold_rl.fold_env import SingleCornerFoldEnv
            return SingleCornerFoldEnv(**kwargs)
        from cloth_fold_rl.quarter_fold_env import QuarterFoldEnv
        return QuarterFoldEnv(**kwargs)

    def make_expert(self, env, seed):
        if self.name == "single":
            from cloth_fold_rl.expert import FoldExpert
            return FoldExpert(env, seed=seed)
        from cloth_fold_rl.quarter_fold_expert import QuarterFoldExpert
        return QuarterFoldExpert(env, seed=seed)

    def corner_starts(self, env) -> np.ndarray:
        """float32[4, 3]: start positions of CORNER_LIST, from a reset env."""
        base = env.unwrapped
        if self.name == "single":
            return np.asarray(env._anchors0, dtype=np.float32)
        return np.asarray(env._start[list(CORNER_LIST)], dtype=np.float32)

    def goals(self, env) -> np.ndarray:
        """float32[goal_dim]: start positions of the goal corners."""
        starts = self.corner_starts(env)
        return np.concatenate([starts[corner_slot(c)] for c in self.goal_corners])

    def stage(self, info) -> int:
        return int(info.get("stage", 0))

    def grasped(self, info) -> bool:
        g = info["grasped"]
        return any(g.values()) if isinstance(g, dict) else bool(g)


LEFT = Arm("left_", VERTEX_DIM, 5, slice(0, 5), weld_corners=(10,))
SINGLE = Task("single", arms=(LEFT,),
              stages=(Stage(moves=(Move(0, (10,), 0),), anchors=((0, 0), (110, 110))),),
              goal_corners=(120,), action_dim=6, grasp_radius=0.03, release=False, settle_steps=10, max_steps=200)

RIGHT_Q = Arm("right_", VERTEX_DIM + ROBOT_DIM, 11, slice(6, 11), weld_corners=(120, 0, 10))
QUARTER = Task("quarter", arms=(LEFT, RIGHT_Q),
               stages=(Stage(moves=(Move(0, (10,), 0), Move(1, (120,), 1)), anchors=((0, 0), (110, 110))),
                       Stage(moves=(Move(1, (0, 10), 1),), anchors=((110, 110), (120, 110)))),
               goal_corners=(0, 110), action_dim=12, grasp_radius=0.04, release=True, settle_steps=20, max_steps=400)
TASKS = {"single": SINGLE, "quarter": QUARTER}
