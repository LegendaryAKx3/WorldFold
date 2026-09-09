"""Full-state episodes from the single-corner fold environment.

An episode stores T+1 observed states and the T actions between them:

    vertices[t]  float32[T+1, N*N, 3]  cloth vertex world positions
    robot[t]     float32[T+1, R]       left arm: joint pos(5), joint vel(5),
                                       gripper ctrl(1), end-effector pos(3),
                                       grasp weld active(1)
    actions[t]   float32[T, A]         action taken between state t and t+1
    rewards[t]   float32[T]            environment reward for that transition (optional)
    terminated[t] bool[T]              environment terminated after it (optional)
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path

import numpy as np

ROBOT_DIM = 15
GRASP_INDEX = 14


@dataclass
class StateEpisode:
    vertices: np.ndarray
    robot: np.ndarray
    actions: np.ndarray
    metadata: dict = field(default_factory=dict)
    rewards: np.ndarray | None = None
    terminated: np.ndarray | None = None

    def __len__(self) -> int:
        """Number of transitions."""
        return int(self.actions.shape[0])

    def validate(self) -> None:
        t = len(self)
        if self.vertices.shape[0] != t + 1 or self.robot.shape[0] != t + 1:
            raise ValueError("vertices and robot must hold one more state than there are actions")
        if self.vertices.ndim != 3 or self.vertices.shape[2] != 3:
            raise ValueError(f"vertices must be [T+1, N*N, 3], got {self.vertices.shape}")
        if self.robot.shape[1] != ROBOT_DIM:
            raise ValueError(f"robot must be [T+1, {ROBOT_DIM}], got {self.robot.shape}")
        for name in ("vertices", "robot", "actions"):
            if getattr(self, name).dtype != np.float32:
                raise ValueError(f"{name} must be float32")
        if self.rewards is not None and (self.rewards.shape != (t,) or self.rewards.dtype != np.float32):
            raise ValueError("rewards must be float32[T]")
        if self.terminated is not None and (self.terminated.shape != (t,) or self.terminated.dtype != np.bool_):
            raise ValueError("terminated must be bool[T]")

    def states(self) -> np.ndarray:
        """float32[T+1, N*N*3 + R]: flattened vertices followed by robot state."""
        return np.concatenate([self.vertices.reshape(len(self) + 1, -1), self.robot], axis=1)


class StateEpisodeStore:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def append(self, episode: StateEpisode) -> Path:
        episode.validate()
        existing = sorted(self.root.glob("episode_*.npz"))
        index = int(existing[-1].stem.split("_")[-1]) + 1 if existing else 0
        path = self.root / f"episode_{index:06d}.npz"
        extra = {k: v for k, v in (("rewards", episode.rewards), ("terminated", episode.terminated)) if v is not None}
        np.savez_compressed(path, vertices=episode.vertices, robot=episode.robot,
                            actions=episode.actions, metadata=json.dumps(episode.metadata), **extra)
        return path

    def load(self, path: str | Path) -> StateEpisode:
        data = np.load(path, allow_pickle=False)
        episode = StateEpisode(vertices=data["vertices"], robot=data["robot"], actions=data["actions"],
                               metadata=json.loads(str(data["metadata"])),
                               rewards=data["rewards"] if "rewards" in data else None,
                               terminated=data["terminated"] if "terminated" in data else None)
        episode.validate()
        return episode

    def load_all(self) -> list[StateEpisode]:
        return [self.load(p) for p in sorted(self.root.glob("episode_*.npz"))]
