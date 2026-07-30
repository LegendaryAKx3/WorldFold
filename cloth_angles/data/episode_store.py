"""Ordered episode append/load with schema validation.

An episode is a contiguous sequence of transitions:

    { obs_t, action_t, next_obs_t, episode_end }

where obs_t / next_obs_t are float32[N, N] angle fields (flattened to
float32[N * N] on disk) and action_t is float32[A]. Episodes are stored one
per .npz file so they can be appended incrementally and loaded independently.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class Episode:
    obs: np.ndarray          # float32[T, N*N]
    actions: np.ndarray      # float32[T, A]
    next_obs: np.ndarray     # float32[T, N*N]
    episode_end: np.ndarray  # bool[T], True on the transition that ends the episode
    grid_size: int
    action_dim: int
    angle_unit: str          # "radians" | "degrees"
    angle_convention: str    # "signed" | "unsigned"

    def __len__(self) -> int:
        return int(self.obs.shape[0])

    def validate(self) -> None:
        t = len(self)
        n2 = self.grid_size * self.grid_size
        if self.obs.shape != (t, n2):
            raise ValueError(f"obs shape {self.obs.shape} != ({t}, {n2})")
        if self.next_obs.shape != (t, n2):
            raise ValueError(f"next_obs shape {self.next_obs.shape} != ({t}, {n2})")
        if self.actions.shape != (t, self.action_dim):
            raise ValueError(f"actions shape {self.actions.shape} != ({t}, {self.action_dim})")
        if self.episode_end.shape != (t,):
            raise ValueError(f"episode_end shape {self.episode_end.shape} != ({t},)")
        if self.obs.dtype != np.float32 or self.next_obs.dtype != np.float32:
            raise ValueError("obs/next_obs must be float32")
        if self.actions.dtype != np.float32:
            raise ValueError("actions must be float32")
        if self.episode_end.dtype != np.bool_:
            raise ValueError("episode_end must be bool")
        if self.angle_unit not in ("radians", "degrees"):
            raise ValueError(f"unknown angle_unit {self.angle_unit!r}")
        if self.angle_convention not in ("signed", "unsigned"):
            raise ValueError(f"unknown angle_convention {self.angle_convention!r}")
        if not bool(self.episode_end[-1]):
            raise ValueError("episode_end[-1] must be True: an episode must end on its last transition")


class EpisodeStore:
    """Appends episodes to disk as individual .npz files under a directory."""

    def __init__(self, root: str | Path, grid_size: int, action_dim: int,
                 angle_unit: str = "radians", angle_convention: str = "signed"):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.grid_size = grid_size
        self.action_dim = action_dim
        self.angle_unit = angle_unit
        self.angle_convention = angle_convention

    def _next_index(self) -> int:
        existing = sorted(self.root.glob("episode_*.npz"))
        if not existing:
            return 0
        last = existing[-1].stem.split("_")[-1]
        return int(last) + 1

    def append(self, episode: Episode) -> Path:
        if episode.grid_size != self.grid_size or episode.action_dim != self.action_dim:
            raise ValueError("episode grid_size/action_dim does not match store schema")
        episode.validate()
        idx = self._next_index()
        path = self.root / f"episode_{idx:06d}.npz"
        np.savez_compressed(
            path,
            obs=episode.obs,
            actions=episode.actions,
            next_obs=episode.next_obs,
            episode_end=episode.episode_end,
            grid_size=episode.grid_size,
            action_dim=episode.action_dim,
            angle_unit=episode.angle_unit,
            angle_convention=episode.angle_convention,
        )
        return path

    def load(self, path: str | Path) -> Episode:
        data = np.load(path, allow_pickle=False)
        episode = Episode(
            obs=data["obs"].astype(np.float32),
            actions=data["actions"].astype(np.float32),
            next_obs=data["next_obs"].astype(np.float32),
            episode_end=data["episode_end"].astype(np.bool_),
            grid_size=int(data["grid_size"]),
            action_dim=int(data["action_dim"]),
            angle_unit=str(data["angle_unit"]),
            angle_convention=str(data["angle_convention"]),
        )
        episode.validate()
        return episode

    def load_all(self) -> list[Episode]:
        paths = sorted(self.root.glob("episode_*.npz"))
        return [self.load(p) for p in paths]
