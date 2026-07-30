"""Collect varied episodes from ClothFoldEnv into an EpisodeStore, for
training the cloth-angle RSSM world model.

Usage:
    python -m cloth_angles.collect_episodes --config cloth_angles/config.yaml \
        --episodes-per-kind 8 --seed 0

Collects three kinds of episodes, per the spec's training procedure ("varied
episodes with no movement, partial folds, and recovery motions"):

- "still":    zero action every step (passive settling / no movement)
- "fold":     a scripted policy that drives both grippers toward the cloth's
              diagonal-opposite corners and closes on approach (partial
              fold attempts, not necessarily reaching the success threshold)
- "recovery": random actions for a random prefix of the episode, then zero
              action for the remainder (perturb-then-settle "recovery")

The angle field is derived (not stored by the env itself) from cloth vertex
world positions via cloth_angles.data.angle_field.compute_angle_field, so
the model only ever sees precomputed scalar angles, never raw positions or
normals.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "mujuco"))

from cloth_angles.data.angle_field import compute_angle_field, vertices_grid_from_flat
from cloth_angles.data.episode_store import Episode, EpisodeStore

EPISODE_KINDS = ("still", "fold", "recovery")


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def make_env(max_episode_steps: int):
    from sim_main import ClothFoldEnv  # noqa: PLC0415 -- mujuco/ only importable once sys.path is set above
    return ClothFoldEnv(max_episode_steps=max_episode_steps, observation_mode="state")


def angle_obs(env, grid_size: int, signed: bool) -> np.ndarray:
    xpos = env.data.xpos[env._cloth_body_ids].copy()
    vertices = vertices_grid_from_flat(xpos, grid_size)
    return compute_angle_field(vertices, signed=signed).reshape(-1).astype(np.float32)


def still_action(env, rng, t: int) -> np.ndarray:
    return np.zeros(env.action_space.shape, dtype=np.float32)


def fold_action(env, rng, t: int) -> np.ndarray:
    """Drives both grippers toward their target corner and closes once close,
    mirroring test_idk's scripted demo policy in mujuco/sim_main.py (a partial
    fold attempt, not guaranteed to reach the success threshold).
    """
    action = np.zeros(env.action_space.shape, dtype=np.float32)
    pos_slice = {"left_": 0, "right_": 7}
    grip_index = {"left_": 6, "right_": 13}
    for prefix in env.prefixes:
        site = env.data.site_xpos[env._site_id[prefix]]
        corner = env.data.xpos[env._corner_body[prefix]]
        direction = corner - site
        cmd = np.clip(direction * 20.0, -1.0, 1.0)
        start = pos_slice[prefix]
        action[start:start + 3] = cmd
        gap = float(np.linalg.norm(direction))
        action[grip_index[prefix]] = -1.0 if gap < 0.02 else 1.0
    return action


def recovery_action(env, rng, t: int, perturb_steps: int) -> np.ndarray:
    if t < perturb_steps:
        return rng.uniform(-1.0, 1.0, size=env.action_space.shape).astype(np.float32)
    return np.zeros(env.action_space.shape, dtype=np.float32)


def collect_episode(env, kind: str, seed: int, grid_size: int, signed: bool, rng: np.random.Generator) -> Episode:
    obs_t, info = env.reset(seed=seed)
    angle = angle_obs(env, grid_size, signed)

    obs_list, action_list, next_obs_list = [], [], []
    perturb_steps = int(rng.integers(1, max(2, env.max_episode_steps // 3)))

    for t in range(env.max_episode_steps):
        if kind == "still":
            action = still_action(env, rng, t)
        elif kind == "fold":
            action = fold_action(env, rng, t)
        elif kind == "recovery":
            action = recovery_action(env, rng, t, perturb_steps)
        else:
            raise ValueError(f"unknown episode kind {kind!r}")

        obs_list.append(angle)
        env.step(action)
        angle = angle_obs(env, grid_size, signed)
        next_obs_list.append(angle)
        action_list.append(action)

    n2 = grid_size * grid_size
    obs = np.stack(obs_list).astype(np.float32).reshape(-1, n2)
    next_obs = np.stack(next_obs_list).astype(np.float32).reshape(-1, n2)
    actions = np.stack(action_list).astype(np.float32)
    episode_end = np.zeros(len(obs), dtype=bool)
    episode_end[-1] = True

    return Episode(
        obs=obs, actions=actions, next_obs=next_obs, episode_end=episode_end,
        grid_size=grid_size, action_dim=int(env.action_space.shape[0]),
        angle_unit="radians", angle_convention="signed" if signed else "unsigned",
    )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to config.yaml")
    parser.add_argument("--episodes-per-kind", type=int, default=8,
                         help=f"How many episodes to collect for each of {EPISODE_KINDS}")
    parser.add_argument("--max-episode-steps", type=int, default=200,
                         help="~100+ steps needed before the scripted fold policy reaches/grasps the cloth")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)
    data_cfg = config["data"]
    grid_size = data_cfg["grid_size"]
    signed = data_cfg["angle_convention"] == "signed"

    env = make_env(args.max_episode_steps)
    store = EpisodeStore(
        data_cfg["episode_dir"], grid_size=grid_size, action_dim=int(env.action_space.shape[0]),
        angle_unit=data_cfg["angle_unit"], angle_convention=data_cfg["angle_convention"],
    )
    rng = np.random.default_rng(args.seed)

    total = 0
    for kind in EPISODE_KINDS:
        for i in range(args.episodes_per_kind):
            seed = int(rng.integers(0, 2**31 - 1))
            episode = collect_episode(env, kind, seed, grid_size, signed, rng)
            store.append(episode)
            total += 1
            print(f"[{kind}] episode {i + 1}/{args.episodes_per_kind} (seed={seed}, {len(episode)} steps)")

    env.close()
    print(f"\nwrote {total} episodes to {data_cfg['episode_dir']}")


if __name__ == "__main__":
    main()
