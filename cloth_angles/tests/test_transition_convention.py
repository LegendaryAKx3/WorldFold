"""Asserts the transition convention: actions[:, t] maps obs[:, t] to next_obs[:, t].

This is a spec requirement (data/sequence_replay.py docstring), not just a
comment -- a synthetic episode with a known, distinguishable action->delta
mapping must round-trip correctly through EpisodeStore -> SequenceReplay.
"""

import numpy as np
import pytest

from cloth_angles.data.episode_store import Episode, EpisodeStore
from cloth_angles.data.sequence_replay import SequenceReplay

GRID_SIZE = 4
ACTION_DIM = 2
N2 = GRID_SIZE * GRID_SIZE


def _make_deterministic_episode(start_angle: float, n_steps: int) -> Episode:
    """Each step's action fully determines its next_obs: next_obs = obs + action[0]
    (broadcast, then wrapped to [-pi, pi]), so recovering action[t] from
    (obs[t], next_obs[t]) proves the alignment actions[:, t] -> obs[:, t] -> next_obs[:, t].
    """
    obs = np.zeros((n_steps, N2), dtype=np.float32)
    actions = np.zeros((n_steps, ACTION_DIM), dtype=np.float32)
    next_obs = np.zeros((n_steps, N2), dtype=np.float32)

    angle = start_angle
    rng = np.random.default_rng(0)
    for t in range(n_steps):
        # distinct, recoverable action per step: action[0] encodes step index directly
        delta = 0.1 * (t + 1)
        action = np.array([delta, rng.uniform(-1, 1)], dtype=np.float32)
        obs[t] = angle
        new_angle = np.arctan2(np.sin(angle + delta), np.cos(angle + delta))
        next_obs[t] = new_angle
        actions[t] = action
        angle = new_angle

    episode_end = np.zeros(n_steps, dtype=bool)
    episode_end[-1] = True
    return Episode(obs=obs, actions=actions, next_obs=next_obs, episode_end=episode_end,
                    grid_size=GRID_SIZE, action_dim=ACTION_DIM,
                    angle_unit="radians", angle_convention="signed")


def test_episode_store_round_trip_preserves_alignment(tmp_path):
    episode = _make_deterministic_episode(start_angle=0.3, n_steps=6)
    store = EpisodeStore(tmp_path, grid_size=GRID_SIZE, action_dim=ACTION_DIM)
    path = store.append(episode)
    loaded = store.load(path)

    for t in range(len(episode)):
        delta = 0.1 * (t + 1)
        expected_next = np.arctan2(np.sin(episode.obs[t, 0] + delta), np.cos(episode.obs[t, 0] + delta))
        assert loaded.actions[t, 0] == pytest.approx(delta)
        assert loaded.next_obs[t, 0] == pytest.approx(expected_next, abs=1e-5)
        assert loaded.obs[t, 0] == pytest.approx(episode.obs[t, 0])


def test_sequence_replay_preserves_transition_alignment():
    """Sample many sequences and check, for every valid (unmasked) step, that
    next_obs[b, t] is EXACTLY the deterministic function of obs[b, t] and
    actions[b, t] that _make_deterministic_episode encodes. If SequenceReplay
    ever mis-shifted obs/actions/next_obs relative to each other, this fails.
    """
    episodes = [_make_deterministic_episode(start_angle=float(i), n_steps=5 + i) for i in range(4)]
    replay = SequenceReplay(episodes, seq_len=8, seed=0)

    for _ in range(50):
        obs, actions, next_obs, mask, is_first = replay.sample(batch_size=6)
        batch, time = mask.shape
        for b in range(batch):
            for t in range(time):
                if mask[b, t] == 0:
                    continue
                delta = actions[b, t, 0]
                expected_next = np.arctan2(np.sin(obs[b, t, 0] + delta), np.cos(obs[b, t, 0] + delta))
                assert next_obs[b, t, 0] == pytest.approx(expected_next, abs=1e-5), (
                    f"transition misaligned at b={b}, t={t}: "
                    f"obs={obs[b, t, 0]}, action={delta}, next_obs={next_obs[b, t, 0]}, "
                    f"expected={expected_next}"
                )


def test_sequence_replay_masks_padded_tail_and_resets_on_is_first():
    """A short episode inside a longer seq_len must pad with mask=0 once the
    chunk runs past the episode's end (whatever start offset within the
    episode was sampled), and every sampled chunk's first (unmasked) step
    must be marked is_first (used by RSSM.observe to reset state at episode
    boundaries) with no other step marked.
    """
    n_steps = 3
    short_episode = _make_deterministic_episode(start_angle=0.0, n_steps=n_steps)
    replay = SequenceReplay([short_episode], seq_len=8, seed=0)

    for _ in range(50):
        obs, actions, next_obs, mask, is_first = replay.sample(batch_size=1)
        valid = int(mask[0].sum())
        assert 1 <= valid <= n_steps
        # mask must be a contiguous run of 1s from the start, then all 0s
        assert list(mask[0, :valid]) == [1.0] * valid
        assert list(mask[0, valid:]) == [0.0] * (8 - valid)
        # exactly one is_first, at step 0 (every sampled chunk starts a fresh state)
        assert is_first[0, 0] == True  # noqa: E712 -- explicit bool check reads clearer here
        assert not np.any(is_first[0, 1:])
