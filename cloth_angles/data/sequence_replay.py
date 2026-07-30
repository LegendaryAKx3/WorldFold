"""Contiguous sequence sampling with padding masks, for RSSM training.

Sampling is by contiguous chunks of a fixed sequence length, never by shuffled
individual transitions, so the recurrent state carries real history within a
batch element. Sequences never cross an episode boundary: a chunk that would
run past `episode_end` is padded instead, and the mask marks the padded tail
invalid so it contributes to no loss.
"""

from __future__ import annotations

import numpy as np

from cloth_angles.data.episode_store import Episode


class SequenceReplay:
    """Holds episodes in memory and samples contiguous [batch, time, ...] chunks.

    The transition convention is explicit: actions[:, t] maps obs[:, t] to
    next_obs[:, t]. Episode boundaries reset the RSSM state; a batch element
    never contains transitions from two different episodes.
    """

    def __init__(self, episodes: list[Episode], seq_len: int, seed: int | None = None):
        if not episodes:
            raise ValueError("SequenceReplay requires at least one episode")
        self.episodes = episodes
        self.seq_len = seq_len
        self.grid_size = episodes[0].grid_size
        self.action_dim = episodes[0].action_dim
        for ep in episodes[1:]:
            if ep.grid_size != self.grid_size or ep.action_dim != self.action_dim:
                raise ValueError("all episodes in a SequenceReplay must share grid_size/action_dim")
        self._rng = np.random.default_rng(seed)

        # Only episodes with at least one transition are samplable.
        self._lengths = np.array([len(ep) for ep in self.episodes])
        if np.all(self._lengths < 1):
            raise ValueError("all episodes are empty")

    def _sample_one(self):
        ep_idx = int(self._rng.integers(0, len(self.episodes)))
        episode = self.episodes[ep_idx]
        t = len(episode)
        start = int(self._rng.integers(0, t))
        end = min(start + self.seq_len, t)
        length = end - start

        n2 = self.grid_size * self.grid_size
        obs = np.zeros((self.seq_len, n2), dtype=np.float32)
        actions = np.zeros((self.seq_len, self.action_dim), dtype=np.float32)
        next_obs = np.zeros((self.seq_len, n2), dtype=np.float32)
        mask = np.zeros((self.seq_len,), dtype=np.float32)
        is_first = np.zeros((self.seq_len,), dtype=np.bool_)

        obs[:length] = episode.obs[start:end]
        actions[:length] = episode.actions[start:end]
        next_obs[:length] = episode.next_obs[start:end]
        mask[:length] = 1.0
        is_first[0] = True

        return obs, actions, next_obs, mask, is_first

    def sample(self, batch_size: int):
        """Returns obs, actions, next_obs, mask, is_first, each [batch, time, ...].

        mask[b, t] == 0 marks a padded (post-episode-end) step: it must not
        contribute to angle or KL loss. is_first[b, t] marks the first valid
        step of a batch element, where the RSSM state should be reset.
        """
        batch = [self._sample_one() for _ in range(batch_size)]
        obs, actions, next_obs, mask, is_first = zip(*batch)
        return (
            np.stack(obs, axis=0),
            np.stack(actions, axis=0),
            np.stack(next_obs, axis=0),
            np.stack(mask, axis=0),
            np.stack(is_first, axis=0),
        )
