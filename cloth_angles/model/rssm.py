"""Recurrent state-space model (RSSM): GRU deterministic state + categorical
stochastic latent, following the spec's minimal DreamerV4-style design.

    e_t = encoder(obs_t)
    h_t = GRU(h_{t-1}, concat(z_{t-1}, action_{t-1}))
    z_t ~ posterior(z | h_t, e_t)
    z_prior_t ~ prior(z | h_t)

`observe()` is used during sequence training: it has access to real
observation embeddings at every step and samples z from the posterior.

`imagine()` is open-loop: it consumes only an initial latent and a sequence
of future actions, sampling z from the prior at every step. It must never see
future true observations, so it is the only valid path for multi-step
rollout evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


@dataclass
class RSSMState:
    h: torch.Tensor       # [batch, h_dim] deterministic state
    z: torch.Tensor       # [batch, n_categoricals * n_classes] sampled stochastic state (flattened one-hot)
    prior_logits: torch.Tensor      # [batch, n_categoricals, n_classes]
    posterior_logits: torch.Tensor | None  # None when produced by imagine()

    def feature(self) -> torch.Tensor:
        """Concatenated [h, z], the standard RSSM feature used for decoding."""
        return torch.cat([self.h, self.z], dim=-1)


def _sample_categorical(logits: torch.Tensor) -> torch.Tensor:
    """logits: [..., n_categoricals, n_classes] -> straight-through one-hot sample, flattened."""
    dist = torch.distributions.OneHotCategoricalStraightThrough(logits=logits)
    sample = dist.rsample()
    return sample.reshape(*sample.shape[:-2], -1)


class RSSM(nn.Module):
    def __init__(self, embed_dim: int, action_dim: int, h_dim: int = 128,
                 n_categoricals: int = 8, n_classes: int = 8, hidden_dim: int = 128):
        super().__init__()
        self.h_dim = h_dim
        self.n_categoricals = n_categoricals
        self.n_classes = n_classes
        self.z_dim = n_categoricals * n_classes

        self.gru_input = nn.Linear(self.z_dim + action_dim, h_dim)
        self.gru_cell = nn.GRUCell(h_dim, h_dim)

        self.prior_net = nn.Sequential(
            nn.Linear(h_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, self.z_dim),
        )
        self.posterior_net = nn.Sequential(
            nn.Linear(h_dim + embed_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, self.z_dim),
        )

    def initial_state(self, batch_size: int, device: torch.device) -> RSSMState:
        h = torch.zeros(batch_size, self.h_dim, device=device)
        z = torch.zeros(batch_size, self.z_dim, device=device)
        prior_logits = torch.zeros(batch_size, self.n_categoricals, self.n_classes, device=device)
        return RSSMState(h=h, z=z, prior_logits=prior_logits, posterior_logits=None)

    def _step_deterministic(self, prev: RSSMState, prev_action: torch.Tensor) -> torch.Tensor:
        gru_in = F.elu(self.gru_input(torch.cat([prev.z, prev_action], dim=-1)))
        return self.gru_cell(gru_in, prev.h)

    def _prior(self, h: torch.Tensor) -> torch.Tensor:
        return self.prior_net(h).reshape(-1, self.n_categoricals, self.n_classes)

    def _posterior(self, h: torch.Tensor, embed: torch.Tensor) -> torch.Tensor:
        return self.posterior_net(torch.cat([h, embed], dim=-1)).reshape(-1, self.n_categoricals, self.n_classes)

    def observe_step(self, prev: RSSMState, prev_action: torch.Tensor, embed: torch.Tensor) -> RSSMState:
        h = self._step_deterministic(prev, prev_action)
        prior_logits = self._prior(h)
        posterior_logits = self._posterior(h, embed)
        z = _sample_categorical(posterior_logits)
        return RSSMState(h=h, z=z, prior_logits=prior_logits, posterior_logits=posterior_logits)

    def imagine_step(self, prev: RSSMState, prev_action: torch.Tensor) -> RSSMState:
        h = self._step_deterministic(prev, prev_action)
        prior_logits = self._prior(h)
        z = _sample_categorical(prior_logits)
        return RSSMState(h=h, z=z, prior_logits=prior_logits, posterior_logits=None)

    def observe(self, embeds: torch.Tensor, actions: torch.Tensor, is_first: torch.Tensor) -> list[RSSMState]:
        """Posterior rollout for training. embeds/actions: [batch, time, ...].

        is_first[:, t] resets the RSSM state to its initial value before
        stepping, so sequences that begin partway through a batch chunk
        (episode boundaries) don't carry over state from a previous episode.
        """
        batch, time = embeds.shape[0], embeds.shape[1]
        device = embeds.device
        state = self.initial_state(batch, device)
        states = []
        prev_action = torch.zeros(batch, actions.shape[-1], device=device)
        for t in range(time):
            reset = is_first[:, t].view(batch, 1)
            init = self.initial_state(batch, device)
            state = RSSMState(
                h=torch.where(reset.bool(), init.h, state.h),
                z=torch.where(reset.bool(), init.z, state.z),
                prior_logits=state.prior_logits,
                posterior_logits=state.posterior_logits,
            )
            action_t = torch.where(reset.bool(), torch.zeros_like(prev_action), prev_action)
            state = self.observe_step(state, action_t, embeds[:, t])
            states.append(state)
            prev_action = actions[:, t]
        return states

    def imagine(self, initial_state: RSSMState, actions: torch.Tensor) -> list[RSSMState]:
        """Open-loop rollout for evaluation. actions: [batch, time, action_dim].

        Consumes only the initial latent and future actions -- no
        observation embeddings -- so it cannot leak future true angles.
        """
        time = actions.shape[1]
        state = initial_state
        states = []
        for t in range(time):
            state = self.imagine_step(state, actions[:, t])
            states.append(state)
        return states
