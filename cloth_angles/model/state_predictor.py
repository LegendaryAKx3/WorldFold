"""Deterministic residual predictor over a flat continuous state vector.

    next_state = state + delta_mean + delta_scale * net(norm(state), norm(prev_state), action)

Used for cloth vertex positions with or without robot state. Inputs are
normalized by training-set mean and standard deviation per dimension, targets
are the per-dimension normalized change. The last layer is zero-initialized so
training starts from persistence. Rollouts feed predictions back and never
consume future observations; the robot part of the state is predicted too.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class ResidualStatePredictor(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256, loss: str = "l1"):
        super().__init__()
        if loss not in ("mse", "l1"):
            raise ValueError(f"unknown loss {loss!r}")
        self.loss_kind = loss
        self.net = nn.Sequential(
            nn.Linear(2 * state_dim + action_dim, hidden_dim), nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ELU(),
            nn.Linear(hidden_dim, state_dim),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        for name in ("state_mean", "delta_mean"):
            self.register_buffer(name, torch.zeros(state_dim))
        for name in ("state_scale", "delta_scale"):
            self.register_buffer(name, torch.ones(state_dim))

    @torch.no_grad()
    def fit_normalizer(self, states, next_states, min_scale: float = 1e-4) -> None:
        states, next_states = torch.as_tensor(states), torch.as_tensor(next_states)
        delta = next_states - states
        self.state_mean.copy_(states.mean(0))
        self.state_scale.copy_(states.std(0).clamp(min=min_scale))
        self.delta_mean.copy_(delta.mean(0))
        self.delta_scale.copy_(delta.std(0).clamp(min=min_scale))

    def _features(self, state, prev_state, action):
        return torch.cat([(state - self.state_mean) / self.state_scale,
                          (prev_state - self.state_mean) / self.state_scale, action], -1)

    def forward(self, state, prev_state, action):
        """state/prev_state: [B, S]; action: [B, A] -> next state [B, S]."""
        return state + self.delta_mean + self.delta_scale * self.net(self._features(state, prev_state, action))

    def loss(self, state, prev_state, action, next_state):
        target = (next_state - state - self.delta_mean) / self.delta_scale
        criterion = F.mse_loss if self.loss_kind == "mse" else F.l1_loss
        return criterion(self.net(self._features(state, prev_state, action)), target)

    @torch.no_grad()
    def rollout(self, state, prev_state, actions):
        """actions: [B, T, A] -> predicted states [B, T, S], fed back autoregressively."""
        predictions = []
        for t in range(actions.shape[1]):
            nxt = self.forward(state, prev_state, actions[:, t])
            predictions.append(nxt)
            prev_state, state = state, nxt
        return torch.stack(predictions, 1)
