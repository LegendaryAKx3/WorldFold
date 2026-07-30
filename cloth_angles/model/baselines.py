"""Evaluation baselines the RSSM must beat: persistence and an
action-agnostic next-field predictor. Both operate on flattened [.., N*N]
angle fields and are intentionally simple, non-recurrent references.
"""

from __future__ import annotations

import torch
from torch import nn


def persistence_predict(obs_flat: torch.Tensor) -> torch.Tensor:
    """angle_hat_{t+1} = angle_t, i.e. predict no change at all."""
    return obs_flat


class ActionAgnosticPredictor(nn.Module):
    """Predicts the next angle field from the current field alone, ignoring
    the action. A small MLP; trained the same way as the RSSM's angle loss
    but with no recurrent state and no action conditioning, so it isolates
    how much of the dynamics are action-independent (e.g. passive settling).
    """

    def __init__(self, grid_size: int, hidden_dim: int = 128):
        super().__init__()
        n2 = grid_size * grid_size
        self.grid_size = grid_size
        self.net = nn.Sequential(
            nn.Linear(n2, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, n2),
        )

    def forward(self, obs_flat: torch.Tensor) -> torch.Tensor:
        """obs_flat: [..., N*N] -> [..., N*N] predicted next field."""
        return self.net(obs_flat)
