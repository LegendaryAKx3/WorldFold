"""Flattened angle field -> embedding. MLP-only, per spec, for small N."""

from __future__ import annotations

import torch
from torch import nn


class Encoder(nn.Module):
    def __init__(self, grid_size: int, hidden_dims: tuple[int, ...] = (128, 64)):
        super().__init__()
        in_dim = grid_size * grid_size
        dims = (in_dim,) + hidden_dims
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.ELU())
        self.net = nn.Sequential(*layers)
        self.embed_dim = hidden_dims[-1]

    def forward(self, obs_flat: torch.Tensor) -> torch.Tensor:
        """obs_flat: [..., N*N] -> [..., embed_dim]"""
        return self.net(obs_flat)
