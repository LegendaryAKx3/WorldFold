"""Latent -> [N, N] angle field. MLP-only, per spec, for small N."""

from __future__ import annotations

import math

import torch
from torch import nn


class Decoder(nn.Module):
    def __init__(self, latent_dim: int, grid_size: int, hidden_dim: int = 128,
                 angle_convention: str = "signed"):
        super().__init__()
        if angle_convention not in ("signed", "unsigned"):
            raise ValueError(f"unknown angle_convention {angle_convention!r}")
        self.grid_size = grid_size
        self.angle_convention = angle_convention
        out_dim = grid_size * grid_size
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        """latent: [..., latent_dim] -> [..., N, N] predicted angle field.

        Applies the output range transform for bounded angles: pi * tanh(raw)
        for signed angles in [-pi, pi], or pi/2 * (tanh(raw) + 1) for unsigned
        angles in [0, pi]. Both keep the decoder differentiable everywhere
        (no hard clipping) while enforcing the physical range.
        """
        raw = self.net(latent)
        if self.angle_convention == "signed":
            angle = math.pi * torch.tanh(raw)
        else:
            angle = (math.pi / 2.0) * (torch.tanh(raw) + 1.0)
        shape = angle.shape[:-1] + (self.grid_size, self.grid_size)
        return angle.reshape(shape)
