"""Ties encoder + RSSM + decoder together; defines loss() and train_step().

L = L_angle + beta * L_KL

L_angle uses Huber loss on the wrapped residual for signed circular angles,
or the ordinary difference for an unsigned angle-to-reference field. L_KL is
the balanced (stop-gradient) KL between posterior and prior, with free bits.
Every loss term is masked: padded episode tails (mask == 0) never contribute.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from cloth_angles.model.decoder import Decoder
from cloth_angles.model.encoder import Encoder
from cloth_angles.model.rssm import RSSM, RSSMState


@dataclass
class LossOutput:
    loss: torch.Tensor
    angle_loss: torch.Tensor
    kl_loss: torch.Tensor
    mae_radians: torch.Tensor


def _wrapped_residual(angle_hat: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
    """Shortest-path angular residual, for signed circular angles."""
    return torch.atan2(torch.sin(angle_hat - angle), torch.cos(angle_hat - angle))


def _kl_categorical(posterior_logits: torch.Tensor, prior_logits: torch.Tensor) -> torch.Tensor:
    """KL(posterior || prior) per categorical, summed over categoricals. [batch, n_cat, n_cls] -> [batch]"""
    posterior = torch.distributions.Categorical(logits=posterior_logits)
    prior = torch.distributions.Categorical(logits=prior_logits)
    kl = torch.distributions.kl_divergence(posterior, prior)  # [batch, n_categoricals]
    return kl.sum(dim=-1)


class WorldModel(nn.Module):
    def __init__(self, grid_size: int, action_dim: int, angle_convention: str = "signed",
                 encoder_hidden: tuple[int, ...] = (128, 64), h_dim: int = 128,
                 n_categoricals: int = 8, n_classes: int = 8, mlp_hidden: int = 128,
                 kl_free_bits: float = 1.0, kl_weight: float = 1.0, huber_delta: float = 1.0):
        super().__init__()
        self.grid_size = grid_size
        self.angle_convention = angle_convention
        self.kl_free_bits = kl_free_bits
        self.kl_weight = kl_weight
        self.huber_delta = huber_delta

        self.encoder = Encoder(grid_size, hidden_dims=encoder_hidden)
        self.rssm = RSSM(embed_dim=self.encoder.embed_dim, action_dim=action_dim, h_dim=h_dim,
                          n_categoricals=n_categoricals, n_classes=n_classes, hidden_dim=mlp_hidden)
        self.decoder = Decoder(latent_dim=h_dim + self.rssm.z_dim, grid_size=grid_size,
                                hidden_dim=mlp_hidden, angle_convention=angle_convention)

    def _angle_error(self, angle_hat: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
        if self.angle_convention == "signed":
            return _wrapped_residual(angle_hat, angle)
        return angle_hat - angle

    def loss(self, obs: torch.Tensor, actions: torch.Tensor, next_obs: torch.Tensor,
              mask: torch.Tensor, is_first: torch.Tensor) -> LossOutput:
        """obs/next_obs: [batch, time, N*N] flattened angle fields.
        actions: [batch, time, action_dim]. mask/is_first: [batch, time].
        """
        batch, time, n2 = obs.shape
        n = self.grid_size

        embeds = self.encoder(obs)
        states = self.rssm.observe(embeds, actions, is_first)

        h = torch.stack([s.h for s in states], dim=1)
        z = torch.stack([s.z for s in states], dim=1)
        posterior_logits = torch.stack([s.posterior_logits for s in states], dim=1)
        prior_logits = torch.stack([s.prior_logits for s in states], dim=1)

        feature = torch.cat([h, z], dim=-1)
        angle_hat = self.decoder(feature)  # [batch, time, N, N]

        target = next_obs.reshape(batch, time, n, n)
        residual = self._angle_error(angle_hat, target)
        huber = F.huber_loss(residual, torch.zeros_like(residual), delta=self.huber_delta, reduction="none")
        huber = huber.mean(dim=(-2, -1))  # [batch, time]

        kl = _kl_categorical(posterior_logits.reshape(batch, time, *posterior_logits.shape[2:]),
                              prior_logits.reshape(batch, time, *prior_logits.shape[2:]))
        kl = torch.clamp(kl, min=self.kl_free_bits)

        mask_sum = mask.sum().clamp(min=1.0)
        angle_loss = (huber * mask).sum() / mask_sum
        kl_loss = (kl * mask).sum() / mask_sum

        total = angle_loss + self.kl_weight * kl_loss

        with torch.no_grad():
            mae = (residual.abs().mean(dim=(-2, -1)) * mask).sum() / mask_sum

        return LossOutput(loss=total, angle_loss=angle_loss, kl_loss=kl_loss, mae_radians=mae)

    def train_step(self, optimizer: torch.optim.Optimizer, obs, actions, next_obs, mask, is_first,
                    grad_clip: float = 100.0) -> LossOutput:
        optimizer.zero_grad(set_to_none=True)
        output = self.loss(obs, actions, next_obs, mask, is_first)
        output.loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), grad_clip)
        optimizer.step()
        return output

    @torch.no_grad()
    def initial_state_from_obs(self, obs_t: torch.Tensor) -> RSSMState:
        """Bootstrap an RSSM state for open-loop imagination from one real observation."""
        batch = obs_t.shape[0]
        device = obs_t.device
        embed = self.encoder(obs_t)
        state = self.rssm.initial_state(batch, device)
        zero_action = torch.zeros(batch, self.rssm.gru_input.in_features - self.rssm.z_dim, device=device)
        return self.rssm.observe_step(state, zero_action, embed)

    @torch.no_grad()
    def imagine_angles(self, initial_state: RSSMState, actions: torch.Tensor) -> torch.Tensor:
        """actions: [batch, time, action_dim] -> predicted angle fields [batch, time, N, N]."""
        states = self.rssm.imagine(initial_state, actions)
        h = torch.stack([s.h for s in states], dim=1)
        z = torch.stack([s.z for s in states], dim=1)
        feature = torch.cat([h, z], dim=-1)
        return self.decoder(feature)
