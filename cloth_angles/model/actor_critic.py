"""Actor-critic trained on imagined rollouts of a state-space world model.

Dreamer-style, without a latent: the world model is a ResidualStatePredictor
over physical state, rewards, terminations and task stages come from
FoldReward, and the actor is trained by backpropagating lambda-returns
through the model's dynamics. The critic regresses to the same returns on
detached states. Returns bootstrap from a slow EMA copy of the critic, and
the value loss is computed in return-scaled units, both to keep the critic
from feeding on its own estimates.
"""

from __future__ import annotations

import copy
import math

import torch
import torch.nn.functional as F
from torch import nn

from cloth_angles.model.fold_reward import FoldReward, gripper_transition
from cloth_angles.tasks import SINGLE, Task

GRIPPER_CLOSED, GRIPPER_OPEN = -0.1, 1.0


def policy_features(state, goal, state_mean, state_scale, task: Task = SINGLE, stage=None):
    """Normalized state, the goal vector, each arm's carried-corner offset to its
    goal in the current stage (zero for an idle arm), and a stage one-hot for
    tasks with more than one stage. stage: long [...] matching state[..., 0]."""
    lead = state.shape[:-1]
    goal = goal.expand(*lead, task.goal_dim)
    goals = goal.reshape(*lead, -1, 3)
    if stage is None:
        stage = torch.zeros(lead, dtype=torch.long, device=state.device)
    offsets = []
    for s, st in enumerate(task.stages):
        per_arm = [torch.zeros(*lead, 3, device=state.device) for _ in task.arms]
        for m in st.moves:
            corner = torch.stack([state[..., 3 * c:3 * c + 3] for c in m.corners]).mean(0)
            per_arm[m.arm] = (corner - goals[..., m.goal, :]) * 10.0
        offsets.append(torch.cat(per_arm, -1))
    offsets = torch.stack(offsets, -2).gather(-2, stage[..., None, None].expand(*lead, 1, 3 * len(task.arms))).squeeze(-2)
    parts = [(state - state_mean) / state_scale, goal, offsets]
    if len(task.stages) > 1:
        parts.append(F.one_hot(stage, len(task.stages)).float())
    return torch.cat(parts, -1)


def feature_dim(task: Task = SINGLE) -> int:
    n_stages = len(task.stages)
    return task.state_dim + task.goal_dim + 3 * len(task.arms) + (n_stages if n_stages > 1 else 0)


class Actor(nn.Module):
    def __init__(self, input_dim: int, action_dim: int, hidden_dim: int = 256, init_std: float = 0.3,
                 max_std: float = 0.5):
        super().__init__()
        self.max_std = max_std
        self.net = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ELU(),
                                 nn.Linear(hidden_dim, hidden_dim), nn.ELU(),
                                 nn.Linear(hidden_dim, action_dim))
        self.log_std = nn.Parameter(torch.full((action_dim,), math.log(init_std)))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Deterministic action in [-1, 1]."""
        return torch.tanh(self.net(features))

    def sample(self, features: torch.Tensor):
        """Reparameterized tanh-squashed Gaussian sample and the Gaussian entropy."""
        mean = self.net(features)
        std = self.log_std.exp().clamp(1e-3, self.max_std)
        pre = mean + std * torch.randn_like(mean)
        entropy = (0.5 * math.log(2 * math.pi * math.e) + std.log()).sum()
        return torch.tanh(pre), entropy


class Critic(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ELU(),
                                 nn.Linear(hidden_dim, hidden_dim), nn.ELU(),
                                 nn.Linear(hidden_dim, 1))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features).squeeze(-1)


def lambda_returns(rewards: torch.Tensor, values: torch.Tensor, continues: torch.Tensor,
                   gamma: float, lam: float) -> torch.Tensor:
    """rewards/continues: [B, T]; values: [B, T+1] (bootstrap at T) -> returns [B, T]."""
    returns = []
    nxt = values[:, -1]
    for t in reversed(range(rewards.shape[1])):
        nxt = rewards[:, t] + gamma * continues[:, t] * ((1 - lam) * values[:, t + 1] + lam * nxt)
        returns.append(nxt)
    return torch.stack(returns[::-1], 1)


class ImaginationTrainer:
    def __init__(self, world_model, actor: Actor, critic: Critic, task: Task = SINGLE, horizon: int = 15,
                 gamma: float = 0.98, lam: float = 0.95, entropy_coef: float = 1e-4,
                 lr: float = 3e-4, target_tau: float = 0.02, soft_reward: bool = True,
                 ood_limit: float = 8.0, bc_coef: float = 0.0, ensemble=None,
                 disagreement_coef: float = 0.0):
        """world_model: the model used for normalization and, without an ensemble, for
        stepping. ensemble: optional list of models; imagination then steps with
        their mean prediction and subtracts disagreement_coef times the members'
        spread (in normalized state units) from every imagined reward.
        """
        self.world_model, self.task = world_model, task
        self.ensemble = list(ensemble) if ensemble else [world_model]
        self.disagreement_coef = disagreement_coef
        for model in {id(m): m for m in self.ensemble + [world_model]}.values():
            for p in model.parameters():
                p.requires_grad_(False)
        self.actor, self.critic = actor, critic
        self.target_critic = copy.deepcopy(critic)
        for p in self.target_critic.parameters():
            p.requires_grad_(False)
        self.target_tau = target_tau
        self.horizon, self.gamma, self.lam, self.entropy_coef = horizon, gamma, lam, entropy_coef
        self.actor_opt = torch.optim.Adam(actor.parameters(), lr=lr)
        self.critic_opt = torch.optim.Adam(critic.parameters(), lr=lr)
        self.return_scale = 1.0
        self.soft_reward = soft_reward
        self.ood_limit = ood_limit
        self.bc_coef = bc_coef

    def features(self, state, goal, stage=None):
        return policy_features(state, goal, self.world_model.state_mean, self.world_model.state_scale, self.task, stage)

    def step(self, state, prev_state, action):
        """Learned continuous dynamics, with each gripper and grasp set by the exact rule.

        The learned model never reproduces grasp and release (about one
        transition per episode, which the L1 loss treats as outliers), so
        those dimensions come from gripper_transition instead; the soft
        version keeps them differentiable for the actor.
        """
        predictions = torch.stack([m(state, prev_state, action) for m in self.ensemble])
        nxt = predictions.mean(0)
        if len(self.ensemble) > 1:
            spread = (predictions / self.world_model.state_scale).std(0)
            self._disagreement.append(spread.pow(2).mean(-1).sqrt())
        mask = torch.zeros_like(nxt)
        replacement = torch.zeros_like(nxt)
        for arm in self.task.arms:
            closed, grasp = gripper_transition(state, action, arm, self.task.grasp_radius, soft=self.soft_reward)
            mask[..., arm.ctrl_index] = 1.0
            mask[..., arm.grasp_index] = 1.0
            replacement[..., arm.ctrl_index] = GRIPPER_OPEN + closed * (GRIPPER_CLOSED - GRIPPER_OPEN)
            replacement[..., arm.grasp_index] = grasp
        return nxt * (1.0 - mask) + replacement * mask

    def imagine(self, state, prev_state, spec, rs):
        """Roll the actor through the model, scoring each step with the reward machine."""
        states, actions, rewards, terminated, stages = [state], [], [], [], [rs["stage"]]
        self._disagreement = []
        for _ in range(self.horizon):
            action, entropy = self.actor.sample(self.features(state, spec.flat_goal, rs["stage"]))
            nxt = self.step(state, prev_state, action)
            reward, term, rs = spec.step(state, nxt, action, rs)
            prev_state, state = state, nxt
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            terminated.append(term)
            stages.append(rs["stage"])
        return (torch.stack(states, 1), torch.stack(actions, 1), torch.stack(rewards, 1),
                torch.stack(terminated, 1), torch.stack(stages, 1), entropy)

    def update(self, state, prev_state, goal, anchors0, stage=None, anchor=None) -> dict:
        """One actor and one critic update from a batch of real start states.

        stage: long [B], the task stage at each start state (default 0).
        Imagined steps whose normalized features leave [-ood_limit, ood_limit]
        are treated as terminal with zero value, so the actor cannot profit
        from states the model was never trained on. With bc_coef > 0 and
        anchor=(states, goals, stages, actions), the actor is also pulled
        toward those actions on those states.
        """
        spec = FoldReward(goal, anchors0, self.task, soft=self.soft_reward)
        spec.flat_goal = goal
        rs = spec.init(torch.zeros(len(state), dtype=torch.long) if stage is None else stage)
        states, actions, rewards, terminated, stages, entropy = self.imagine(state, prev_state, spec, rs)
        disagreement = torch.stack(self._disagreement, 1) if self._disagreement else torch.zeros_like(rewards)
        rewards = rewards - self.disagreement_coef * disagreement
        features = self.features(states, goal[:, None], stages)
        out_of_distribution = features[:, 1:].detach().abs().amax(-1) > self.ood_limit
        terminated = terminated | out_of_distribution
        continues = 1.0 - terminated.float()
        alive = torch.cat([torch.ones_like(continues[:, :1]), torch.cumprod(continues, 1)[:, :-1]], 1)
        values = self.target_critic(features)
        returns = lambda_returns(rewards, values, continues, self.gamma, self.lam)

        with torch.no_grad():
            spread = torch.quantile(returns, 0.95) - torch.quantile(returns, 0.05)
            self.return_scale = 0.99 * self.return_scale + 0.01 * max(1.0, spread.item())
        actor_loss = -(alive * returns).sum() / alive.sum() / self.return_scale - self.entropy_coef * entropy
        if self.bc_coef > 0 and anchor is not None:
            anchor_state, anchor_goal, anchor_stage, anchor_action = anchor
            actor_loss = actor_loss + self.bc_coef * F.mse_loss(
                self.actor(self.features(anchor_state, anchor_goal, anchor_stage)), anchor_action)
        self.actor_opt.zero_grad(set_to_none=True)
        self.critic_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), 100.0)
        self.actor_opt.step()

        target = returns.detach() / self.return_scale
        predicted = self.critic(features[:, :-1].detach()) / self.return_scale
        critic_loss = (alive * F.mse_loss(predicted, target, reduction="none")).sum() / alive.sum()
        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), 100.0)
        self.critic_opt.step()
        with torch.no_grad():
            for p, tp in zip(self.critic.parameters(), self.target_critic.parameters()):
                tp.lerp_(p, self.target_tau)

        grasp = torch.stack([spec.grasped(states[:, -1], a) for a in self.task.arms], -1).any(-1)
        return {"actor_loss": actor_loss.item(), "critic_loss": critic_loss.item(),
                "mean_return": returns[:, 0].mean().item(), "mean_reward": rewards.mean().item(),
                "imagined_grasp_rate": grasp.float().mean().item(),
                "imagined_success_rate": (terminated & (rewards > 10)).any(1).float().mean().item(),
                "imagined_stage_advance": (stages[:, -1] > stages[:, 0]).float().mean().item(),
                "max_feature": features.detach().abs().max().item(),
                "ood_rate": out_of_distribution.any(1).float().mean().item(),
                "disagreement": disagreement.mean().item(),
                "entropy": entropy.item()}
