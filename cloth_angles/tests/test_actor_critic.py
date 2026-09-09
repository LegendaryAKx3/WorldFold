"""Lambda returns, squashed actor, and one imagination update on a tiny model."""

import torch

from cloth_angles.model.actor_critic import Actor, Critic, ImaginationTrainer, feature_dim, lambda_returns
from cloth_angles.model.state_predictor import ResidualStatePredictor


def test_lambda_returns_reduce_to_discounted_sum_when_lambda_is_one():
    rewards = torch.tensor([[1.0, 2.0, 3.0]])
    values = torch.tensor([[9.0, 9.0, 9.0, 4.0]])
    continues = torch.ones(1, 3)
    returns = lambda_returns(rewards, values, continues, gamma=0.5, lam=1.0)
    expected = torch.tensor([[1 + 0.5 * (2 + 0.5 * (3 + 0.5 * 4)), 2 + 0.5 * (3 + 0.5 * 4), 3 + 0.5 * 4]])
    torch.testing.assert_close(returns, expected)


def test_lambda_returns_stop_at_termination():
    rewards = torch.tensor([[1.0, 5.0]])
    values = torch.tensor([[0.0, 7.0, 7.0]])
    continues = torch.tensor([[0.0, 1.0]])
    returns = lambda_returns(rewards, values, continues, gamma=0.9, lam=0.8)
    assert returns[0, 0].item() == 1.0


def test_actor_outputs_stay_in_action_range():
    actor = Actor(input_dim=4, action_dim=2)
    features = torch.randn(16, 4) * 50
    assert actor(features).abs().max() <= 1.0
    sample, entropy = actor.sample(features)
    assert sample.abs().max() <= 1.0 and torch.isfinite(entropy)


def test_imagination_update_changes_actor_and_returns_metrics():
    torch.manual_seed(0)
    world_model = ResidualStatePredictor(378, 6, hidden_dim=16)
    world_model.fit_normalizer(torch.randn(20, 378) * 0.1, torch.randn(20, 378) * 0.1)
    actor, critic = Actor(feature_dim(378), 6, hidden_dim=16), Critic(feature_dim(378), hidden_dim=16)
    trainer = ImaginationTrainer(world_model, actor, critic, horizon=3)
    before = [p.clone() for p in actor.parameters()]
    state = torch.randn(4, 378) * 0.1
    goal = torch.tensor([[0.15, 0.1, 0.43]]).expand(4, 3)
    anchors0 = torch.zeros(4, 4, 3)
    metrics = trainer.update(state, state, goal, anchors0)
    assert set(metrics) >= {"actor_loss", "critic_loss", "mean_return", "imagined_grasp_rate"}
    assert any(not torch.equal(a, b) for a, b in zip(before, actor.parameters()))
    assert all(not p.requires_grad for p in world_model.parameters())


def test_ensemble_disagreement_penalizes_reward():
    torch.manual_seed(0)
    members = []
    for k in range(3):
        m = ResidualStatePredictor(378, 6, hidden_dim=16)
        m.fit_normalizer(torch.randn(20, 378) * 0.1, torch.randn(20, 378) * 0.1)
        with torch.no_grad():
            m.net[-1].weight.normal_(std=0.1 * (k + 1))
        members.append(m)
    actor, critic = Actor(feature_dim(378), 6, hidden_dim=16), Critic(feature_dim(378), hidden_dim=16)
    state = torch.randn(4, 378) * 0.1
    goal = torch.tensor([[0.15, 0.1, 0.43]]).expand(4, 3)
    anchors0 = torch.zeros(4, 4, 3)
    torch.manual_seed(1)
    plain = ImaginationTrainer(members[0], actor, critic, horizon=3, ensemble=members, disagreement_coef=0.0)
    plain_metrics = plain.update(state, state, goal, anchors0)
    torch.manual_seed(1)
    penalized = ImaginationTrainer(members[0], actor, critic, horizon=3, ensemble=members, disagreement_coef=5.0)
    penalized_metrics = penalized.update(state, state, goal, anchors0)
    assert plain_metrics["disagreement"] > 0
    assert penalized_metrics["mean_reward"] < plain_metrics["mean_reward"]
