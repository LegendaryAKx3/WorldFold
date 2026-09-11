"""Residual state predictor and full-state episode store."""

import json

import numpy as np
import pytest
import torch

from cloth_angles.data.state_episode import ROBOT_DIM, StateEpisode, StateEpisodeStore
from cloth_angles.model.state_predictor import ResidualStatePredictor


def make(loss="l1"):
    torch.manual_seed(0)
    model = ResidualStatePredictor(state_dim=6, action_dim=2, hidden_dim=8, loss=loss)
    rng = np.random.default_rng(0)
    states = rng.normal(0, 2, (50, 6)).astype(np.float32)
    nxt = (states + rng.normal(0.1, 0.05, states.shape)).astype(np.float32)
    model.fit_normalizer(states, nxt)
    return model


def test_starts_as_persistence_plus_mean_change():
    model = make()
    state = torch.rand(4, 6)
    torch.testing.assert_close(model(state, state, torch.zeros(4, 2)), state + model.delta_mean)


@pytest.mark.parametrize("loss", ["l1", "mse"])
def test_action_reaches_prediction_after_training_step(loss):
    model = make(loss)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    state, prev, action = torch.rand(8, 6), torch.rand(8, 6), torch.randn(8, 2)
    optimizer.zero_grad()
    model.loss(state, prev, action, state + 0.3).backward()
    optimizer.step()
    action.requires_grad_()
    grad, = torch.autograd.grad(model(state, prev, action).sum(), action)
    assert grad.abs().sum() > 0


def test_rollout_feeds_back_its_own_predictions():
    model = make()
    with torch.no_grad():
        model.net[-1].weight.normal_()
    state, prev = torch.rand(2, 6), torch.rand(2, 6)
    actions = torch.randn(2, 3, 2)
    rolled = model.rollout(state, prev, actions)
    step1 = model(state, prev, actions[:, 0])
    step2 = model(step1, state, actions[:, 1])
    torch.testing.assert_close(rolled[:, 0], step1)
    torch.testing.assert_close(rolled[:, 1], step2)


def test_episode_store_round_trip(tmp_path):
    rng = np.random.default_rng(0)
    grid = np.stack(np.meshgrid(np.arange(3.0), np.arange(3.0), indexing="ij"), -1).reshape(9, 2)
    vertices = np.concatenate([grid, np.zeros((9, 1))], 1)[None].repeat(4, 0).astype(np.float32)
    vertices[:, :, 2] += rng.normal(0, 0.01, (4, 9)).astype(np.float32)
    episode = StateEpisode(vertices, rng.normal(size=(4, ROBOT_DIM)).astype(np.float32),
                           rng.normal(size=(3, 2)).astype(np.float32), {"kind": "test", "seed": 1})
    store = StateEpisodeStore(tmp_path)
    store.append(episode)
    loaded = store.load_all()[0]
    assert len(loaded) == 3
    np.testing.assert_array_equal(loaded.vertices, vertices)
    assert loaded.metadata == json.loads(json.dumps(episode.metadata))
    assert loaded.states().shape == (4, 27 + ROBOT_DIM)


def test_episode_rejects_mismatched_lengths():
    episode = StateEpisode(np.zeros((3, 4, 3), np.float32), np.zeros((3, ROBOT_DIM), np.float32),
                           np.zeros((3, 2), np.float32))
    with pytest.raises(ValueError):
        episode.validate()
