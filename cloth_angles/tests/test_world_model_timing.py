"""Actions must cause the next state, in training and prior-only evaluation."""

import pytest
import torch

from cloth_angles.evaluate import one_step_eval, open_loop_eval
from cloth_angles.model.world_model import WorldModel


def make_model():
    torch.manual_seed(0)
    return WorldModel(grid_size=3, action_dim=2, encoder_hidden=(8,),
                      h_dim=8, n_categoricals=2, n_classes=2, mlp_hidden=8,
                      kl_free_bits=0.0)


def batch(time=4):
    torch.manual_seed(1)
    observations = torch.randn(1, time + 1, 9)
    actions = torch.randn(1, time, 2)
    first = torch.zeros(1, time, dtype=torch.bool)
    first[:, 0] = True
    return observations[:, :-1], actions, observations[:, 1:], torch.ones(1, time), first


@pytest.mark.parametrize("time", [1, 4])
def test_each_action_affects_its_own_next_state_loss(time):
    model = make_model()
    obs, actions, nxt, mask, first = batch(time)
    actions.requires_grad_()
    for t in range(time):
        step_mask = torch.zeros_like(mask)
        step_mask[:, t] = 1
        torch.manual_seed(22)
        loss = model.loss(obs, actions, nxt, step_mask, first).loss
        grad, = torch.autograd.grad(loss, actions, allow_unused=True)
        assert grad is not None, "The transition action is missing from the loss graph"
        assert grad[:, t].abs().sum() > 1e-8
        assert torch.count_nonzero(grad[:, t + 1:]) == 0


class FixedReplay:
    def __init__(self, tensors):
        self.arrays = tuple(t.detach().numpy() for t in tensors)

    def sample(self, n_sequences):
        return self.arrays


def test_one_step_eval_matches_one_step_imagination():
    model = make_model()
    replay = FixedReplay(batch(time=1))
    torch.manual_seed(22)
    one = one_step_eval(model, replay, "cpu", 1)
    torch.manual_seed(22)
    imagined, _, _, _ = open_loop_eval(model, replay, "cpu", [1], 1)
    assert one["rssm_mae_rad"] == pytest.approx(imagined["open_loop_1step_mae_rad"])


def test_masked_tail_does_not_change_training_loss():
    model = make_model()
    obs, actions, nxt, mask, first = batch()
    mask[:, 2:] = 0
    torch.manual_seed(22)
    original = model.loss(obs, actions, nxt, mask, first)
    changed_obs, changed_actions, changed_next = obs.clone(), actions.clone(), nxt.clone()
    changed_obs[:, 2:] = 10
    changed_actions[:, 2:] = -10
    changed_next[:, 2:] = 5
    torch.manual_seed(22)
    changed = model.loss(changed_obs, changed_actions, changed_next, mask, first)
    torch.testing.assert_close(original.loss, changed.loss)


def test_episode_reset_cuts_gradients_to_previous_episode():
    model = make_model()
    obs, actions, nxt, mask, first = batch()
    first[:, 2] = True
    mask[:, :2] = 0
    actions.requires_grad_()
    loss = model.loss(obs, actions, nxt, mask, first).loss
    grad, = torch.autograd.grad(loss, actions)
    assert torch.count_nonzero(grad[:, :2]) == 0
    assert torch.all(grad[:, 2:].abs().sum(-1) > 1e-8)


def test_one_step_predictions_do_not_see_future_observations():
    model = make_model()
    tensors = batch()
    decoded = []
    handle = model.decoder.register_forward_hook(
        lambda module, args, output: decoded.append(output.detach().clone()))
    try:
        torch.manual_seed(22)
        one_step_eval(model, FixedReplay(tensors), "cpu", 1)
        original_first_prediction = decoded[0]
        decoded.clear()
        obs, actions, nxt, mask, first = (t.clone() for t in tensors)
        obs[:, 1:] = 50
        nxt[:] = -50
        torch.manual_seed(22)
        one_step_eval(model, FixedReplay((obs, actions, nxt, mask, first)), "cpu", 1)
        torch.testing.assert_close(original_first_prediction, decoded[0])
    finally:
        handle.remove()
