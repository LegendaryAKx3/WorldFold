"""Analytic fold reward mirrors SingleCornerFoldEnv's rules."""

import torch

from cloth_angles.model.fold_reward import EE_SLICE, GRASP_INDEX, MOVING_CORNER, FoldReward
from cloth_fold_rl.fold_env import CTRL_COST, DRAG_LIMIT, GRASP_BONUS, SUCCESS_BONUS, W_CARRY, W_REACH


def flat_state(corner, ee, grasp, anchor0=(0.0, 0.0, 0.0)):
    state = torch.zeros(378)
    state[0:3] = torch.tensor(anchor0)
    state[3 * MOVING_CORNER:3 * MOVING_CORNER + 3] = torch.tensor(corner)
    state[EE_SLICE] = torch.tensor(ee)
    state[GRASP_INDEX] = grasp
    return state


def spec():
    # cloth_0 and cloth_110 both sit at the origin in these synthetic states
    anchors0 = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.3, 0.0], [0.0, 0.0, 0.0], [0.3, 0.3, 0.0]])
    return FoldReward(goal=[0.3, 0.3, 0.0], anchors0=anchors0)


def test_potential_switches_regime_on_grasp():
    r = spec()
    free = flat_state((0.0, 0.3, 0.0), (0.1, 0.3, 0.0), 0.0)
    held = flat_state((0.0, 0.3, 0.0), (0.0, 0.3, 0.0), 1.0)
    torch.testing.assert_close(r.potential(free[None]), torch.tensor([-W_REACH * 0.1 - W_CARRY * 0.3]))
    torch.testing.assert_close(r.potential(held[None]), torch.tensor([GRASP_BONUS - W_CARRY * 0.3]))


def test_shaping_is_potential_difference_minus_control_cost():
    r = spec()
    s0 = flat_state((0.0, 0.3, 0.0), (0.0, 0.3, 0.0), 1.0)
    s1 = flat_state((0.1, 0.3, 0.0), (0.1, 0.3, 0.0), 1.0)
    action = torch.ones(1, 6)
    expected = W_CARRY * 0.1 - CTRL_COST * 6
    torch.testing.assert_close(r.shaping(s0[None], s1[None], action), torch.tensor([expected]))


def test_success_bonus_after_hold_and_drag_penalty():
    r = spec()
    placed = flat_state((0.3, 0.3, 0.0), (0.3, 0.3, 0.0), 1.0)
    states = placed.expand(12, -1)[None]
    rewards, terminated = r.rollout(states, torch.zeros(1, 11, 6))
    assert terminated[0, 9] and not terminated[0, 8]
    assert rewards[0, 9].item() == SUCCESS_BONUS and rewards[0, 8].item() == 0.0
    dragged = flat_state((0.3, 0.3, 0.0), (0.3, 0.3, 0.0), 1.0, anchor0=(DRAG_LIMIT + 0.01, 0.0, 0.0))
    rewards, terminated = r.rollout(torch.stack([placed, dragged])[None], torch.zeros(1, 1, 6))
    assert terminated[0, 0] and rewards[0, 0].item() == -5.0


def test_soft_potential_matches_hard_on_real_flags_and_blends_between():
    hard, soft = spec(), spec()
    soft.soft = True
    for flag in (0.0, 1.0):
        state = flat_state((0.1, 0.3, 0.0), (0.2, 0.3, 0.0), flag)[None]
        torch.testing.assert_close(soft.potential(state), hard.potential(state))
    half = flat_state((0.1, 0.3, 0.0), (0.2, 0.3, 0.0), 0.5)[None]
    lo = hard.potential(flat_state((0.1, 0.3, 0.0), (0.2, 0.3, 0.0), 0.0)[None])
    hi = hard.potential(flat_state((0.1, 0.3, 0.0), (0.2, 0.3, 0.0), 1.0)[None])
    torch.testing.assert_close(soft.potential(half), 0.5 * (lo + hi))


def test_gripper_rule_hysteresis_and_weld():
    from cloth_angles.model.fold_reward import CTRL_INDEX, gripper_transition
    near = flat_state((0.0, 0.3, 0.0), (0.0, 0.32, 0.0), 0.0)     # 2 cm away, open
    far = flat_state((0.0, 0.3, 0.0), (0.0, 0.40, 0.0), 0.0)      # 10 cm away, open
    held = flat_state((0.0, 0.3, 0.0), (0.0, 0.3, 0.0), 1.0)
    held[CTRL_INDEX] = -0.1
    close, open_, hold = torch.zeros(6), torch.zeros(6), torch.zeros(6)
    close[5], open_[5], hold[5] = -1.0, 1.0, 0.1
    assert gripper_transition(near[None], close[None]) == (1.0, 1.0)
    assert gripper_transition(far[None], close[None]) == (1.0, 0.0)
    assert gripper_transition(held[None], hold[None]) == (1.0, 1.0)
    assert gripper_transition(held[None], open_[None]) == (0.0, 0.0)
    for state, action in ((near, close), (far, close), (held, hold), (held, open_)):
        hard = gripper_transition(state[None], action[None])
        soft = gripper_transition(state[None], action[None], soft=True)
        torch.testing.assert_close(torch.stack(soft), torch.stack(hard))
    ramp = near.clone()
    ramp[EE_SLICE] = torch.tensor([0.0, 0.335, 0.0])            # 3.5 cm: hard misses, soft partial
    assert gripper_transition(ramp[None], close[None])[1] == 0.0
    assert 0.0 < gripper_transition(ramp[None], close[None], soft=True)[1] < 1.0
