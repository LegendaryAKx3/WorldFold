"""Analytic fold reward mirrors the fold envs' rules, for the single and quarter tasks."""

import torch

from cloth_angles.model.fold_reward import FoldReward, gripper_transition
from cloth_angles.tasks import LEFT, QUARTER, SINGLE
from cloth_fold_rl.fold_env import CTRL_COST, DRAG_LIMIT, GRASP_BONUS, SUCCESS_BONUS, W_CARRY, W_REACH
from cloth_fold_rl.quarter_fold_env import RELEASE_BONUS, STAGE_BONUS


def flat_state(corner, ee, grasp, anchor0=(0.0, 0.0, 0.0)):
    state = torch.zeros(SINGLE.state_dim)
    state[0:3] = torch.tensor(anchor0)
    state[30:33] = torch.tensor(corner)          # cloth_10
    state[LEFT.ee_slice] = torch.tensor(ee)
    state[LEFT.grasp_index] = grasp
    return state


def spec():
    # cloth_0 and cloth_110 both sit at the origin in these synthetic states
    anchors0 = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.3, 0.0], [0.0, 0.0, 0.0], [0.3, 0.3, 0.0]])
    return FoldReward(goal=[0.3, 0.3, 0.0], anchors0=anchors0)


def test_potential_switches_regime_on_grasp():
    r = spec()
    zero = torch.zeros(1, dtype=torch.long)
    free = flat_state((0.0, 0.3, 0.0), (0.1, 0.3, 0.0), 0.0)
    held = flat_state((0.0, 0.3, 0.0), (0.0, 0.3, 0.0), 1.0)
    torch.testing.assert_close(r.potential(free[None], zero), torch.tensor([-W_REACH * 0.1 - W_CARRY * 0.3]))
    torch.testing.assert_close(r.potential(held[None], zero), torch.tensor([GRASP_BONUS - W_CARRY * 0.3]))


def test_step_reward_is_potential_difference_minus_control_cost():
    r = spec()
    s0 = flat_state((0.0, 0.3, 0.0), (0.0, 0.3, 0.0), 1.0)
    s1 = flat_state((0.1, 0.3, 0.0), (0.1, 0.3, 0.0), 1.0)
    reward, terminated, _ = r.step(s0[None], s1[None], torch.ones(1, 6), r.init([0]))
    torch.testing.assert_close(reward, torch.tensor([W_CARRY * 0.1 - CTRL_COST * 6]))
    assert not terminated.item()


def test_success_bonus_after_hold_and_drag_penalty():
    r = spec()
    placed = flat_state((0.3, 0.3, 0.0), (0.3, 0.3, 0.0), 1.0)
    states = placed.expand(12, -1)[None]
    rewards, terminated, stages = r.rollout(states, torch.zeros(1, 11, 6))
    assert terminated[0, 9] and not terminated[0, 8]
    assert rewards[0, 9].item() == SUCCESS_BONUS and rewards[0, 8].item() == 0.0
    assert (stages == 0).all()
    dragged = flat_state((0.3, 0.3, 0.0), (0.3, 0.3, 0.0), 1.0, anchor0=(DRAG_LIMIT + 0.01, 0.0, 0.0))
    rewards, terminated, _ = r.rollout(torch.stack([placed, dragged])[None], torch.zeros(1, 1, 6))
    assert terminated[0, 0] and rewards[0, 0].item() == -5.0


def test_soft_potential_matches_hard_on_real_flags_and_blends_between():
    hard, soft = spec(), spec()
    soft.soft = True
    zero = torch.zeros(1, dtype=torch.long)
    for flag in (0.0, 1.0):
        state = flat_state((0.1, 0.3, 0.0), (0.2, 0.3, 0.0), flag)[None]
        torch.testing.assert_close(soft.potential(state, zero), hard.potential(state, zero))
    half = flat_state((0.1, 0.3, 0.0), (0.2, 0.3, 0.0), 0.5)[None]
    lo = hard.potential(flat_state((0.1, 0.3, 0.0), (0.2, 0.3, 0.0), 0.0)[None], zero)
    hi = hard.potential(flat_state((0.1, 0.3, 0.0), (0.2, 0.3, 0.0), 1.0)[None], zero)
    torch.testing.assert_close(soft.potential(half, zero), 0.5 * (lo + hi))


def test_gripper_rule_hysteresis_and_weld():
    near = flat_state((0.0, 0.3, 0.0), (0.0, 0.32, 0.0), 0.0)     # 2 cm away, open
    far = flat_state((0.0, 0.3, 0.0), (0.0, 0.40, 0.0), 0.0)      # 10 cm away, open
    held = flat_state((0.0, 0.3, 0.0), (0.0, 0.3, 0.0), 1.0)
    held[LEFT.ctrl_index] = -0.1
    close, open_, hold = torch.zeros(6), torch.zeros(6), torch.zeros(6)
    close[5], open_[5], hold[5] = -1.0, 1.0, 0.1
    rule = lambda s, a, soft=False: gripper_transition(s[None], a[None], LEFT, SINGLE.grasp_radius, soft)
    assert rule(near, close) == (1.0, 1.0)
    assert rule(far, close) == (1.0, 0.0)
    assert rule(held, hold) == (1.0, 1.0)
    assert rule(held, open_) == (0.0, 0.0)
    for state, action in ((near, close), (far, close), (held, hold), (held, open_)):
        torch.testing.assert_close(torch.stack(rule(state, action, True)), torch.stack(rule(state, action)))
    ramp = near.clone()
    ramp[LEFT.ee_slice] = torch.tensor([0.0, 0.335, 0.0])            # 3.5 cm: hard misses, soft partial
    assert rule(ramp, close)[1] == 0.0
    assert 0.0 < rule(ramp, close, True)[1] < 1.0


def quarter_state(c0, c10, c110, c120, left_grasp=0.0, right_grasp=0.0, right_ee=(0.0, 0.0, 0.0)):
    state = torch.zeros(QUARTER.state_dim)
    for vertex, pos in ((0, c0), (10, c10), (110, c110), (120, c120)):
        state[3 * vertex:3 * vertex + 3] = torch.tensor(pos)
    left, right = QUARTER.arms
    state[left.grasp_index], state[right.grasp_index] = left_grasp, right_grasp
    state[right.ee_slice] = torch.tensor(right_ee)
    return state


def test_quarter_stage_advances_with_bonus_then_succeeds():
    h = 0.15
    starts = torch.tensor([[-h, -h, 0.0], [-h, h, 0.0], [h, -h, 0.0], [h, h, 0.0]])
    r = FoldReward(goal=torch.cat([starts[0], starts[2]]), anchors0=starts, task=QUARTER)
    half = quarter_state(starts[0], starts[0], starts[2], starts[2])      # both corners placed, released
    rewards, terminated, stages = r.rollout(half.expand(22, -1)[None], torch.zeros(1, 21, 12))
    assert stages[0, 19] == 0 and stages[0, 20] == 1                      # advance after 20 settled steps
    assert rewards[0, 19].item() == STAGE_BONUS and not terminated.any()
    # stage 1: stack at (h, -h) released -> success after another 20 steps
    quarter = quarter_state(starts[2], starts[2], starts[2], starts[2])
    states = torch.cat([half.expand(21, -1), quarter.expand(21, -1)])[None]
    rewards, terminated, stages = r.rollout(states, torch.zeros(1, 41, 12))
    assert terminated[0, 39] and rewards[0, 39].item() == SUCCESS_BONUS      # 20 settled steps in stage 1
    assert not terminated[0, :39].any() and stages[0, 20] == 1


def test_quarter_release_regime_is_a_step_up_from_carrying():
    h = 0.15
    starts = torch.tensor([[-h, -h, 0.0], [-h, h, 0.0], [h, -h, 0.0], [h, h, 0.0]])
    r = FoldReward(goal=torch.cat([starts[0], starts[2]]), anchors0=starts, task=QUARTER)
    one = torch.ones(1, dtype=torch.long)
    held = quarter_state(starts[2], starts[2], starts[2], starts[2], right_grasp=1.0, right_ee=starts[2])
    released = quarter_state(starts[2], starts[2], starts[2], starts[2], right_grasp=0.0, right_ee=starts[2])
    torch.testing.assert_close(r.potential(released[None], one) - r.potential(held[None], one), torch.tensor([RELEASE_BONUS]))


def test_quarter_right_gripper_welds_any_stacked_corner():
    right = QUARTER.arms[1]
    state = quarter_state((0.0, 0.0, 0.0), (0.03, 0.0, 0.0), (1.0, 1.0, 0.0), (1.0, -1.0, 0.0), right_ee=(0.015, 0.0, 0.0))
    close = torch.zeros(12)
    close[right.gripper] = -1.0
    assert gripper_transition(state[None], close[None], right, QUARTER.grasp_radius) == (1.0, 1.0)
    far = quarter_state((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (1.0, 1.0, 0.0), (1.0, -1.0, 0.0), right_ee=(0.1, 0.0, 0.0))
    assert gripper_transition(far[None], close[None], right, QUARTER.grasp_radius) == (1.0, 0.0)
