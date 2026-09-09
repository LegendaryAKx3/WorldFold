# World model: benchmarks, analytic reward, imagination

The repo's world model went from a latent RSSM over angle fields that lost to
persistence by an order of magnitude, to a residual MLP over physical state
that predicts held-out fold episodes with a corner error of 10.6 mm at ten
steps (persistence: 35.5 mm). The fold reward is computed from predicted
state with the environment's own formula. Training an actor on imagined
rollouts through that model raised a behavior-cloned policy from 17.5% to
87.5% success in one round and 95% after a round of its own collected data.
The expert and PPO still solve the task at about
100%, so the imagination result is a demonstration that the model is useful
for policy learning. Sections 1 and 2 are history on the stock 14-action
environment; everything from section 3 on runs on the 6-action
`SingleCornerFoldEnv`.

All prediction benchmarks use the same protocol: fixed windows of ten
observed transitions followed by 1, 5 and 10 predicted steps (stride ten),
predictions fed back autoregressively with recorded actions, three training
seeds, sample standard deviation across seeds, persistence (repeat the origin
state) and constant velocity (extrapolate the last observed change) as
baselines. Every metric is averaged over held-out test episodes.

## 1. RSSM timing fix

The RSSM trained its decoder against a target one step ahead of its latent.
`rssm.observe_transitions` and the prior-only one-step evaluator fix that.
Matched old and fixed models (2,000 updates, 18 training episodes, 10 test
episodes, 190 windows) on the stock environment:

| Horizon | Old MAE (rad) | Fixed MAE (rad) | Persistence |
|---|---:|---:|---:|
| 1 | 0.1459 +/- 0.0044 | 0.1428 +/- 0.0021 | 0.0108 |
| 5 | 0.1532 +/- 0.0050 | 0.1513 +/- 0.0023 | 0.0331 |
| 10 | 0.1637 +/- 0.0061 | 0.1637 +/- 0.0034 | 0.0518 |

The fix repairs the training contract. The model still reconstructs a field
it has just observed with 0.14 rad of error while the cloth moves 0.011 rad
per step. Checkpoints trained before the fix need retraining.

## 2. Residual angle predictor

Same data and windows, angle-only inputs, 10,000 updates. Each column removes
one thing from the column to its left: the stochastic recurrent latent, then
regenerating the whole field instead of predicting its change, then the MSE
loss.

| Horizon | RSSM | Absolute MLP | Residual MLP, MSE | Residual MLP, L1 | Persistence | Const. vel. |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 0.1437 | 0.0845 | 0.0386 | 0.0147 | 0.0108 | 0.0149 |
| 5 | 0.1511 | 0.1658 | 0.1017 | 0.0535 | 0.0331 | 0.0640 |
| 10 | 0.1611 | 0.2734 | 0.1426 | 0.0915 | 0.0517 | 0.1139 |

Three things carried forward. The latent bottleneck was the largest single
cause of error. Training length was irrelevant (the RSSM at 10,000 updates
matches itself at 2,000). The L1 loss matters because the signed angle
convention flips sign when a hanging cell's normal crosses x = 0: 0.2% of
cell changes carry 78% of the squared change, and MSE chases them. The L1
residual model beat persistence on training windows at every horizon and
lost on test windows, with a seed spread of 0.0001 rad, so what remained was
generalization across cloth parameters, from 18 episodes with angle-only
inputs that cannot tell a grasped corner from a free one.

## 3. Fold-environment state predictor

`scripts/collect_fold_state_episodes.py` records, every control step of
`SingleCornerFoldEnv` under physical randomization: 121 cloth vertex
positions, the left arm's joint positions and velocities, gripper command,
end-effector position and grasp weld flag (378 values), the 6-dimensional
action, reward, termination, and per episode the goal and anchor corners.
Five kinds of episode with disjoint seeds: scripted expert, expert with
Gaussian action noise (std 0.3), expert with a forced 15-step release, PPO
(`outputs/cloth_fold_rl/run2/best.zip`, stochastic), PPO with forced
release. The model is `ResidualStatePredictor`: the same residual MLP,
predicting the per-dimension normalized change of the whole state vector,
robot part included. `scripts/benchmark_state_predictor.py` trains and scores it.

Test windows, vertex MAE and moving-corner error in millimetres. The first
dataset has 50 training and 10 test episodes; the second has 250 and 50
(634 windows, 254 with the corner grasped at the origin). Constant velocity
is shown for vertex MAE only.

| Data | Horizon | Vertex only, L1 | Full state, L1 | Full state, MSE | Persistence | Const. vel. |
|---|---:|---:|---:|---:|---:|---:|
| 50 eps | 1 | 0.189 / 3.9 | 0.171 / 3.1 | 0.236 / 4.3 | 0.327 / 5.7 | 0.122 |
| 50 eps | 5 | 0.923 / 18.9 | 0.813 / 15.4 | 1.127 / 19.2 | 1.574 / 24.9 | 0.859 |
| 50 eps | 10 | 1.880 / 37.9 | 1.617 / 31.0 | 2.207 / 35.4 | 3.032 / 45.7 | 2.153 |
| 250 eps | 1 | | 0.093 / 1.19 | | 0.274 / 4.59 | 0.106 |
| 250 eps | 5 | | 0.375 / 4.70 | | 1.268 / 18.87 | 0.731 |
| 250 eps | 10 | | 0.702 / 10.56 | | 2.419 / 35.49 | 1.841 |

On grasped windows at ten steps with 250 episodes, corner error is 15.4 mm
against 74.5 mm for persistence. Seed spread is below 0.02 mm on vertex
error. Adding robot and grasp state to the vertices helps at every horizon,
L1 beats MSE again (grasp and release are large jumps in the change
distribution), and with 250 episodes the training-window error (0.094 mm)
equals the test error (0.093 mm), so the generalization gap from section 2
is closed by data.

## 4. Reward and termination from predicted state

`cloth_angles/model/fold_reward.py` mirrors `cloth_fold_rl/fold_env.py` on
the 378-value state: potential shaping from gripper-to-corner and
corner-to-goal distance switched by the grasp flag, control cost, success
bonus after ten consecutive placed-and-grasped steps, drag penalty and
termination when an anchor corner moves more than 20 cm. The "unstable"
termination reads joint accelerations the state omits and is not
reproduced. Against the recorded rewards of all 300 episodes the maximum
difference is 2e-6 and every termination flag matches.

`gripper_transition` reproduces the simulator's gripper and weld rule
(close below -0.3, open above +0.3, hysteresis between, weld within 3 cm
while closed, release on open). Computed from recorded state it matches all
42,230 transitions. Imagination uses it because the learned model never
reproduces grasp or release: they are about one transition per episode and
the L1 loss treats them as outliers. A soft version with linear ramps keeps
the actor's gradient path through the gripper command.

## 5. Actor-critic in imagination

`cloth_angles/model/actor_critic.py`, driven by
`scripts/train_imagined_actor.py`. Dreamer-style without a latent: batches
of 256 real training states are rolled ten steps through the frozen model
with the actor's sampled actions, scored by the analytic reward, and the
actor is trained by backpropagating lambda-returns (gamma 0.98, lambda 0.95)
through the dynamics. The critic regresses to the same returns on detached
states and bootstraps from a slow EMA copy of itself, with the value loss in
return-scaled units. Actor features are the normalized state, the episode
goal and the corner's offset to it; actions are a tanh-squashed Gaussian
with a learned standard deviation capped at 0.5 and a 1e-4 entropy bonus.

Three protections were each added after a failed run. Without the EMA target
critic the critic loss reached 1e9 within 1,000 updates. Without the exact
gripper rule the imagined grasp rate was frozen at the fraction of start
states already grasped and both actors scored 0% in the simulator. Without
an out-of-distribution cutoff (imagined steps whose normalized features
exceed 8 standard deviations are terminal with zero value) and a
behavior-cloning regularizer, the actor drove the model 25 standard
deviations from the data and the imagined return rose while real grasps
collapsed.

Evaluation on 60 fresh seeds, physical randomization on, deterministic actions:

| Policy | Success | Grasp | Fold score | Mean steps |
|---|---:|---:|---:|---:|
| expert | 60/60 | 100% | 0.925 | 89 |
| ppo | 59/60 | 100% | 0.909 | 87 |
| bc (3,000 cloning updates on the dataset) | 26/60 | 100% | 0.589 | 163 |
| imagined_bc (bc + 3,000 imagination updates, regularizer 1.0) | 35/60 | 100% | 0.637 | 128 |
| imagined from scratch | 0/60 | 0% | 0.001 | 200 |

Paired by seed the imagined actor wins on 15 seeds and loses on 6 (sign
test p = 0.08). Pooled with the 20 development seeds: 34/80 against 48/80.
Failures of both actors are time limits while carrying, with no drags or
instability. Imagination from a random actor never grasps: dynamics
backpropagation gives no gradient toward an event that has not happened,
and the loop has no exploration or real-data collection yet.

## 6. Closing the loop: two initializations

`scripts/loop_experiment.py`. Two starting actors were compared: arm A
cloned from the diverse 300-episode dataset, arm B cloned from 60
deterministic PPO episodes (all successful). Both share a 4-member world-model ensemble (10,000 updates each) and are trained
in imagination with a disagreement penalty (coefficient 2) plus a behavior
anchor toward their own cloning data decayed 1.0, 0.5, 0.25 by round. After
each round an arm collects 50 episodes with exploration noise (std 0.3)
that join the ensemble's training data. Evaluation on 40 seeds under the
training distribution, cloth offset widened from 2.5 cm to 6 cm, and cloth
mass scaled by 1.6.

A first attempt with the penalty and no anchor collapsed both arms to 0%
while every imagined metric looked healthy. Calibration showed why: members
trained on the same data disagree about as much on real transitions (0.022)
as after ten random actions (0.028), so the penalty has no signal to work
with. Round 0 with the anchor:

| Policy | In distribution | Mass x1.6 | Offset 6 cm |
|---|---:|---:|---:|
| expert | 100% | 100% | 100% |
| ppo | 100% | 100% | 85% |
| A diverse, clone | 17.5% | 72.5% | 22.5% |
| A diverse, round 0 | 87.5% | 30.0% | 40.0% |
| B ppo, clone | 80.0% | 55.0% | 35.0% |
| B ppo, round 0 | 90.0% | 0% (grasp 0%) | 25.0% |
| A diverse, round 1 | 95.0% | 65.0% | 42.5% |

Collecting with exploration noise after round 0, arm A succeeded in 86% of
its episodes and arm B in 20%. The PPO head start disappears after one round
and arm B is worse under every perturbation, since a clone of a deterministic
policy only knows the states that policy visits. Arm B is dropped and the
script now runs the single diverse-start actor.

Round 1 retrained the ensemble on 49,692 transitions (base data plus both
collections) and trained the actor for another 3,000 imagination updates
with the anchor at 0.5. Paired by seed against round 0: in distribution 3
improved and 0 regressed; under heavier cloth 18 improved and 4 regressed;
under the wider offset 4 improved and 3 regressed. Every remaining failure
is a time limit. One round of the actor's own data recovered most of the
heavier-cloth robustness that round 0 had lost (72.5% at the clone, 30% at
round 0, 65% at round 1), even though none of the collected episodes used
shifted physics: the round-0 actor visited states the base data lacked, and
the retrained model extrapolates from them. The wider offset stays near 40%
across rounds, so offset coverage has to come from collection. To resume
(next step: collect 50 episodes with the round-1 actor, retrain, round 2):

```bash
OMP_NUM_THREADS=1 .venv/bin/python scripts/loop_experiment.py --stage run --output outputs/cloth_angles/loop_experiment_anchored --disagreement-coef 2.0 --bc-coef-schedule 1.0 0.5 0.25
```

## 7. A harder task: the quarter fold

The single-corner task is solved at 100% by PPO with no world model, so it
cannot show what imagination adds. `cloth_fold_rl/quarter_fold_env.py` is a
two-stage task on the same cloth. In stage 0 the left arm carries cloth_10
onto cloth_0 and the right arm carries cloth_120 onto cloth_110 (a fold about
y = 0), both release, and the corners must stay within 5 cm of their goals
for 20 steps. In stage 1 the right arm picks up the stacked cloth_0 and
cloth_10, carries them onto cloth_110 (a fold about x = 0), releases, and
the stack must stay placed for 20 steps. Only the right arm reaches the
stack (the README's reach table). Rewards are the same potential shaping per
move with a third regime for a released and placed corner, a stage bonus of
10, a success bonus of 20, drag and instability terminations, 12 actions,
400 steps.

Two simulator changes were needed, both as constructor options with the old
behaviour as default. Each gripper welds one hard-wired vertex in the stock
env; the right gripper here may weld cloth_120, cloth_0 and cloth_10, every
one within the grasp radius when it closes, so it can pick up a stack. The
grasp radius is 4 cm instead of 3: corners placed within the 5 cm tolerance
can sit 5 cm apart, beyond what a 3 cm radius covers from any one point.

The scripted expert runs one single-corner phase machine per move, with the
left arm parking out of the way in stage 1. Two things make the task hard
for it. A released corner springs back toward the fold line, by 4 to 5 cm
for the half fold and anywhere from under 1 cm to over 8 cm for the
two-layer stack, depending on the randomized cloth. The expert places past
the goal by the mean spring-back, and in stage 1 also waits for the cloth to
settle and re-grasps with the measured error added to its goal, up to twice.
On 30 fresh seeds with physical randomization:

| | Episodes |
|---|---:|
| success | 21 / 30 |
| reached stage 1 | 29 / 30 |
| needed a corrective re-placement | 14 / 30 |
| mean steps | 317 |

Failures are time limits on the second placement, and one pickup that pushed
the stack apart. The world-model stack is now parametrized by a task
(`cloth_angles/tasks.py`: arms, staged moves, weld corners, release rule),
and the analytic reward carries the stage and settle counter through a
rollout. Checked against the recorded rewards of collected quarter-fold
episodes, the maximum difference is 2e-6 and every stage transition and
termination matches.

Data: 300 episodes (100 each of expert, noisy expert and forced release,
seeds 40000 upward, 10 per kind held out), 95,682 training transitions. The
same full-state L1 predictor (393 values, 12 actions, 10,000 updates, three
seeds) on the 30 test episodes, vertex MAE and cloth_10 error in millimetres:

| Horizon | World model | Persistence | Const. vel. (vertex) |
|---|---:|---:|---:|
| 1 | 0.181 / 1.5 | 0.365 / 3.4 | 0.142 |
| 5 | 0.782 / 5.9 | 1.672 / 14.6 | 1.080 |
| 10 | 1.470 / 11.6 | 3.156 / 26.6 | 2.762 |
| 10, corner grasped at origin | 1.704 / 20.1 | 8.440 / 85.0 | 7.350 |

Seed spread is below 0.01 mm. The training-window vertex error is 0.147 mm
against 0.181 on test, a wider gap than the single task's at a similar
number of transitions; the episodes are longer and more varied.

### First imagination run: a negative result

Same recipe as section 5 (world model 20,000 updates, clone 3,000 updates,
imagination 3,000 updates at horizon 10 with anchor weight 1), 20 seeds:

| Policy | Success | Reached stage 1 | Grasp | Fold score |
|---|---:|---:|---:|---:|
| expert | 10/20 | 18/20 | 100% | 0.88 |
| clone | 0/20 | 15/20 | 95% | 0.69 |
| clone + imagination | 0/20 | 0/20 | 5% | 0.02 |
| scratch + imagination | 0/20 | 0/20 | 0% | 0.00 |

The clone completes the half fold on most seeds and never the second fold.
Imagination then destroyed it. The imagined actor's error to the data
actions grew from 0.063 to 0.097, nearly all of it a constant bias on two
joints (left elbow +0.28, right shoulder pan +0.18 where the data averages
zero). Over ten imagined steps that moves an end effector 3 to 4 cm and
earns a small positive potential; over 400 real steps it walks the arms
away from the cloth, which a ten-step horizon never shows the model. The
anchor weight that held on the single task does not hold here.

## Artifacts and reproduction

Everything under `outputs/cloth_angles/` is gitignored. `timing_benchmark/`
and `residual_benchmark/` hold the section 1 and 2 data, checkpoints and
`summary.json`; the scripts that produced them and the original four
long-form writeups are kept in `archive/`. `fold_state/` (60 episodes) and
`fold_state_v2/` (300) are the section 3 datasets, scored in
`state_benchmark/`, `state_benchmark_v2_50/` and `state_benchmark_v2_250/`.
`imagined_actor_v3/` holds the section 5 world model, actors, imagination
logs and evaluations. `loop_experiment/` (penalty only),
`loop_experiment_anchored/` (arm A files renamed to `actor*`; arm B's
`B_ppo*` files and `collect_0_B_ppo` retained as data) and `ppo_det/` (the
arm B cloning data) hold section 6.

```bash
OMP_NUM_THREADS=1 .venv/bin/python scripts/collect_fold_state_episodes.py --per-kind 60 --test-per-kind 10 --seed-base 30000 --output outputs/cloth_angles/fold_state_v2
OMP_NUM_THREADS=1 .venv/bin/python -m cloth_fold_rl.quarter_fold_expert --episodes 30 --seed-base 100 --quiet
OMP_NUM_THREADS=1 .venv/bin/python scripts/collect_fold_state_episodes.py --task quarter --per-kind 100 --test-per-kind 10 --seed-base 40000 --output outputs/cloth_angles/quarter_state
OMP_NUM_THREADS=1 .venv/bin/python scripts/benchmark_state_predictor.py --data outputs/cloth_angles/fold_state_v2 --output outputs/cloth_angles/state_benchmark_v2_250 --variants state_l1 vertex_l1 state_mse
OMP_NUM_THREADS=4 .venv/bin/python scripts/train_imagined_actor.py --data outputs/cloth_angles/fold_state_v2 --output outputs/cloth_angles/imagined_actor_v3
OMP_NUM_THREADS=1 .venv/bin/python scripts/train_imagined_actor.py --output outputs/cloth_angles/imagined_actor_v3 --eval-only --eval-episodes 60 --eval-seed-base 60000
OMP_NUM_THREADS=1 .venv/bin/python scripts/loop_experiment.py --stage run --output outputs/cloth_angles/loop_experiment_anchored --disagreement-coef 2.0 --bc-coef-schedule 1.0 0.5 0.25
```

Tests: `python -m pytest cloth_angles/tests`.
