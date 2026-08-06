# cloth_fold_rl — RL policy that actually folds the cloth

Trains a PPO policy to fold one corner of the cloth in `ClothFoldEnv`, and runs
it in the mjviser viewer with a live success readout.

## Just run it

The trained weights are committed, so this works straight from a clone — no
training required:

```bash
pip install -r cloth_fold_rl/requirements.txt
python -m cloth_fold_rl.run_trained
```

Use `cloth_fold_rl/requirements.txt`, **not** the repo-root one — the root pins
so101-nexus 0.4.8 / mujoco 3.10.0, but these checkpoints were trained on
0.5.1 / 3.11.0, and MuJoCo minor versions can shift contact-solver behaviour.

Opens the viewer at http://localhost:8080 running `outputs/cloth_fold_rl/run2/best.zip`
(100% success). It folds a corner every episode and auto-resets with a new random
cloth offset each time. The sidebar shows live `fold_score`, distance to goal,
and grasp state.

Checkpoints in the repo:

| file | what | success |
|---|---|---|
| `outputs/cloth_fold_rl/run2/best.zip` | **the model** — use this | 100% |
| `outputs/cloth_fold_rl/bc.zip` | behavior-cloned, pre-RL | 83% |
| `outputs/cloth_fold_rl/run2/latest.zip` | last fine-tune round (regressed) | 62% |
| `outputs/cloth_fold_rl/run1/*` | PPO from scratch — kept as the negative result | 0% |

Everything here is a **wrapper** around `mujuco/sim_main.py`. `ClothFoldEnv` is
not modified, so `cloth_angles/` keeps collecting data against the stock env.

## Why a wrapper was necessary

Three findings from measuring the stock env, all of which independently prevent
a fold from ever being learned:

**1. The default goal is not a fold.** `_goal_corners = corners0[[3, 2, 1, 0]]`
([sim_main.py:335](../mujuco/sim_main.py#L335)) sends every corner to the
diagonally-opposite corner's start position — all four travel 0.424 m:

```
(-0.15,-0.15) -> ( 0.15, 0.15)      (-0.15, 0.15) -> ( 0.15,-0.15)
( 0.15,-0.15) -> (-0.15, 0.15)      ( 0.15, 0.15) -> (-0.15,-0.15)
```

That is a 180° in-plane rotation of a *flat* sheet. Folding the cloth in half
stacks corners and makes `fold_score` go **down**. No policy can succeed at it
by folding.

**2. The cross-diagonal fold is kinematically impossible.** Sampling 40k joint
configurations per arm and taking forward kinematics:

| target | left arm | right arm |
|---|---|---|
| `cloth_0`   | reachable (0.013 m) | reachable (0.010 m) |
| `cloth_10`  | reachable (0.007 m) | **out of reach (0.080 m)** |
| `cloth_110` | **out of reach (0.083 m)** | reachable (0.007 m) |
| `cloth_120` | reachable (0.017 m) | reachable (0.014 m) |

Each arm's welded corner is reachable only by that arm, and the *opposite*
diagonal corner is ~8 cm outside its entire workspace at every height. So
`cloth_10 -> cloth_110` cannot be done. `cloth_10 -> cloth_120` (a 0.30 m edge
fold along the arm's strong +x direction) is fully reachable, and is the task
implemented here.

**3. The env's IK stalls.** `ik_substep()`
([sim_main.py:369](../mujuco/sim_main.py#L369)) is an undamped differential
solver that misses `cloth_10` by 0.104 m even though a valid configuration
exists within 0.007 m. This wrapper uses `action_mode="joint_delta"` to bypass
it entirely; `expert.py` carries its own damped-least-squares IK with random
restarts.

Also worth knowing: the stock reward is `-mean(corner_dists)` plus a sparse
`+10` ([sim_main.py:270](../mujuco/sim_main.py#L270)) — nothing rewards
approaching, grasping, or lifting, so random exploration essentially never
triggers a weld.

## The task as implemented

- **Goal** — `cloth_10` travels to `cloth_120`'s start position (0.30 m edge fold).
- **Success** — corner within 5 cm of target, held 10 control steps (0.5 s), with
  the two anchor corners not dragged more than 20 cm.
- **Action** — left arm only, 6-D: 5 joint deltas + gripper. Halves the
  exploration space; the right arm is held still.
- **Reward** — potential-based shaping, `reach -> grasp -> carry -> place`, plus
  a grasp bonus and a success bonus. Potential-based means releasing the cloth
  costs exactly what grasping paid, so the +3 grasp bonus cannot be farmed.

## Run 1: why PPO-from-scratch failed, and the bug it exposed

First 100k-step run looked like it was working — grasp rate went 0% → 25% →
100% and `fold_score` reached 0.135. It was not working. Evaluated on *varied*
starts the same checkpoint scored:

| | fixed start | varied starts |
|---|---|---|
| grasp rate | 100% | **0%** |
| fold_score | 0.108 | 0.0003 |
| return | 3.9 | 1.07 |

The policy had memorized one open-loop trajectory. The cause: `ClothFoldEnv.reset`
only consumes `np_random` when `domain_randomization=True`, so **every seed
produced a byte-identical start state** — all 8 parallel envs were running the
same episode, and the "6-episode" eval was one episode measured six times.

Fixed with `CLOTH_JITTER` (±2.5 cm per-episode offset, passed through the env's
existing `cloth_pose` option). Anything measured before that fix should be
treated as single-trajectory and not as evidence of a learned skill.

The scripted expert, being closed-loop, scores **4/4 under jitter** — which is
why it is used as a BC teacher rather than discarded after the feasibility gate.

## Usage

Prove the task is achievable (runs the scripted expert — do this before training):

```bash
python -m cloth_fold_rl.prove_feasible --episodes 3
```

Warm-start from the expert (recommended — PPO-from-scratch did not learn
closed-loop grasping, see above):

```bash
python -m cloth_fold_rl.collect_demos --episodes 200 --workers 8
python -m cloth_fold_rl.bc --epochs 30
```

Train. Runs in **rounds**: each round trains, evaluates on held-out seeds,
checkpoints, and the next round resumes from it. Re-launching with the same
`--run-dir` picks up where it left off. Add
`--init-from outputs/cloth_fold_rl/bc.zip` to start from the cloned policy.

```bash
python -m cloth_fold_rl.train --rounds 8 --steps-per-round 25000 --n-envs 8
tensorboard --logdir outputs/cloth_fold_rl/run1/tb
```

Watch a policy in the viewer, with live `fold_score` / grasp / success readout:

```bash
python -m cloth_fold_rl.run_trained                    # trained checkpoint
python -m cloth_fold_rl.run_trained --policy expert    # scripted expert
```

## Throughput

Measured on an M4 (10 cores): **9.0 control steps/sec** single-env — 100 physics
substeps per control step, 375 DOF, plus a 1.8 s settle per reset. About 45/sec
across 8 subprocess envs, so **1M steps ≈ 6 hours**. Budget accordingly.
