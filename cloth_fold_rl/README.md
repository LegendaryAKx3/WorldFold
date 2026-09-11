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

Everything here is a **wrapper** around `mujuco/sim_main.py`. `ClothFoldEnv`'s
defaults are unchanged, so `cloth_angles/` keeps collecting data against the
stock env; the quarter-fold wrapper (`quarter_fold_env.py`) passes it two
options, `grasp_corners` (several weldable vertices per gripper) and
`grasp_radius`.

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

## Physical grasp (no weld cheat)

Everything above fakes the grasp with a weld. `--physical` runs the same
pipeline on the grabber from `mujuco/prove_grabber.py`: scoop ramp + paddle
plates on the SO101 jaws, weld never engaged, holding the cloth is contact and
friction only. Code: `physical_env.py` (env), `physical_expert.py` (scripted
scoop-pinch-carry expert). Every script takes `--physical`; outputs go to
`outputs/cloth_fold_rl/physical/`.

```bash
python -m cloth_fold_rl.prove_feasible --physical --episodes 4
python -m cloth_fold_rl.collect_demos --physical --episodes 200 --workers 8
python -m cloth_fold_rl.bc --physical --epochs 30
python -m cloth_fold_rl.train --physical --run-dir outputs/cloth_fold_rl/physical/run1 \
    --init-from outputs/cloth_fold_rl/physical/bc.zip --rounds 4
python -m cloth_fold_rl.record_video --physical --checkpoint outputs/cloth_fold_rl/physical/run1/best.zip
```

**Pin MuJoCo 3.10.0 for this** (the repo-root `requirements.txt`, not this
directory's). The grabber proof passes on 3.10.0 and fails on 3.11.0 with
either so101-nexus version: on 3.11 the cloth settles ~1 cm above the table
instead of on it, the ramp no longer gets under the corner, and the jaw ejects
it on closing. The weld checkpoints above are the only thing that needs 3.11.

### What is different from the weld task, and why

Three measured facts about the SO101 + plates forced three changes:

1. **Grasp signal.** The weld flag (reward potential, success test, one proprio
   observation slot) is replaced by a physical one: jaw commanded closed AND the
   corner within 3 cm of the ramp/paddle pocket and not lying under it. Success
   no longer requires the flag -- a corner set down on its target is a fold.
2. **Cloth placed 2 cm further from the left arm** (`CLOTH_SHIFT`, jitter kept).
   The scoop posture has `shoulder_lift` on its joint limit with the arm
   stretched low; for corners closer than ~15 cm to the arm base the gripper
   body self-collides with the shoulder body (36 N contact, elbow and wrist
   servos saturated at 3.35 N m) and the ramp stops ~1 cm above the table. The
   +-2.5 cm jitter reached that region for half the episodes.
3. **Fold goal at half the edge fold** (`FOLD_FRACTION = 0.5`, 0.15 m carry).
   The pocket only holds while the jaw keeps its grasp-time tilt, and with
   that orientation held the 5-DOF arm cannot track the carry line past
   x = +0.10 m (kinematic probe: 12 mm error at 80% of the full fold, 55-77 mm
   at the goal). Position-only tracking reaches the goal but rotates the jaw
   30-50 deg, which dumps the corner around x = 0. Set `FOLD_FRACTION = 1.0`
   to get the weld task's goal back.

Episodes are 250 steps (the approach needs ~50 more than the weld task).

### Expert

`ScoopExpert` ports the proof's routine to joint-delta actions:
hover -> down -> slide -> close -> lift -> carry -> place -> hold. Two IK
details matter. The approach phases use position-only DLS with a nullspace
pull toward the proof's scoop posture (`Q_SCOOP`), because the ramp is bolted
to the gripper and a free wrist puts its leading edge in the table or in the
air. The carry uses a 6-DoF tracker holding the grasp-time tilt (yaw free), on
a rate-limited target that only advances while the arm keeps up -- the first
version let the IK swap arm configuration mid-carry and yanked the corner out.
The slide is the proof's profile (5 mm target hops, then hold): a smooth creep
at the same average speed shoves the cloth along the table instead of
scooping it. Note `mju_subQuat` returns the rotation error in the site's local
frame while `mj_jacSite` is in world frame; the env's own `ik_substep` stacks
them without rotating, which is part of why it stalls.

### Results

| stage | result |
|---|---|
| feasibility gate (4 seeds) | 3/4 success, grasp rate 100%, best fold_score 0.813 |
| expert demos (200 episodes) | 70 successes (35%), 10,161 transitions |
| BC policy (6 held-out seeds) | _see below_ |
| PPO fine-tune | _see below_ |

The expert's 35% over the full jitter box (vs 3/4 on the gate seeds) is the
scoop: the corner either rides the ramp within a few hops or gets pushed ahead
of it, and the carry drops it if the cloth tension peaks before the goal.
