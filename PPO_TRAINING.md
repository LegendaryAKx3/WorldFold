# PPO Baseline: MuJoCoTouch-v1

## Task

Train a small PPO baseline using SO101-Nexus, for comparison against the
imported MolmoAct2 policy and the random baseline (see `MOLMOACT_IMPORT.md`).

## Environment Choice

`MuJoCoTouch-v1`, matching the env used for the imported MolmoAct2 policy:


With default (non-visual) observations, `MuJoCoTouch-v1` gives a flat
`Box(18,)` state observation and `Box(6,)` continuous action space, so a
plain `MlpPolicy` PPO trains without needing an image encoder or GPU.

## Commands

Train:

```bash
python scripts/train_ppo.py --env MuJoCoTouch-v1 --timesteps 100000
```

Writes:
- checkpoint: `outputs/ppo/MuJoCoTouch-v1/model.zip`
- tensorboard logs: `outputs/ppo/MuJoCoTouch-v1/tensorboard/`

Evaluate the checkpoint through the same runner used for every other policy:

```bash
python scripts/run_policy.py --env MuJoCoTouch-v1 --policy ppo \
    --checkpoint outputs/ppo/MuJoCoTouch-v1/model.zip --episodes 5
```

View training curves (reward, episode length, loss) in TensorBoard:

```bash
tensorboard --logdir outputs/ppo/MuJoCoTouch-v1/tensorboard
```


## Results

### Smoke run: 4,096 timesteps

Training completed cleanly in ~1 second. `ep_rew_mean` rose from ~70 to
~82 within 2 rollout iterations, showing PPO is receiving a real learning
signal on this env immediately. No success within these very few episodes.

Eval (checkpoint at 4,096 steps, 3 episodes, max 100 steps):

| episode | reward | length | success |
|---------|--------|--------|---------|
| 1       | 19.88  | 100    | False   |
| 2       | 21.64  | 100    | False   |
| 3       | 10.64  | 100    | False   |

avg reward: 17.39, avg length: 100.00, success rate: 0%

### Longer run: 100,000 timesteps

Training completed cleanly in ~27 seconds (100k steps at ~3,858 fps, 48
PPO update iterations). `ep_rew_mean` plateaued around 82-88 and
`ep_len_mean` dropped from ~340 (random-ish early rollouts) to ~200-215,
consistent with the policy learning more decisive trajectories rather than
wandering for the full 512-step budget.

Eval (checkpoint at 100,000 steps, 10 episodes, max 512 steps):

| episode | reward | length | success |
|---------|--------|--------|---------|
| 1       | 274.85 | 512    | False   |
| 2       | 234.72 | 512    | False   |
| 3       | 282.87 | 512    | False   |
| 4       | 9.22   | 18     | True    |
| 5       | 280.04 | 512    | False   |
| 6       | 271.34 | 512    | False   |
| 7       | 282.72 | 512    | False   |
| 8       | 249.37 | 512    | False   |
| 9       | 9.07   | 19     | True    |
| 10      | 200.48 | 512    | False   |

avg reward: 209.47, avg length: 413.30, success rate: 20%

Both successes touched the target almost immediately (18-19 steps) and
terminated early, which is why their per-episode reward is much lower than
the failed, full-length episodes (512-step episodes accumulate more
per-step reward even without reaching the touch threshold). This is a real
signal: the policy has learned a strategy that occasionally reaches the
object quickly, but is inconsistent — most rollouts still run out the full
episode without touching the target.

## Rollout Videos

`scripts/record_rollout.py` renders a single rollout (same env-stepping loop
as `run_policy.py`) to an mp4, using `env.render(mode="rgb_array")` per step
and `imageio` to encode the frames:

```bash
python scripts/record_rollout.py --env MuJoCoTouch-v1 --policy random \
    --seed 0 --max-steps 200

python scripts/record_rollout.py --env MuJoCoTouch-v1 --policy ppo \
    --checkpoint outputs/ppo/MuJoCoTouch-v1/model.zip --seed 3 --max-steps 512 \
    --output outputs/videos/MuJoCoTouch-v1_ppo_success.mp4

python scripts/record_rollout.py --env MuJoCoTouch-v1 --policy ppo \
    --checkpoint outputs/ppo/MuJoCoTouch-v1/model.zip --seed 0 --max-steps 512 \
    --output outputs/videos/MuJoCoTouch-v1_ppo_typical.mp4
```

Recorded (matches the eval numbers above exactly, same seeds):

| video | policy | seed | steps | reward | success |
|---|---|---|---|---|---|
| `outputs/videos/MuJoCoTouch-v1_random.mp4` | random | 0 | 200 | 44.16 | False |
| `outputs/videos/MuJoCoTouch-v1_ppo_success.mp4` | ppo (100k ckpt) | 3 | 18 | 9.22 | True |
| `outputs/videos/MuJoCoTouch-v1_ppo_typical.mp4` | ppo (100k ckpt) | 0 | 512 | 274.85 | False |

`ppo_success` shows the policy reaching the target almost immediately;
`ppo_typical` shows the more common outcome, where the arm moves purposefully
but doesn't reach the touch margin within the episode budget. These files
are gitignored (`outputs/`) since they're generated artifacts — rerun the
commands above to regenerate them.

