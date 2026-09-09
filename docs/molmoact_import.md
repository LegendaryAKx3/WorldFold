# Imported Policy Side: MolmoAct2

This branch focuses only on the imported-policy side of Task 1.

## Target

Use MolmoAct2 first because it is already SO100/SO101-targeted and SO101-Nexus
ships a dedicated adapter for it.

## Upstream Links

- MolmoAct policy docs: [molmoact.mdx](https://github.com/johnsutor/so101-nexus/blob/main/docs/content/docs/policies/molmoact.mdx)
- MolmoAct adapter source: [molmoact.py](https://github.com/johnsutor/so101-nexus/blob/main/src/so101_nexus/policy_adapters/molmoact.py)
- MolmoAct rollout smoke test: [smoke_molmoact_rollout.py](https://github.com/johnsutor/so101-nexus/blob/main/scripts/smoke_molmoact_rollout.py)

## Environment Choice

Use `MuJoCoTouch-v1` first. It is the easiest local import path because it is a
small SO101-Nexus MuJoCo task, avoids the extra difficulty of grasping/lifting,
and still exercises the same adapter contract that MolmoAct needs:

- `observation.state`
- `observation.images.overhead`
- `observation.images.wrist`
- absolute 6-joint actions in degree units from the policy, converted back to
  radians by `RolloutRecorder`

The local installed SO101-Nexus version needs an explicit visual config to
produce `state`, `overhead_camera`, and `wrist_camera` observations, so the
smoke script adds that config before creating the env.

## Commands

Base install:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

Dry-run the adapter and rollout path without downloading the real model:

```bash
.venv/bin/python scripts/smoke_molmoact_import.py
```

Install the optional real-model dependencies:

```bash
.venv/bin/python -m pip install -r requirements-molmoact.txt
```

Run the real MolmoAct2 checkpoint:

```bash
.venv/bin/python scripts/smoke_molmoact_import.py \
  --load-model \
  --dtype bfloat16 \
  --device-map auto
```

On this Mac, the lighter local CPU smoke command that completed was:

```bash
.venv/bin/python scripts/smoke_molmoact_import.py \
  --load-model \
  --dtype bfloat16 \
  --max-steps 1 \
  --chunk-size 1 \
  --num-steps 1 \
  --disable-cuda-graph
```

## Current Local Result

The dry-run path passed locally. It imports SO101-Nexus `MolmoActPolicy` and
`RolloutRecorder`, builds a visual `MuJoCoTouch-v1` environment, and sends a
local echo model through the same chunked policy interface. Results are written
to `outputs/molmoact_import/probe.json`.

Observed dry-run contract:

- observation keys: `state`, `overhead_camera`, `wrist_camera`
- state shape: `(6,)`
- overhead camera shape: `(120, 160, 3)`
- wrist camera shape: `(120, 160, 3)`
- rollout: 1 episode, 8 steps, action chunks shaped `(8, 6)`

The real checkpoint path also ran locally after installing
`requirements-molmoact.txt`. The Hugging Face cache for
`allenai/MolmoAct2-SO100_101` is about 20 GB on this machine.

Observed real-model smoke result:

- status: `model_rollout_ran`
- env: `MuJoCoTouch-v1`
- rollout: 1 episode, 1 step
- action output shape: `(1, 6)`
- success: `false`
- output: `outputs/molmoact_import/model_attempt.json`

The script applies a small compatibility shim because the installed
SO101-Nexus `MolmoActPolicy` adapter calls `predict_action(action_mode=...)`,
while the downloaded MolmoAct2 checkpoint currently expects
`predict_action(inference_action_mode=...)`.
