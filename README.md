# WorldFold

Dual-arm cloth manipulation in MuJoCo. Two SO101 arms and a simulated cloth, wrapped as a Gymnasium environment, plus a world model trained on the cloth's surface-angle field

WatAI, Spring/Fall 2026

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r cloth_fold_rl/requirements.txt
```

Windows: see [mujuco/other/setup.md](mujuco/other/setup.md).

## Usage

```bash
# watch the trained fold policy (100% success), viewer at http://localhost:8080
python -m cloth_fold_rl.run_trained --checkpoint outputs/cloth_fold_rl/run2/best.zip
python -m cloth_fold_rl.run_trained --policy expert          # scripted expert instead

# retrain from scratch: expert demos -> behavior cloning -> PPO fine-tune
python -m cloth_fold_rl.prove_feasible --episodes 3
python -m cloth_fold_rl.collect_demos --episodes 200 --workers 8
python -m cloth_fold_rl.bc --epochs 30
python -m cloth_fold_rl.train --init-from outputs/cloth_fold_rl/bc.zip --rounds 8
python -m cloth_fold_rl.record_video --checkpoint outputs/cloth_fold_rl/run2/best.zip

# world model
python -m cloth_angles.collect_episodes --config cloth_angles/config.yaml
python -m cloth_angles.train --config cloth_angles/config.yaml
```

## Layout

```
mujuco/sim_main.py   ClothFoldEnv: tasks fold/drop/push/drag, state/pixels/hybrid obs, 14-dim action
cloth_fold_rl/       single-corner fold task: env wrapper, scripted expert, BC, PPO, committed checkpoints
mujuco/simulations/  earlier prototypes and smoke test
cloth_angles/        RSSM world model over cloth angle fields: collect, train, evaluate
policy_runner/       policy interface and registry (random, ppo)
scripts/             train, evaluate, record videos
docs/                experiment notes
outputs/videos/      recorded rollouts
```

## Docs

- [Fold task: why the wrapper, results, physical grasp](cloth_fold_rl/README.md)
- [World model: benchmarks, analytic reward, imagination](docs/world_model.md)
- [Adding a policy](policy_runner/NOTES.md)
