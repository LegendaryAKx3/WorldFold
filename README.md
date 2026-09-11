# WorldFold

Dual-arm cloth manipulation in MuJoCo. Two SO101 arms and a simulated cloth, wrapped as a Gymnasium environment, plus a world model trained on the cloth's surface-angle field

WatAI, Spring/Fall 2026

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -r requirements-molmoact.txt   # optional, for MolmoAct2
```

Windows: see [mujuco/other/setup.md](mujuco/other/setup.md).

## Usage

```bash
python mujuco/sim_main.py                                  # scripted fold demo in the viewer
python mujuco/prove_grabber.py                             # physical grabber PoC (no weld cheat)
python -m cloth_angles.collect_episodes --config cloth_angles/config.yaml
python -m cloth_angles.train --config cloth_angles/config.yaml
python scripts/train_ppo.py --env MuJoCoTouch-v1 --timesteps 100000
python scripts/run_policy.py --env MuJoCoTouch-v1 --policy ppo \
    --checkpoint outputs/ppo/MuJoCoTouch-v1/model.zip --episodes 5
```

## Layout

```
mujuco/sim_main.py   ClothFoldEnv: tasks fold/drop/push/drag, state/pixels/hybrid obs, 14-dim action
mujuco/simulations/  earlier prototypes and smoke test
cloth_angles/        RSSM world model over cloth angle fields: collect, train, evaluate
policy_runner/       policy interface and registry (random, ppo)
scripts/             train, evaluate, record videos
docs/                experiment notes
outputs/videos/      recorded rollouts
```

## Docs

World-model training consumes each `action_t` before reconstructing `next_obs_t`.
Evaluation reports prior-only next-state prediction MAE separately from posterior
reconstruction MAE. Retrain world-model checkpoints produced before this timing
fix: their decoder was trained against a target one timestep ahead of its latent.
Existing PPO checkpoints are unaffected.

Run the cloth-angle tests with `python -m pytest cloth_angles/tests` (install
`pytest` in the development environment first).

- [PPO baseline](docs/ppo_training.md)
- [World model: benchmarks, analytic reward, imagination](docs/world_model.md)
- [MolmoAct2 import](docs/molmoact_import.md)
- [Adding a policy](policy_runner/NOTES.md)
