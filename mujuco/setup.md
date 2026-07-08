# SETUP

## 1. Create a virtual environment

```powershell
cd <your-work-dir>
python -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\.venv\Scripts\Activate.ps1
```

Python 3.12+ is required (the Windows patch below uses `delete_on_close`,
added in 3.12).

## 2. Install

CPU-only machine:

```powershell
pip install so101-nexus imageio
```

Machine with an NVIDIA GPU (adds the Warp backend):

```powershell
pip install "so101-nexus[warp]" imageio
```

## 3. Patch the installed package for Windows (required)

so101-nexus 0.4.6 has two bugs that only bite on Windows. Without this step,
**every** environment crashes at construction with
`ValueError: ParseXML: empty file '...\tmpXXXX.xml'`, and `render_mode="human"`
crashes with `AttributeError: module 'mujoco' has no attribute 'viewer'`.
Both fixes are hand edits inside `.venv\Lib\site-packages\so101_nexus\`.

**Fix 1 — temp-file crash.** The envs write their scene XML to a
`NamedTemporaryFile` and have MuJoCo re-open it by name while Python still
holds it open — fine on Linux/macOS, not allowed on Windows, so MuJoCo reads
zero bytes. In all 7 files that contain the pattern —
`mujoco\look_at_env.py`, `mujoco\move_env.py`, `mujoco\pick_and_place.py`,
`mujoco\pick_env.py`, `warp\look_at_env.py`, `warp\move_env.py`,
`warp\pick_env.py` — change:

```python
# before
with tempfile.NamedTemporaryFile(mode="w", suffix=".xml", dir=_SO101_DIR, delete=True) as f:
    f.write(xml_string)
    f.flush()
# after (delete_on_close needs Python 3.12+)
with tempfile.NamedTemporaryFile(mode="w", suffix=".xml", dir=_SO101_DIR, delete=True, delete_on_close=False) as f:
    f.write(xml_string)
    f.close()
```

The `mujoco.MjModel.from_xml_path(f.name)` line that follows stays as-is.
If an unpatched run already crashed, also delete any leftover `tmp*.xml` in
`so101_nexus\assets\SO101_menagerie\`.

**Fix 2 — viewer import.** `mujoco.viewer` is a submodule that `import mujoco`
alone doesn't load, and `mujoco\base_env.py` never imports it. Add under
`import mujoco` at the top of `base_env.py`:

```python
import mujoco.viewer
```

(Alternatively, put that import in your own script — it only matters for
`render_mode="human"`.)

**Re-apply both fixes after any `pip install`/`--upgrade` of so101-nexus**
(reinstalling overwrites the patched files).

## 4. GPU only: install CUDA-enabled PyTorch

The Warp environments return observations as torch tensors on the GPU, but the
default PyPI `torch` wheel on Windows is CPU-only. Replace it:

```powershell
pip install torch --index-url https://download.pytorch.org/whl/cu130 --upgrade
```

(~2 GB download. RTX 50-series cards need cu128 or newer; cu130 covers
everything current. Verify with
`python -c "import torch; print(torch.cuda.is_available())"` — must print `True`.)

## 5. Run the smoke test

```powershell
python smoke_test.py          # all 10 envs (5 CPU + 5 Warp)
python smoke_test.py --cpu    # CPU/MuJoCo envs only
python smoke_test.py --warp   # Warp GPU envs only
```

Each CPU env is created, reset, stepped 20x with random actions, and one
rendered frame is saved to `proof/<EnvId>.png`. Each Warp env runs 8 batched
worlds on CUDA for 20 steps. The script prints action/observation spaces,
rewards, and a PASS/FAIL summary; exit code 0 means everything passed.

The first Warp run spends ~2 minutes JIT-compiling CUDA kernels; they're
cached (`%LOCALAPPDATA%\NVIDIA\warp\Cache`), so later runs start fast.
`num_envs=8` is deliberately small — fine even on an 8 GB laptop GPU; watch
`nvidia-smi` memory before scaling it up.

## 6. Watching it live (interactive viewer)

`render_mode="human"` opens the MuJoCo viewer, but only when you call
`env.render()` — `step()` alone never opens a window (`sim.py` in this folder
is a ready-to-run version of this):

```python
import time

import gymnasium as gym
import so101_nexus.mujoco  # registers the MuJoCo*-v1 env IDs

env = gym.make("MuJoCoPickLift-v1", render_mode="human")
obs, info = env.reset()
for _ in range(1000):
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
    env.render()        # opens the viewer on first call, then syncs it
    time.sleep(1 / 50)  # real time (render_fps = 50)
    if terminated or truncated:
        obs, info = env.reset()
env.close()
```

## Environment IDs

Registered on `import so101_nexus.mujoco` / `import so101_nexus.warp`
(docs: https://so101-nexus.com/docs/getting-started/environment-ids):

| CPU (gym.make) | GPU (gym.make_vec) | Task |
|---|---|---|
| MuJoCoTouch-v1 | WarpTouch-v1 | Touch an object on the table |
| MuJoCoLookAt-v1 | WarpLookAt-v1 | Orient the end-effector toward a target |
| MuJoCoMove-v1 | WarpMove-v1 | Move the TCP in a cardinal direction |
| MuJoCoPickLift-v1 | WarpPickLift-v1 | Pick up and lift an object |
| MuJoCoPickAndPlace-v1 | WarpPickAndPlace-v1 | Pick and place at a target |

Warp envs are vector envs: `gym.make_vec("WarpTouch-v1", num_envs=8, device="cuda")`.

## Known non-fatal warnings

- `Failed to extract texture for YCB '...'` — the object renders gray instead
  of textured. Harmless.
- First use of a YCB object (e.g. pick tasks) downloads mesh assets to
  `~\.cache\so101_nexus\ycb\` — one-time, needs network.
