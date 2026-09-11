"""Render an offline video of the quarter-fold scripted expert -- no live
viewer, so it isn't bottlenecked by real-time playback (self-collision alone
runs the live sim at ~5 steps/s during the actual fold). Renders at the sim's
native control rate (1/control_dt = 20fps) and lets ffmpeg upsample to a
smooth 60fps container on encode, so real-time duration is preserved.

    python scripts/render_quarter_fold.py --episodes 1 --out renders/quarter_fold.mp4
"""

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "mujuco"))

import imageio.v2 as imageio
import mujoco
import sim_main

from cloth_fold_rl.quarter_fold_env import QuarterFoldEnv
from cloth_fold_rl.quarter_fold_expert import QuarterFoldExpert

NATIVE_FPS = 20   # 1 / control_dt
OUT_FPS = 60
# mujoco.Renderer draws the flex's actual COLLISION geometry (CLOTH_RADIUS =
# 0.01, physics-only) as its visual thickness -- there's no per-frame custom
# mesh injection like mjviser's (viser is a separate web scene, not MuJoCo's
# renderer, which is how make_render_fn gets away with a thin cosmetic slab
# there). Fix: compile a SECOND model, identical except for a much thinner
# CLOTH_RADIUS, and each frame copy the real sim's qpos into it before
# rendering -- same physics elsewhere, thinner cloth only in this cosmetic copy.
RENDER_CLOTH_RADIUS = 0.0015


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=1)
    ap.add_argument("--seed-base", type=int, default=0)
    ap.add_argument("--out", default=str(REPO / "renders" / "quarter_fold.mp4"))
    ap.add_argument("--width", type=int, default=960)
    ap.add_argument("--height", type=int, default=540)
    ap.add_argument("--azimuth", type=float, default=135.0, help="degrees around z, 0 = +x")
    ap.add_argument("--elevation", type=float, default=-45.0, help="degrees, negative = looking down")
    ap.add_argument("--distance", type=float, default=0.9, help="metres from lookat")
    args = ap.parse_args()

    env = QuarterFoldEnv()
    base = env.unwrapped
    base.domain_randomization = True   # matches quarter_fold_expert.py's own validation harness
    expert = QuarterFoldExpert(env)

    orig_radius = sim_main.CLOTH_RADIUS
    sim_main.CLOTH_RADIUS = RENDER_CLOTH_RADIUS
    try:
        render_model = sim_main.compile_model(sim_main.ARM_TIMESTEP, base.grasp_corners)
    finally:
        sim_main.CLOTH_RADIUS = orig_radius   # only the render copy is thin; real sim untouched
    render_data = mujoco.MjData(render_model)
    renderer = mujoco.Renderer(render_model, height=args.height, width=args.width)

    # a free (non-model) camera instead of the fixed "main" camera, so the
    # angle is a CLI param instead of baked into the compiled model
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = sim_main.CAMERA_TARGET
    cam.distance = args.distance
    cam.azimuth = args.azimuth
    cam.elevation = args.elevation

    frame_dir = Path(tempfile.mkdtemp(prefix="quarter_fold_frames_"))
    frame_idx = 0

    def capture():
        nonlocal frame_idx
        render_data.qpos[:] = base.data.qpos
        mujoco.mj_forward(render_model, render_data)
        renderer.update_scene(render_data, camera=cam)
        imageio.imwrite(frame_dir / f"frame_{frame_idx:06d}.png", renderer.render())
        frame_idx += 1

    for i in range(args.episodes):
        seed = args.seed_base + i
        obs, info = env.reset(seed=seed)
        expert.reset()
        capture()
        for t in range(env.unwrapped.max_episode_steps):
            obs, reward, terminated, truncated, info = env.step(expert.act())
            capture()
            if terminated or truncated:
                print(f"episode seed={seed}: {info['termination_reason'] or 'truncated'}, "
                      f"fold_score={info['fold_score']:.3f}, frames so far={frame_idx}")
                break

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([
        "ffmpeg", "-y", "-framerate", str(NATIVE_FPS),
        "-i", str(frame_dir / "frame_%06d.png"),
        "-r", str(OUT_FPS), "-pix_fmt", "yuv420p", str(out_path),
    ], check=True)
    shutil.rmtree(frame_dir)
    print(f"saved -> {out_path}")


if __name__ == "__main__":
    main()
