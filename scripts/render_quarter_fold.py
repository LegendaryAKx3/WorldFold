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
import numpy as np
import sim_main

from cloth_fold_rl.expert import JOINT_DELTA_SCALE, solve_ik
from cloth_fold_rl.quarter_fold_env import QuarterFoldEnv
from cloth_fold_rl.quarter_fold_expert import QuarterFoldExpert

NATIVE_FPS = 20   # 1 / control_dt
OUT_FPS = 60
RETREAT_STEPS = 35   # extra control steps at episode end, arms parking clear of the folded cloth
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
    ap.add_argument("--seeds", type=int, nargs="+", default=None,
                    help="explicit list of seeds (overrides --episodes/--seed-base)")
    ap.add_argument("--out", default=str(REPO / "renders" / "quarter_fold.mp4"))
    ap.add_argument("--width", type=int, default=960)
    ap.add_argument("--height", type=int, default=540)
    ap.add_argument("--azimuth", type=float, default=90.0,
                    help="degrees around z; 90 faces the arms (they sit south of the cloth)")
    ap.add_argument("--elevation", type=float, default=-20.0, help="degrees, negative = looking down")
    ap.add_argument("--distance", type=float, default=1.0, help="metres from lookat")
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
    render_cloth_body_ids = [render_model.body(f"cloth_{i}").id
                             for i in range(sim_main.CLOTH_COUNT ** 2)]

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

    # cloth vertices sink a few mm into the table from soft contact even in the
    # REAL sim (sim_main.py's make_render_fn clamps this away for the live
    # viewer too) -- the original 1cm collision radius was fat enough to hide
    # it, but a thin 1.5mm render radius has no margin left, so it shows as
    # clipping through the table. Clamp the render-only copy's z so its bottom
    # never dips below the table top.
    min_z = sim_main.TABLE_TOP_Z + RENDER_CLOTH_RADIUS + 0.0005

    def capture():
        nonlocal frame_idx
        render_data.qpos[:] = base.data.qpos
        mujoco.mj_forward(render_model, render_data)   # pass 1: get world positions from the copied qpos
        # cloth qpos is local/relative (not world xyz -- verified empirically,
        # shift_cloth() only ever += deltas into it, never reads it as world
        # space), so clamp by ADDING the shortfall as a delta, not overwriting.
        for j, bid in zip(range(0, len(base._cloth_qpos_adr), 3), render_cloth_body_ids):
            world_z = render_data.xpos[bid][2]
            if world_z < min_z:
                z_adr = base._cloth_qpos_adr[j + 2]
                render_data.qpos[z_adr] += min_z - world_z
        mujoco.mj_forward(render_model, render_data)   # pass 2: apply the correction
        renderer.update_scene(render_data, camera=cam)
        imageio.imwrite(frame_dir / f"frame_{frame_idx:06d}.png", renderer.render())
        frame_idx += 1

    def retreat():
        """Lift both arms up and back toward their own base, clear of the folded
        cloth, so the final shot isn't blocked by a gripper. Bypasses env.step()
        (and its reward/termination bookkeeping) -- purely cosmetic tail footage,
        driven with the same low-level joint_delta path ClothFoldEnv.step() uses."""
        targets, q_targets = {}, {}
        for prefix in base.prefixes:
            site = base.data.site_xpos[base._site_id[prefix]].copy()
            sign = -1.0 if prefix == "left_" else 1.0
            targets[prefix] = site + np.array([sign * 0.15, -0.15, 0.2])
            q_targets[prefix], _ = solve_ik(base.model, base.data, base._site_id[prefix],
                                            base._arm_qpos_adr[prefix], base._arm_dof_adr[prefix],
                                            [base.model.joint(f"{prefix}{n}").range for n in
                                             ["shoulder_pan", "shoulder_lift", "elbow_flex",
                                              "wrist_flex", "wrist_roll"]],
                                            targets[prefix])
        for _ in range(RETREAT_STEPS):
            for prefix in base.prefixes:
                base.set_gripper(prefix, 1.0)   # open
                q_now = np.array([base.data.qpos[a] for a in base._arm_qpos_adr[prefix]])
                deltas = np.clip((q_targets[prefix] - q_now) / JOINT_DELTA_SCALE, -1.0, 1.0)
                base.apply_joint_delta(prefix, deltas)
            for _ in range(base.n_substeps):
                mujoco.mj_step(base.model, base.data)
            capture()

    seeds = args.seeds if args.seeds is not None else [args.seed_base + i for i in range(args.episodes)]
    for seed in seeds:
        obs, info = env.reset(seed=seed)
        expert.reset()
        capture()
        for t in range(env.unwrapped.max_episode_steps):
            obs, reward, terminated, truncated, info = env.step(expert.act())
            capture()
            if terminated or truncated:
                print(f"episode seed={seed}: {info['termination_reason'] or 'truncated'}, "
                      f"fold_score={info['fold_score']:.3f}, frames so far={frame_idx}")
                retreat()
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
