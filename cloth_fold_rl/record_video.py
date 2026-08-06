"""Record the fold policy to an mp4, with a metrics overlay burned into the frames.

    python -m cloth_fold_rl.record_video --episodes 3
    python -m cloth_fold_rl.record_video --policy expert --out outputs/videos/expert.mp4

Renders offscreen through MuJoCo's own renderer (not mjviser), so this works
headless and does not need the viewer running.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import mujoco
from PIL import Image, ImageDraw, ImageFont

from cloth_fold_rl.fold_env import SingleCornerFoldEnv, SUCCESS_DIST
from cloth_fold_rl.expert import FoldExpert

FONT_CANDIDATES = [
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
]


def load_font(size):
    for path in FONT_CANDIDATES:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


def overlay(frame, lines, font, accent=(90, 220, 120)):
    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img, "RGBA")
    pad, lh = 14, font.size + 8
    box_h = pad * 2 + lh * len(lines)
    draw.rectangle([0, 0, 340, box_h], fill=(0, 0, 0, 150))
    for i, (text, hot) in enumerate(lines):
        draw.text((pad, pad + i * lh), text, font=font,
                  fill=accent if hot else (235, 235, 235))
    return np.asarray(img)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", choices=["ppo", "expert"], default="ppo")
    ap.add_argument("--checkpoint", default="outputs/cloth_fold_rl/run2/best.zip")
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--out", default="outputs/videos/cloth_fold_ppo.mp4")
    ap.add_argument("--width", type=int, default=960)
    ap.add_argument("--height", type=int, default=544)   # multiple of 16 for libx264
    ap.add_argument("--camera", default="main")
    ap.add_argument("--fps", type=int, default=20)   # control_dt=0.05 -> real time
    ap.add_argument("--hold-frames", type=int, default=25)  # freeze on the result
    args = ap.parse_args()

    env = SingleCornerFoldEnv()
    base = env.unwrapped

    if args.policy == "expert":
        agent = FoldExpert(env)
        label = "scripted expert"
        act = lambda obs: agent.act()          # noqa: E731
    else:
        from stable_baselines3 import PPO
        ckpt = Path(args.checkpoint)
        if not ckpt.exists():
            raise SystemExit(f"no checkpoint at {ckpt}")
        model = PPO.load(ckpt)
        agent = None
        label = f"PPO ({ckpt.name})"
        act = lambda obs: model.predict(obs, deterministic=True)[0]   # noqa: E731

    renderer = mujoco.Renderer(base.model, height=args.height, width=args.width)
    font = load_font(20)
    frames = []
    results = []

    for ep in range(args.episodes):
        obs, info = env.reset(seed=100 + ep)
        if agent is not None:
            agent.reset()
        print(f"episode {ep}: ", end="", flush=True)

        for t in range(base.max_episode_steps):
            obs, r, term, trunc, info = env.step(act(obs))
            renderer.update_scene(base.data, camera=args.camera)
            frames.append(overlay(renderer.render(), [
                (label, False),
                (f"episode {ep + 1}/{args.episodes}   step {t + 1}", False),
                (f"fold_score  {info['fold_score']:.3f}", info["fold_score"] > 0.8),
                (f"corner->goal  {info['corner_to_goal']:.3f} m "
                 f"(<{SUCCESS_DIST})", info["corner_to_goal"] < SUCCESS_DIST),
                (f"grasped  {'YES' if info['grasped'] else 'no'}", info["grasped"]),
            ], font))
            if term or trunc:
                break

        ok = bool(info["success"])
        results.append(ok)
        print(f"{'SUCCESS' if ok else 'failed'} "
              f"(score {info['fold_score']:.3f}, {t + 1} steps)")

        # hold the final frame so the result is readable
        tail = overlay(renderer.render(), [
            (label, False),
            (f"episode {ep + 1}/{args.episodes}", False),
            ("SUCCESS - cloth folded" if ok else "failed", ok),
            (f"fold_score  {info['fold_score']:.3f}", ok),
            (f"corner->goal  {info['corner_to_goal']:.3f} m", ok),
        ], font)
        frames.extend([tail] * args.hold_frames)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    import imageio.v3 as iio
    iio.imwrite(out, np.stack(frames), fps=args.fps, codec="libx264")

    n = sum(results)
    print(f"\n{len(frames)} frames @ {args.fps}fps "
          f"({len(frames)/args.fps:.1f}s) -> {out}")
    print(f"{n}/{len(results)} episodes succeeded")


if __name__ == "__main__":
    main()
