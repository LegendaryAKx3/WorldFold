"""Run a trained fold policy (or the scripted expert) in the mjviser viewer,
with a live readout of the metrics sim_main.py's viewer never showed.

    python -m cloth_fold_rl.run_trained                      # best checkpoint
    python -m cloth_fold_rl.run_trained --policy expert      # scripted expert
    python -m cloth_fold_rl.run_trained --checkpoint path.zip

Unlike sim_main.main(), this drives the env through env.step(), so fold_score,
grasp state and success are actually computed -- and it auto-resets at the end
of an episode instead of freezing in whatever pose it ended in.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import mjviser

from cloth_fold_rl.fold_env import make_fold_env, make_expert, SUCCESS_DIST

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "mujuco"))
from sim_main import make_render_fn  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", choices=["ppo", "expert"], default="ppo")
    ap.add_argument("--checkpoint", default="outputs/cloth_fold_rl/run1/best.zip")
    ap.add_argument("--max-episode-steps", type=int, default=None,
                    help="default 200 (weld) / 250 (physical)")
    ap.add_argument("--physical", action="store_true", help="use the physical grabber (plates, no weld) -- see physical_env.py")
    args = ap.parse_args()

    env = make_fold_env(args.physical, max_episode_steps=args.max_episode_steps)
    base = env.unwrapped

    expert = None
    model = None
    if args.policy == "expert":
        expert = make_expert(env, args.physical)
        label = "scripted expert" + (" (physical grasp)" if args.physical else "")
    else:
        ckpt = Path(args.checkpoint)
        if not ckpt.exists():
            raise SystemExit(
                f"no checkpoint at {ckpt}.\n"
                f"Train one:  python -m cloth_fold_rl.train\n"
                f"Or watch the scripted expert:  "
                f"python -m cloth_fold_rl.run_trained --policy expert")
        from stable_baselines3 import PPO
        model = PPO.load(ckpt)
        label = f"PPO ({ckpt.name})"

    obs, info = env.reset(seed=0)
    if expert:
        expert.reset()

    state = {"obs": obs, "substep": 0, "ep": 0, "last": dict(info),
             "reward": 0.0, "result": "running"}

    def policy_action():
        if expert is not None:
            return expert.act()
        action, _ = model.predict(state["obs"], deterministic=True)
        return action

    def step_fn(model_, data_):
        # one env.step() == n_substeps physics steps; feed the viewer one at a time
        if state["substep"] == 0:
            action = policy_action()
            obs, r, term, trunc, info = env.step(action)
            state.update(obs=obs, last=dict(info))
            state["reward"] += r
            if term or trunc:
                state["result"] = (
                    "SUCCESS" if info["success"]
                    else f"failed ({info['termination_reason'] or 'time limit'})")
                state["ep"] += 1
                o, i = env.reset(seed=state["ep"])
                if expert:
                    expert.reset()
                state.update(obs=o, last=dict(i), reward=0.0)
        state["substep"] = (state["substep"] + 1) % base.n_substeps

    def reset_fn(model_, data_):
        o, i = env.reset(seed=state["ep"])
        if expert:
            expert.reset()
        state.update(obs=o, last=dict(i), substep=0, reward=0.0, result="running")

    base_render = make_render_fn(base.model, base.data)
    panel = {"handle": None}

    def render_fn(scene):
        base_render(scene)
        info = state["last"]
        text = (
            f"### {label}\n\n"
            f"| | |\n|---|---|\n"
            f"| **fold_score** | {info.get('fold_score', 0):.3f} |\n"
            f"| **corner -> goal** | {info.get('corner_to_goal', 0):.3f} m "
            f"(need < {SUCCESS_DIST}) |\n"
            f"| **grasped** | {'YES' if info.get('grasped') else 'no'} |\n"
            f"| **anchor drift** | {info.get('anchor_drift', 0):.3f} m |\n"
            f"| **return** | {state['reward']:.1f} |\n"
            f"| **episode** | {state['ep']} |\n"
            f"| **last result** | {state['result']} |\n"
        )
        if panel["handle"] is None:
            panel["handle"] = scene.server.gui.add_markdown(text)
        else:
            panel["handle"].content = text

    print(f"running {label} -- open the viewer URL below")
    mjviser.Viewer(base.model, base.data, step_fn=step_fn, reset_fn=reset_fn,
                   render_fn=render_fn).run()


if __name__ == "__main__":
    main()
