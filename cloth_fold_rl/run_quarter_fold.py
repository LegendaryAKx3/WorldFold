"""Watch the quarter-fold scripted expert in the mjviser viewer.

There is no trained PPO checkpoint for this task yet -- only QuarterFoldExpert
exists, so this always runs the scripted phase machine, not a learned policy.

    python -m cloth_fold_rl.run_quarter_fold

Mirrors run_trained.py's structure (env.step() driven, live metrics panel,
auto-reset at episode end) but for QuarterFoldEnv's two-arm, two-stage task.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import mjviser

from cloth_fold_rl.quarter_fold_env import QuarterFoldEnv
from cloth_fold_rl.quarter_fold_expert import QuarterFoldExpert

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "mujuco"))
from sim_main import make_render_fn  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    env = QuarterFoldEnv()
    base = env.unwrapped

    expert = QuarterFoldExpert(env)

    obs, info = env.reset(seed=args.seed)
    expert.reset()

    state = {"substep": 0, "ep": 0, "last": dict(info), "reward": 0.0, "result": "running"}

    def policy_action():
        return expert.act()

    def step_fn(model_, data_):
        if state["substep"] == 0:
            action = policy_action()
            obs, r, term, trunc, info = env.step(action)
            state.update(last=dict(info))
            state["reward"] += r
            if term or trunc:
                state["result"] = ("SUCCESS" if info["success"]
                                    else f"failed ({info['termination_reason'] or 'time limit'})")
                state["ep"] += 1
                env.reset(seed=state["ep"])
                expert.reset()
                state.update(reward=0.0)
        state["substep"] = (state["substep"] + 1) % base.n_substeps

    def reset_fn(model_, data_):
        env.reset(seed=state["ep"])
        expert.reset()
        state.update(substep=0, reward=0.0, result="running")

    base_render = make_render_fn(base.model, base.data)
    panel = {"handle": None}

    def render_fn(scene):
        base_render(scene)
        info = state["last"]
        phases = expert.phases()
        d = info.get("move_distance", [])
        text = (
            f"### quarter fold (scripted expert)\n\n"
            f"| | |\n|---|---|\n"
            f"| **stage** | {info.get('stage', 0)} |\n"
            f"| **left phase** | {phases.get('left_', '-')} |\n"
            f"| **right phase** | {phases.get('right_', '-')} |\n"
            f"| **fold_score** | {info.get('fold_score', 0):.3f} |\n"
            f"| **move dist** | {' / '.join(f'{x:.3f}' for x in d)} m |\n"
            f"| **anchor drift** | {info.get('anchor_drift', 0):.3f} m |\n"
            f"| **return** | {state['reward']:.1f} |\n"
            f"| **episode** | {state['ep']} |\n"
            f"| **last result** | {state['result']} |\n"
        )
        if panel["handle"] is None:
            panel["handle"] = scene.server.gui.add_markdown(text)
        else:
            panel["handle"].content = text

    print("running quarter-fold scripted expert -- open the viewer URL below")
    mjviser.Viewer(base.model, base.data, step_fn=step_fn, reset_fn=reset_fn,
                   render_fn=render_fn).run()


if __name__ == "__main__":
    main()
