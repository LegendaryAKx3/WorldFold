"""Record the physical-grabber proof of concept to an mp4.

Same routine as prove_grabber.py, with frames captured from the "main" camera:

    python mujuco/record_grabber_demo.py [--output outputs/videos/grabber_poc.mp4]
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import imageio
import mujoco
import numpy as np

from prove_grabber import GrabberDemo, MIN_PEAK_LIFT


class RecordingDemo(GrabberDemo):

    def __init__(self, every=400, width=640, height=360):
        super().__init__()
        self.frames = []
        self.every = every
        self._count = 0
        self._renderer = mujoco.Renderer(self.m, height=height, width=width)

    def run_to(self, target, grip, steps):
        self.env._target_pos[self.p] = np.array(target, dtype=float)
        for _ in range(steps):
            self.env.ik_substep(self.p)
            self.env.ik_substep("right_")
            self.d.ctrl[self.grip_act] = grip
            mujoco.mj_step(self.m, self.d)
            self._count += 1
            if self._count % self.every == 0:
                self._renderer.update_scene(self.d, camera="main")
                self.frames.append(self._renderer.render())
        assert self.d.eq_active[self.weld_id] == 0, "weld engaged -- not a physical grasp"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="outputs/videos/grabber_poc.mp4")
    parser.add_argument("--fps", type=int, default=25)
    args = parser.parse_args()

    demo = RecordingDemo()
    result = demo.run()

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimwrite(out, demo.frames, fps=args.fps)
    print(f"\nwrote {len(demo.frames)} frames to {out}")
    print(f"peak lift {result['peak_lift']:.3f}m, carry {result['carry_dist']:.3f}m, "
          f"fold progress {result['fold_progress']:.3f}m "
          f"({'grasp held' if result['peak_lift'] >= MIN_PEAK_LIFT else 'grasp failed'})")


if __name__ == "__main__":
    main()
