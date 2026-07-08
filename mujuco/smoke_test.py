# this just runs a test to verify if the environments are working, and saves a render frame for each env to proof

import sys
import traceback
from pathlib import Path

import gymnasium as gym
import numpy as np

CPU_ENV_IDS = [
    "MuJoCoTouch-v1",
    "MuJoCoLookAt-v1",
    "MuJoCoMove-v1",
    "MuJoCoPickLift-v1",
    "MuJoCoPickAndPlace-v1",
]

WARP_ENV_IDS = [
    "WarpTouch-v1",
    "WarpLookAt-v1",
    "WarpMove-v1",
    "WarpPickLift-v1",
    "WarpPickAndPlace-v1",
]

N_STEPS = 20
NUM_WARP_ENVS = 8 # my laptop is RTX 5070, 8GB
PROOF_DIR = Path(__file__).parent / "proof"

def describe(obs) -> str:
    """One-line description of an observation (dict / numpy array / torch tensor)."""
    if isinstance(obs, dict):
        return "{" + ", ".join(f"{k}: {describe(v)}" for k, v in obs.items()) + "}"
    if hasattr(obs, "device"):  # torch tensor
        return f"shape={tuple(obs.shape)} dtype={obs.dtype} device={obs.device}"
    arr = np.asarray(obs)
    return f"shape={arr.shape} dtype={arr.dtype}"

# cpu env
def run_cpu_env(env_id: str) -> bool:
    print(f"\n=== {env_id} (CPU / MuJoCo) ===")
    env = gym.make(env_id, render_mode="rgb_array")
    try:
        print(f"action space:      {env.action_space}")
        print(f"observation space: {env.observation_space}")

        obs, info = env.reset(seed=0)
        print(f"reset -> obs {describe(obs)}")
        if info:
            print(f"reset -> info keys: {sorted(info)}")

        total_reward = 0.0
        for step in range(N_STEPS):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += float(reward)
            if terminated or truncated:
                print(f"episode ended at step {step + 1} (terminated={terminated}, truncated={truncated}); resetting")
                obs, info = env.reset()
        print(f"stepped {N_STEPS}x with random actions, total reward: {total_reward:.4f}")
        print(f"last step -> reward={float(reward):.4f} terminated={terminated} truncated={truncated}")

        frame = env.render()
        if frame is not None:
            frame = np.asarray(frame)
            PROOF_DIR.mkdir(exist_ok=True)
            out = PROOF_DIR / f"{env_id}.png"
            import imageio.v3 as iio

            iio.imwrite(out, frame)
            print(f"saved render frame {frame.shape} -> {out}")
        print(f"PASS {env_id}")
        return True
    except Exception:
        traceback.print_exc()
        print(f"FAIL {env_id}")
        return False
    finally:
        env.close()

# gpu env
def run_warp_env(env_id: str) -> bool:
    print(f"\n=== {env_id} (GPU / Warp, num_envs={NUM_WARP_ENVS}) ===")
    envs = None
    try:
        envs = gym.make_vec(env_id, num_envs=NUM_WARP_ENVS, device="cuda", seed=0)
        print(f"action space:      {envs.action_space}")
        print(f"observation space: {envs.observation_space}")

        obs, info = envs.reset(seed=0)
        print(f"reset -> obs {describe(obs)}")

        total_reward = 0.0
        for _ in range(N_STEPS):
            actions = envs.action_space.sample()  # batched numpy; env converts to torch
            obs, rewards, terminated, truncated, info = envs.step(actions)
            total_reward += float(rewards.float().mean())
        print(f"stepped {N_STEPS}x with random actions, mean reward/step: {total_reward / N_STEPS:.4f}")
        print(
            f"last step -> rewards {describe(rewards)}, "
            f"terminated {int(terminated.sum())}/{NUM_WARP_ENVS}, "
            f"truncated {int(truncated.sum())}/{NUM_WARP_ENVS}"
        )
        print(f"PASS {env_id}")
        return True
    except Exception:
        traceback.print_exc()
        print(f"FAIL {env_id}")
        return False
    finally:
        if envs is not None:
            envs.close()

def main() -> int:
    argv = sys.argv[1:]
    only_cpu = "--cpu" in argv
    only_warp = "--warp" in argv
    ids = [a for a in argv if not a.startswith("--")]

    if not ids:
        ids = []
        if only_cpu or not only_warp:
            ids += CPU_ENV_IDS
        if only_warp or not only_cpu:
            ids += WARP_ENV_IDS

    results = {}
    for env_id in ids:
        if env_id.startswith("Warp"):
            import so101_nexus.warp  

            results[env_id] = run_warp_env(env_id)
        else:
            import so101_nexus.mujoco 

            results[env_id] = run_cpu_env(env_id)

    print("\n=== summary ===")
    for env_id, ok in results.items():
        print(f"{'PASS' if ok else 'FAIL'}  {env_id}")
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
