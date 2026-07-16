#!/usr/bin/env python3
"""Smoke test the MolmoAct2 import path for SO101-Nexus environments."""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import inspect
import json
import platform
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import gymnasium as gym
import numpy as np


DEFAULT_OUTPUT = Path("outputs/molmoact_import/probe.json")


class EchoMolmoModel:
    """Tiny local stand-in for MolmoAct2 that exercises the adapter contract."""

    def __init__(self, chunk_size: int) -> None:
        self.chunk_size = chunk_size

    def predict_action(self, **kwargs: Any) -> SimpleNamespace:
        state = np.asarray(kwargs["state"], dtype=np.float32).reshape(1, 1, -1)
        actions = np.repeat(state, self.chunk_size, axis=1)
        return SimpleNamespace(actions=actions)


class PredictActionCompatModel:
    """Map older adapter keyword names onto the downloaded model API."""

    def __init__(self, model: Any) -> None:
        self._model = model

    def __getattr__(self, name: str) -> Any:
        return getattr(self._model, name)

    def predict_action(self, **kwargs: Any) -> Any:
        if "action_mode" in kwargs and "inference_action_mode" not in kwargs:
            kwargs["inference_action_mode"] = kwargs.pop("action_mode")
        return self._model.predict_action(**kwargs)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env", default="MuJoCoTouch-v1")
    parser.add_argument("--repo-id", default="allenai/MolmoAct2-SO100_101")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--width", type=int, default=160)
    parser.add_argument("--height", type=int, default=120)
    parser.add_argument("--chunk-size", type=int, default=8)
    parser.add_argument("--num-steps", type=int, default=10)
    parser.add_argument("--disable-cuda-graph", action="store_true")
    parser.add_argument("--load-model", action="store_true")
    parser.add_argument("--device-map", default=None)
    parser.add_argument(
        "--dtype",
        choices=("none", "float32", "float16", "bfloat16"),
        default="none",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def import_backend_for_env_id(env_id: str) -> None:
    if env_id.startswith("MuJoCo"):
        importlib.import_module("so101_nexus.mujoco")
        return
    if env_id.startswith("Warp"):
        importlib.import_module("so101_nexus.warp")
        return
    raise ValueError(f"Unsupported SO101 env id: {env_id!r}")


def visual_config_for_env(env_id: str, width: int, height: int):
    from so101_nexus.config import LookAtConfig, MoveConfig, PickAndPlaceConfig, PickConfig, TouchConfig
    from so101_nexus.observations import JointPositions, OverheadCamera, WristCamera

    observations = [
        JointPositions(),
        OverheadCamera(width=width, height=height),
        WristCamera(width=width, height=height),
    ]
    kwargs = {"obs_mode": "visual", "observations": observations}
    if "PickAndPlace" in env_id:
        return PickAndPlaceConfig(**kwargs)
    if "PickLift" in env_id:
        return PickConfig(**kwargs)
    if "Touch" in env_id:
        return TouchConfig(**kwargs)
    if "LookAt" in env_id:
        return LookAtConfig(**kwargs)
    if "Move" in env_id:
        return MoveConfig(**kwargs)
    raise ValueError(f"No visual config mapping for {env_id!r}")


def module_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def dtype_from_name(name: str):
    if name == "none":
        return None
    import torch

    return getattr(torch, name)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_jsonable(payload), indent=2, sort_keys=True) + "\n")


def to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def make_env(env_id: str, width: int, height: int):
    config = visual_config_for_env(env_id, width=width, height=height)
    return gym.make(env_id, config=config, render_mode="rgb_array", control_mode="pd_joint_pos")


def describe_visual_obs(obs: dict[str, np.ndarray]) -> dict[str, Any]:
    return {
        "keys": sorted(obs.keys()),
        "state_shape": list(obs["state"].shape),
        "overhead_camera_shape": list(obs["overhead_camera"].shape),
        "wrist_camera_shape": list(obs["wrist_camera"].shape),
    }


def apply_predict_action_compat(policy: Any) -> bool:
    """Return True when the model needed the action-mode keyword shim."""
    signature = inspect.signature(policy.model.predict_action)
    if "action_mode" in signature.parameters:
        return False
    if "inference_action_mode" not in signature.parameters:
        return False
    policy.model = PredictActionCompatModel(policy.model)
    return True


def main() -> int:
    args = parse_args()
    payload: dict[str, Any] = {
        "env": args.env,
        "repo_id": args.repo_id,
        "python": sys.version,
        "platform": platform.platform(),
        "checks": {
            "gymnasium": module_available("gymnasium"),
            "huggingface_hub": module_available("huggingface_hub"),
            "so101_nexus": module_available("so101_nexus"),
            "so101_nexus_policy_adapters": module_available("so101_nexus.policy_adapters"),
            "torch": module_available("torch"),
            "transformers": module_available("transformers"),
        },
        "status": "not_started",
    }

    try:
        import_backend_for_env_id(args.env)
        from so101_nexus.policy_adapters import MolmoActPolicy, RolloutRecorder

        env = make_env(args.env, width=args.width, height=args.height)
        try:
            obs, _ = env.reset(seed=args.seed)
            payload["visual_observation"] = describe_visual_obs(obs)

            if args.load_model:
                policy = MolmoActPolicy.from_pretrained(
                    args.repo_id,
                    device=args.device_map,
                    dtype=dtype_from_name(args.dtype),
                    chunk_size=args.chunk_size,
                    num_steps=args.num_steps,
                    enable_cuda_graph=not args.disable_cuda_graph,
                )
                payload["predict_action_compat"] = apply_predict_action_compat(policy)
            else:
                policy = MolmoActPolicy(
                    EchoMolmoModel(chunk_size=args.chunk_size),
                    processor=None,
                    chunk_size=args.chunk_size,
                )
                payload["predict_action_compat"] = False

            recorder = RolloutRecorder(
                env,
                policy,
                max_steps_per_episode=args.max_steps,
            )
            results = recorder.record_episodes(args.episodes, seed=args.seed)
        finally:
            env.close()

        payload["status"] = "model_rollout_ran" if args.load_model else "adapter_dry_run_ran"
        payload["episodes"] = [
            {
                "episode": i,
                "steps": result.n_steps,
                "success": result.success,
                "actions_shape": list(result.actions_deg.shape),
                "states_shape": list(result.states_deg.shape),
                "info": result.info,
            }
            for i, result in enumerate(results)
        ]
        write_json(args.output, payload)
        print(f"{payload['status']}: wrote {args.output}")
        return 0
    except Exception as exc:  # noqa: BLE001 - this is a smoke-test/probe script.
        payload["status"] = "blocked"
        payload["blocker"] = f"{type(exc).__name__}: {exc}"
        write_json(args.output, payload)
        print(f"blocked: {payload['blocker']}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
