"""Collect full-state episodes from a fold environment.

    .venv/bin/python scripts/collect_fold_state_episodes.py --task quarter --output outputs/cloth_angles/quarter_state

Drives the task's scripted expert (plain, with Gaussian noise on the joint
deltas, or with the grippers forced open for RELEASE_HOLD steps part way
through a grasp), the trained single-corner PPO policy (single task only), or
a saved actor with exploration noise. Records cloth vertices, every arm's
state, actions, rewards, terminations and the task stage every step, under
the environment's physical randomization.
"""
import argparse
import json
import multiprocessing as mp
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cloth_angles.data.fold_observation import cloth_vertices, robot_state
from cloth_angles.data.state_episode import StateEpisode, StateEpisodeStore
from cloth_angles.tasks import TASKS

KINDS = ("expert", "expert_noisy", "expert_release", "ppo", "ppo_release")
EXTRA_KINDS = ("actor",)   # a saved actor with exploration noise
PPO_CHECKPOINT = ROOT / "outputs/cloth_fold_rl/run2/best.zip"
NOISE_STD = 0.3
RELEASE_HOLD = 15


def collect(job):
    task_name, kind, seed, max_steps, actor_checkpoint = job
    task = TASKS[task_name]
    env = task.make_env(max_steps)
    env.unwrapped.domain_randomization = True
    base = env.unwrapped
    rng = np.random.default_rng(seed)
    obs, info = env.reset(seed=seed)
    domain = info["domain_parameters"]
    goal, anchors0 = task.goals(env), task.corner_starts(env)
    if kind.startswith("expert"):
        expert = task.make_expert(env, seed)
        expert.reset()
        policy = lambda: expert.act()
    elif kind == "actor":
        import torch
        from cloth_angles.data.fold_observation import observe_state
        from cloth_angles.model.actor_critic import Actor, feature_dim, policy_features
        torch.set_num_threads(1)
        saved = torch.load(actor_checkpoint)
        actor = Actor(feature_dim(task), task.action_dim)
        actor.load_state_dict(saved["actor"])
        actor.eval()
        goal_t = torch.as_tensor(goal)[None]

        def policy():
            state = torch.as_tensor(observe_state(base, task))[None]
            stage = torch.tensor([task.stage(info)])
            with torch.no_grad():
                mean = actor(policy_features(state, goal_t, saved["state_mean"], saved["state_scale"], task, stage))[0].numpy()
            return np.clip(mean + rng.normal(0.0, NOISE_STD, task.action_dim), -1.0, 1.0)
    else:
        assert task.name == "single", "the PPO checkpoint is a single-corner policy"
        from stable_baselines3 import PPO
        model = PPO.load(PPO_CHECKPOINT)
        model.set_random_seed(seed)
        policy = lambda: model.predict(obs, deterministic=False)[0]
    release_after = int(rng.integers(10, 40)) if kind.endswith("release") else None
    release_step = None
    vertices = [cloth_vertices(base)]
    robot = [np.concatenate([robot_state(base, a.prefix) for a in task.arms])]
    stages = [task.stage(info)]
    actions, rewards, terminated = [], [], []
    grasp_steps = 0
    for t in range(base.max_episode_steps):
        action = np.asarray(policy(), dtype=np.float32).copy()
        if kind == "expert_noisy":
            for arm in task.arms:
                action[arm.joints] = np.clip(action[arm.joints] + rng.normal(0.0, NOISE_STD, 5), -1.0, 1.0)
        if release_after is not None and release_step is None and grasp_steps >= release_after:
            release_step = t
        if release_step is not None and t < release_step + RELEASE_HOLD:
            for arm in task.arms:
                action[arm.gripper] = 1.0
        obs, reward, term, trunc, info = env.step(action)
        actions.append(action)
        rewards.append(reward)
        terminated.append(term)
        vertices.append(cloth_vertices(base))
        robot.append(np.concatenate([robot_state(base, a.prefix) for a in task.arms]))
        stages.append(task.stage(info))
        grasp_steps += int(task.grasped(info))
        if term or trunc:
            break
    env.close()
    metadata = {"task": task.name, "kind": kind, "seed": seed, "length": len(actions), "grasp_steps": grasp_steps,
                "success": bool(info["success"]), "termination_reason": info["termination_reason"],
                "final_stage": stages[-1], "release_step": release_step, "domain": domain,
                "goal": goal.tolist(), "anchors0": anchors0.tolist()}
    return StateEpisode(np.stack(vertices), np.stack(robot), np.stack(actions), metadata,
                        rewards=np.array(rewards, dtype=np.float32), terminated=np.array(terminated, dtype=np.bool_),
                        stage=np.array(stages, dtype=np.int64))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="single", choices=list(TASKS))
    ap.add_argument("--output", default="outputs/cloth_angles/fold_state")
    ap.add_argument("--per-kind", type=int, default=12)
    ap.add_argument("--test-per-kind", type=int, default=2)
    ap.add_argument("--max-steps", type=int, default=None, help="Episode length (default: the task's)")
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--seed-base", type=int, default=20000)
    ap.add_argument("--kinds", nargs="+", default=None, choices=KINDS + EXTRA_KINDS,
                    help="Default: the expert kinds, plus the PPO kinds for the single task")
    ap.add_argument("--actor-checkpoint", default=None, help="Actor .pt for the 'actor' kind")
    args = ap.parse_args()
    kinds = args.kinds or [k for k in KINDS if args.task == "single" or k.startswith("expert")]
    out = ROOT / args.output
    store = StateEpisodeStore(out)
    if store.load_all():
        raise RuntimeError("Output directory already holds episodes; use a fresh one")
    jobs = [(args.task, kind, args.seed_base + k * 1000 + i, args.max_steps, args.actor_checkpoint)
            for k, kind in enumerate(kinds) for i in range(args.per_kind)]
    manifest = []
    with mp.get_context("spawn").Pool(args.workers) as pool:
        for index, episode in enumerate(pool.imap(collect, jobs)):
            i = index % args.per_kind
            episode.metadata["split"] = "test" if i >= args.per_kind - args.test_per_kind else "train"
            episode.metadata["file"] = store.append(episode).name
            manifest.append(episode.metadata)
            print(f"{index + 1}/{len(jobs)} {episode.metadata}", flush=True)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
