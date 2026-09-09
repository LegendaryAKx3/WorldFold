"""Train an actor-critic in imagination on a fold task's world model, then
evaluate it in the real environment against the expert, PPO and a BC control.

    .venv/bin/python scripts/train_imagined_actor.py --task quarter --data outputs/cloth_angles/quarter_state

Policies evaluated on the same seeds:
    expert        the task's scripted expert
    ppo           outputs/cloth_fold_rl/run2/best.zip, deterministic (single task only)
    bc            actor after behavior cloning on the dataset only
    imagined_bc   BC actor after imagination training
    imagined      randomly initialized actor after imagination training
"""
import argparse
import json
import multiprocessing as mp
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cloth_angles.data.fold_observation import observe_state
from cloth_angles.data.state_episode import StateEpisodeStore
from cloth_angles.model.actor_critic import Actor, Critic, ImaginationTrainer, feature_dim, policy_features
from cloth_angles.model.state_predictor import ResidualStatePredictor
from cloth_angles.tasks import TASKS

PPO_CHECKPOINT = ROOT / "outputs/cloth_fold_rl/run2/best.zip"


def load_train(data):
    """-> (cur, prev, act, nxt, goal, anchors, stage) tensors over all training transitions, n_episodes."""
    episodes = [ep for ep in StateEpisodeStore(data).load_all() if ep.metadata["split"] == "train"]
    states = [ep.states() for ep in episodes]
    cur = np.concatenate([s[:-1] for s in states])
    prev = np.concatenate([np.concatenate([s[:1], s[:-2]]) for s in states])
    act = np.concatenate([ep.actions for ep in episodes])
    nxt = np.concatenate([s[1:] for s in states])
    goal = np.concatenate([np.repeat(np.array(ep.metadata["goal"], np.float32)[None], len(ep), 0) for ep in episodes])
    anchors = np.concatenate([np.repeat(np.array(ep.metadata["anchors0"], np.float32)[None], len(ep), 0) for ep in episodes])
    stage = np.concatenate([ep.stage[:-1] if ep.stage is not None else np.zeros(len(ep), np.int64) for ep in episodes])
    return tuple(torch.as_tensor(x) for x in (cur, prev, act, nxt, goal, anchors, stage)), len(episodes)


def train_world_model(task, cur, prev, act, nxt, steps, seed, path):
    if path.exists():
        model = ResidualStatePredictor(task.state_dim, task.action_dim)
        model.load_state_dict(torch.load(path)["state_dict"])
        return model
    torch.manual_seed(seed)
    model = ResidualStatePredictor(task.state_dim, task.action_dim, loss="l1")
    model.fit_normalizer(cur.numpy(), nxt.numpy())
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    rng = np.random.default_rng(seed)
    for step in range(1, steps + 1):
        idx = torch.as_tensor(rng.integers(0, len(cur), 512))
        optimizer.zero_grad(set_to_none=True)
        loss = model.loss(cur[idx], prev[idx], act[idx], nxt[idx])
        loss.backward()
        optimizer.step()
        if step % 2000 == 0:
            print(f"world model step={step} loss={loss.item():.4f}", flush=True)
    torch.save({"state_dict": model.state_dict(), "steps": steps}, path)
    return model


def behavior_clone(task, actor, world_model, data, steps, seed):
    cur, _, act, _, goal, _, stage = data
    optimizer = torch.optim.Adam(actor.parameters(), lr=3e-4)
    rng = np.random.default_rng(seed)
    for step in range(1, steps + 1):
        idx = torch.as_tensor(rng.integers(0, len(cur), 512))
        features = policy_features(cur[idx], goal[idx], world_model.state_mean, world_model.state_scale, task, stage[idx])
        loss = torch.nn.functional.mse_loss(actor(features), act[idx])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step % 1000 == 0:
            print(f"bc step={step} loss={loss.item():.4f}", flush=True)


def imagine(task, actor, world_model, data, steps, seed, horizon, log, bc_coef=0.0):
    cur, prev, act, _, goal, anchors, stage = data
    torch.manual_seed(seed)
    critic = Critic(feature_dim(task))
    trainer = ImaginationTrainer(world_model, actor, critic, task, horizon=horizon, bc_coef=bc_coef)
    rng = np.random.default_rng(seed)
    history = []
    for step in range(1, steps + 1):
        idx = torch.as_tensor(rng.integers(0, len(cur), 256))
        anchor = (cur[idx], goal[idx], stage[idx], act[idx]) if bc_coef > 0 else None
        metrics = trainer.update(cur[idx], prev[idx], goal[idx], anchors[idx], stage[idx], anchor=anchor)
        history.append(metrics)
        if step % 500 == 0:
            recent = {k: float(np.mean([m[k] for m in history[-500:]])) for k in metrics}
            print(f"{log} step={step} " + " ".join(f"{k}={v:.3f}" for k, v in recent.items()), flush=True)
    return history


def save_actor(actor, world_model, path):
    torch.save({"actor": actor.state_dict(), "state_mean": world_model.state_mean,
                "state_scale": world_model.state_scale}, path)


def run_episode(job):
    """One real-environment episode; returns summary metrics. Runs in a worker.

    job: (task, policy, checkpoint, seed[, condition]) where condition may set
    cloth_jitter (metres, default 0.025) and mass_scale (multiplies the cloth's
    base mass before the usual 0.7 to 1.3 randomization).
    """
    task_name, policy, checkpoint, seed, *rest = job
    task = TASKS[task_name]
    condition = rest[0] if rest else {}
    torch.set_num_threads(1)
    env = task.make_env(cloth_jitter=condition.get("cloth_jitter", 0.025))
    env.unwrapped.domain_randomization = True
    base = env.unwrapped
    if condition.get("mass_scale", 1.0) != 1.0:
        for bid in base._cloth_body_ids:
            base._base_body_mass[bid] *= condition["mass_scale"]
    obs, info = env.reset(seed=seed)
    if policy == "expert":
        expert = task.make_expert(env, seed)
        expert.reset()
        act = lambda: expert.act()
    elif policy == "ppo":
        from stable_baselines3 import PPO
        model = PPO.load(PPO_CHECKPOINT)
        act = lambda: model.predict(obs, deterministic=True)[0]
    else:
        saved = torch.load(checkpoint)
        actor = Actor(feature_dim(task), task.action_dim)
        actor.load_state_dict(saved["actor"])
        actor.eval()
        goal = torch.as_tensor(task.goals(env))[None]

        def act():
            state = torch.as_tensor(observe_state(base, task))[None]
            stage = torch.tensor([task.stage(info)])
            with torch.no_grad():
                return actor(policy_features(state, goal, saved["state_mean"], saved["state_scale"], task, stage))[0].numpy()
    total, grasped, steps = 0.0, False, 0
    for _ in range(base.max_episode_steps):
        obs, reward, term, trunc, info = env.step(act())
        total += reward
        grasped |= task.grasped(info)
        steps += 1
        if term or trunc:
            break
    env.close()
    return {"policy": policy, "seed": seed, "condition": condition, "success": bool(info["success"]),
            "fold_score": info["fold_score"], "grasped": grasped, "stage": task.stage(info), "return": total,
            "steps": steps, "reason": info["termination_reason"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="single", choices=list(TASKS))
    ap.add_argument("--data", default="outputs/cloth_angles/fold_state_v2")
    ap.add_argument("--output", default="outputs/cloth_angles/imagined_actor")
    ap.add_argument("--wm-steps", type=int, default=20000)
    ap.add_argument("--bc-steps", type=int, default=3000)
    ap.add_argument("--imagine-steps", type=int, default=3000)
    ap.add_argument("--horizon", type=int, default=10)
    ap.add_argument("--bc-coef", type=float, default=1.0,
                    help="Behavior regularization weight for the BC-initialized actor")
    ap.add_argument("--eval-episodes", type=int, default=20)
    ap.add_argument("--eval-seed-base", type=int, default=50000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--eval-only", action="store_true")
    args = ap.parse_args()
    task = TASKS[args.task]
    torch.set_num_threads(4)
    out = ROOT / args.output
    out.mkdir(parents=True, exist_ok=True)

    if not args.eval_only:
        data, n_episodes = load_train(ROOT / args.data)
        cur, prev, act, nxt = data[:4]
        print(f"{n_episodes} training episodes, {len(cur)} transitions", flush=True)
        started = time.perf_counter()
        world_model = train_world_model(task, cur, prev, act, nxt, args.wm_steps, args.seed, out / "world_model.pt")
        print(f"world model ready in {time.perf_counter() - started:.0f}s", flush=True)

        torch.manual_seed(args.seed)
        actor = Actor(feature_dim(task), task.action_dim)
        behavior_clone(task, actor, world_model, data, args.bc_steps, args.seed)
        save_actor(actor, world_model, out / "bc.pt")
        logs = {}
        logs["imagined_bc"] = imagine(task, actor, world_model, data, args.imagine_steps, args.seed, args.horizon,
                                      "imagined_bc", bc_coef=args.bc_coef)
        save_actor(actor, world_model, out / "imagined_bc.pt")

        torch.manual_seed(args.seed + 1)
        actor = Actor(feature_dim(task), task.action_dim)
        logs["imagined"] = imagine(task, actor, world_model, data, args.imagine_steps, args.seed, args.horizon, "imagined")
        save_actor(actor, world_model, out / "imagined.pt")
        (out / "imagination_logs.json").write_text(json.dumps(logs))

    policies = [("expert", None)] + ([("ppo", None)] if task.name == "single" else []) + [
        ("bc", str(out / "bc.pt")), ("imagined_bc", str(out / "imagined_bc.pt")), ("imagined", str(out / "imagined.pt"))]
    seeds = [args.eval_seed_base + i for i in range(args.eval_episodes)]
    jobs = [(task.name, name, ckpt, seed) for name, ckpt in policies for seed in seeds]
    rows = []
    with mp.get_context("spawn").Pool(args.workers) as pool:
        for row in pool.imap_unordered(run_episode, jobs):
            rows.append(row)
    (out / "evaluation.json").write_text(json.dumps(rows, indent=2))
    summary = []
    for name, _ in policies:
        mine = [r for r in rows if r["policy"] == name]
        summary.append({"policy": name, "episodes": len(mine),
                        "success_rate": float(np.mean([r["success"] for r in mine])),
                        "grasp_rate": float(np.mean([r["grasped"] for r in mine])),
                        "mean_stage": float(np.mean([r["stage"] for r in mine])),
                        "mean_fold_score": float(np.mean([r["fold_score"] for r in mine])),
                        "mean_return": float(np.mean([r["return"] for r in mine])),
                        "mean_steps": float(np.mean([r["steps"] for r in mine]))})
    (out / "evaluation_summary.json").write_text(json.dumps(summary, indent=2))
    for row in summary:
        print(" ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items()))


if __name__ == "__main__":
    main()
