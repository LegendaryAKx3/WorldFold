"""Train an actor-critic in imagination on the fold-environment world model, then
evaluate it in the real environment against the expert, PPO and a BC control.

    .venv/bin/python scripts/train_imagined_actor.py --data outputs/cloth_angles/fold_state_v2

Policies evaluated on the same seeds:
    expert        scripted FoldExpert
    ppo           outputs/cloth_fold_rl/run2/best.zip, deterministic
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

STATE_DIM, ACTION_DIM = 378, 6
PPO_CHECKPOINT = ROOT / "outputs/cloth_fold_rl/run2/best.zip"


def load_train(data):
    episodes = [ep for ep in StateEpisodeStore(data).load_all() if ep.metadata["split"] == "train"]
    states = [ep.states() for ep in episodes]
    cur = np.concatenate([s[:-1] for s in states])
    prev = np.concatenate([np.concatenate([s[:1], s[:-2]]) for s in states])
    act = np.concatenate([ep.actions for ep in episodes])
    nxt = np.concatenate([s[1:] for s in states])
    goal = np.concatenate([np.repeat(np.array(ep.metadata["goal"], np.float32)[None], len(ep), 0) for ep in episodes])
    anchors = np.concatenate([np.repeat(np.array(ep.metadata["anchors0"], np.float32)[None], len(ep), 0) for ep in episodes])
    return tuple(torch.as_tensor(x) for x in (cur, prev, act, nxt, goal, anchors)), len(episodes)


def train_world_model(cur, prev, act, nxt, steps, seed, path):
    if path.exists():
        model = ResidualStatePredictor(STATE_DIM, ACTION_DIM)
        model.load_state_dict(torch.load(path)["state_dict"])
        return model
    torch.manual_seed(seed)
    model = ResidualStatePredictor(STATE_DIM, ACTION_DIM, loss="l1")
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


def behavior_clone(actor, world_model, cur, act, goal, steps, seed):
    optimizer = torch.optim.Adam(actor.parameters(), lr=3e-4)
    rng = np.random.default_rng(seed)
    for step in range(1, steps + 1):
        idx = torch.as_tensor(rng.integers(0, len(cur), 512))
        features = policy_features(cur[idx], goal[idx], world_model.state_mean, world_model.state_scale)
        loss = torch.nn.functional.mse_loss(actor(features), act[idx])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step % 1000 == 0:
            print(f"bc step={step} loss={loss.item():.4f}", flush=True)


def imagine(actor, world_model, cur, prev, act, goal, anchors, steps, seed, horizon, log, bc_coef=0.0):
    torch.manual_seed(seed)
    critic = Critic(feature_dim(STATE_DIM))
    trainer = ImaginationTrainer(world_model, actor, critic, horizon=horizon, bc_coef=bc_coef)
    rng = np.random.default_rng(seed)
    history = []
    for step in range(1, steps + 1):
        idx = torch.as_tensor(rng.integers(0, len(cur), 256))
        metrics = trainer.update(cur[idx], prev[idx], goal[idx], anchors[idx], act[idx])
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

    job: (policy, checkpoint, seed, max_steps[, condition]) where condition may
    set cloth_jitter (metres, default 0.025) and mass_scale (multiplies the
    cloth's base mass before the usual 0.7 to 1.3 randomization).
    """
    policy, checkpoint, seed, max_steps, *rest = job
    condition = rest[0] if rest else {}
    torch.set_num_threads(1)
    from cloth_fold_rl.expert import FoldExpert
    from cloth_fold_rl.fold_env import SingleCornerFoldEnv

    env = SingleCornerFoldEnv(max_episode_steps=max_steps, cloth_jitter=condition.get("cloth_jitter", 0.025))
    env.unwrapped.domain_randomization = True
    base = env.unwrapped
    if condition.get("mass_scale", 1.0) != 1.0:
        for bid in base._cloth_body_ids:
            base._base_body_mass[bid] *= condition["mass_scale"]
    obs, info = env.reset(seed=seed)
    if policy == "expert":
        expert = FoldExpert(env, seed=seed)
        expert.reset()
        act = lambda: expert.act()
    elif policy == "ppo":
        from stable_baselines3 import PPO
        model = PPO.load(PPO_CHECKPOINT)
        act = lambda: model.predict(obs, deterministic=True)[0]
    else:
        saved = torch.load(checkpoint)
        actor = Actor(feature_dim(STATE_DIM), ACTION_DIM)
        actor.load_state_dict(saved["actor"])
        actor.eval()
        goal = torch.as_tensor(env._goal, dtype=torch.float32)

        def act():
            state = torch.as_tensor(observe_state(base))[None]
            with torch.no_grad():
                return actor(policy_features(state, goal[None], saved["state_mean"], saved["state_scale"]))[0].numpy()
    total, grasped, steps = 0.0, False, 0
    for _ in range(max_steps):
        obs, reward, term, trunc, info = env.step(act())
        total += reward
        grasped |= bool(info["grasped"])
        steps += 1
        if term or trunc:
            break
    env.close()
    return {"policy": policy, "seed": seed, "condition": condition, "success": bool(info["success"]),
            "fold_score": info["fold_score"], "grasped": grasped, "return": total, "steps": steps,
            "reason": info["termination_reason"]}


def main():
    ap = argparse.ArgumentParser()
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
    torch.set_num_threads(4)
    out = ROOT / args.output
    out.mkdir(parents=True, exist_ok=True)

    if not args.eval_only:
        (cur, prev, act, nxt, goal, anchors), n_episodes = load_train(ROOT / args.data)
        print(f"{n_episodes} training episodes, {len(cur)} transitions", flush=True)
        started = time.perf_counter()
        world_model = train_world_model(cur, prev, act, nxt, args.wm_steps, args.seed, out / "world_model.pt")
        print(f"world model ready in {time.perf_counter() - started:.0f}s", flush=True)

        torch.manual_seed(args.seed)
        actor = Actor(feature_dim(STATE_DIM), ACTION_DIM)
        behavior_clone(actor, world_model, cur, act, goal, args.bc_steps, args.seed)
        save_actor(actor, world_model, out / "bc.pt")
        logs = {}
        logs["imagined_bc"] = imagine(actor, world_model, cur, prev, act, goal, anchors, args.imagine_steps,
                                      args.seed, args.horizon, "imagined_bc", bc_coef=args.bc_coef)
        save_actor(actor, world_model, out / "imagined_bc.pt")

        torch.manual_seed(args.seed + 1)
        actor = Actor(feature_dim(STATE_DIM), ACTION_DIM)
        logs["imagined"] = imagine(actor, world_model, cur, prev, act, goal, anchors, args.imagine_steps,
                                   args.seed, args.horizon, "imagined")
        save_actor(actor, world_model, out / "imagined.pt")
        (out / "imagination_logs.json").write_text(json.dumps(logs))

    policies = [("expert", None), ("ppo", None), ("bc", str(out / "bc.pt")),
                ("imagined_bc", str(out / "imagined_bc.pt")), ("imagined", str(out / "imagined.pt"))]
    seeds = [args.eval_seed_base + i for i in range(args.eval_episodes)]
    jobs = [(name, ckpt, seed, 200) for name, ckpt in policies for seed in seeds]
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
                        "mean_fold_score": float(np.mean([r["fold_score"] for r in mine])),
                        "mean_return": float(np.mean([r["return"] for r in mine])),
                        "mean_steps": float(np.mean([r["steps"] for r in mine]))})
    (out / "evaluation_summary.json").write_text(json.dumps(summary, indent=2))
    for row in summary:
        print(" ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items()))


if __name__ == "__main__":
    main()
