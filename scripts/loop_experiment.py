"""Iterated imagination loop: train in imagination, evaluate, collect, retrain.

The actor starts from behavior cloning on the base dataset and is trained in
imagination against a K-member world-model ensemble with a disagreement
penalty and a behavior anchor toward the cloning data (weight per round from
--bc-coef-schedule). Each round it is evaluated in the real environment under
three conditions, then used with exploration noise to collect new episodes
that join the ensemble's training data for the next round.

    .venv/bin/python scripts/loop_experiment.py --stage calibrate   # disagreement scale
    .venv/bin/python scripts/loop_experiment.py --stage run --disagreement-coef C

Every step writes its outputs under --output/round_<r>/ and is skipped on
rerun if they already exist, so the script can be resumed.
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
sys.path.insert(0, str(ROOT / "scripts"))

from cloth_angles.data.state_episode import StateEpisodeStore
from cloth_angles.model.actor_critic import Actor, Critic, ImaginationTrainer, feature_dim
from cloth_angles.model.state_predictor import ResidualStatePredictor
from collect_fold_state_episodes import collect
from train_imagined_actor import ACTION_DIM, STATE_DIM, behavior_clone, load_train, run_episode, save_actor

ACTOR = "actor"
CONDITIONS = {"in_dist": {}, "shift_offset": {"cloth_jitter": 0.06}, "shift_mass": {"mass_scale": 1.6}}


def load_many(dirs):
    parts = [load_train(Path(d))[0] for d in dirs]
    return tuple(torch.cat([p[i] for p in parts]) for i in range(6))


def train_member(job):
    dirs, seed, steps, path = job
    torch.set_num_threads(1)
    cur, prev, act, nxt, _, _ = load_many(dirs)
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
    torch.save({"state_dict": model.state_dict(), "steps": steps, "transitions": len(cur), "seed": seed}, path)
    return str(path)


def load_member(path):
    model = ResidualStatePredictor(STATE_DIM, ACTION_DIM)
    model.load_state_dict(torch.load(path)["state_dict"])
    model.eval()
    return model


def ensemble_paths(round_dir, members):
    return [round_dir / f"wm_{k}.pt" for k in range(members)]


def train_ensemble(dirs, round_dir, members, steps, seed, pool):
    paths = ensemble_paths(round_dir, members)
    jobs = [(dirs, seed * 100 + k, steps, p) for k, p in enumerate(paths) if not p.exists()]
    for done in pool.imap_unordered(train_member, jobs):
        print(f"trained {done}", flush=True)
    return [load_member(p) for p in paths]


def imagine(actor, ensemble, data, steps, seed, args, log, bc_coef=0.0, anchor_data=None):
    """anchor_data: the cloning data; with bc_coef > 0 the actor is pulled toward
    those actions on those states (sampled separately from the start states)."""
    cur, prev, act, nxt, goal, anchors = data
    batch = args.imagine_batch
    torch.manual_seed(seed)
    critic = Critic(feature_dim(STATE_DIM))
    trainer = ImaginationTrainer(ensemble[0], actor, critic, horizon=args.horizon, ensemble=ensemble,
                                 disagreement_coef=args.disagreement_coef, ood_limit=args.ood_limit, bc_coef=bc_coef)
    rng = np.random.default_rng(seed)
    history = []
    for step in range(1, steps + 1):
        idx = torch.as_tensor(rng.integers(0, len(cur), batch))
        anchor = None
        if bc_coef > 0 and anchor_data is not None:
            a_cur, _, a_act, _, a_goal, _ = anchor_data
            j = torch.as_tensor(rng.integers(0, len(a_cur), batch))
            anchor = (a_cur[j], a_goal[j], a_act[j])
        history.append(trainer.update(cur[idx], prev[idx], goal[idx], anchors[idx], anchor=anchor))
        if step % 500 == 0:
            recent = {k: float(np.mean([m[k] for m in history[-500:]])) for k in history[-1]}
            print(f"{log} step={step} " + " ".join(f"{k}={v:.3f}" for k, v in recent.items()), flush=True)
    return history


def evaluate(policies, seeds, pool, path):
    if path.exists():
        return json.loads(path.read_text())
    jobs = [(name, ckpt, seed, 200, cond) for name, ckpt in policies
            for cond in CONDITIONS.values() for seed in seeds]
    rows = list(pool.imap_unordered(run_episode, jobs))
    path.write_text(json.dumps(rows, indent=2))
    return rows


def summarize(rows):
    table = {}
    for name, cond in {(r["policy"], json.dumps(r["condition"], sort_keys=True)) for r in rows}:
        mine = [r for r in rows if r["policy"] == name and json.dumps(r["condition"], sort_keys=True) == cond]
        label = next(k for k, v in CONDITIONS.items() if json.dumps(v, sort_keys=True) == cond)
        table[(name, label)] = {"n": len(mine), "success": float(np.mean([r["success"] for r in mine])),
                                "grasp": float(np.mean([r["grasped"] for r in mine])),
                                "return": float(np.mean([r["return"] for r in mine])),
                                "steps": float(np.mean([r["steps"] for r in mine]))}
    for (name, label), v in sorted(table.items()):
        print(f"  {name:14s} {label:13s} n={v['n']} success={v['success']:.3f} grasp={v['grasp']:.2f} "
              f"return={v['return']:.2f} steps={v['steps']:.0f}", flush=True)
    return {f"{name}|{label}": v for (name, label), v in table.items()}


def collect_with_actor(checkpoint, out_dir, n, seed_base, pool):
    store = StateEpisodeStore(out_dir)
    if (out_dir / "manifest.json").exists():
        return
    manifest = []
    jobs = [("actor", seed_base + i, 200, str(checkpoint)) for i in range(n)]
    for episode in pool.imap(collect, jobs):
        episode.metadata["split"] = "train"
        episode.metadata["file"] = store.append(episode).name
        manifest.append(episode.metadata)
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"collected {n} episodes into {out_dir}: success rate "
          f"{np.mean([m['success'] for m in manifest]):.2f}", flush=True)


def calibrate(args, pool):
    """Disagreement on real transitions vs on imagined rollouts with random actions."""
    round_dir = ROOT / args.output / "round_0"
    round_dir.mkdir(parents=True, exist_ok=True)
    ensemble = train_ensemble([ROOT / args.base_data], round_dir, args.members, args.wm_steps, args.seed, pool)
    cur, prev, act, nxt, goal, anchors = load_train(ROOT / args.base_data)[0]
    idx = torch.as_tensor(np.random.default_rng(0).integers(0, len(cur), 2048))
    trainer = ImaginationTrainer(ensemble[0], Actor(feature_dim(STATE_DIM), ACTION_DIM),
                                 Critic(feature_dim(STATE_DIM)), horizon=10, ensemble=ensemble)
    with torch.no_grad():
        trainer._disagreement = []
        trainer.step(cur[idx], prev[idx], act[idx])
        real = trainer._disagreement[0]
        trainer._disagreement = []
        state, prev_state = cur[idx], prev[idx]
        for _ in range(10):
            action = torch.rand(len(idx), ACTION_DIM) * 2 - 1
            nxt_state = trainer.step(state, prev_state, action)
            prev_state, state = state, nxt_state
        random_walk = torch.stack(trainer._disagreement, 1)
    print(f"disagreement on real transitions: mean {real.mean():.4f}, 95th pct {real.quantile(0.95):.4f}")
    print(f"disagreement under random actions: step 1 {random_walk[:, 0].mean():.4f}, "
          f"step 5 {random_walk[:, 4].mean():.4f}, step 10 {random_walk[:, 9].mean():.4f}")


def run(args, pool):
    out = ROOT / args.output
    results_path = out / "results.json"
    results = json.loads(results_path.read_text()) if results_path.exists() else {}
    seeds = [args.eval_seed_base + i for i in range(args.eval_seeds)]
    collected = []
    for r in range(args.rounds + 1):
        round_dir = out / f"round_{r}"
        round_dir.mkdir(parents=True, exist_ok=True)
        dirs = [ROOT / args.base_data] + sorted({*collected, *(d for d in out.glob("collect_*") if int(d.name.split("_")[1]) < r)})
        started = time.perf_counter()
        ensemble = train_ensemble(dirs, round_dir, args.members, args.wm_steps, args.seed + r, pool)
        data = load_many(dirs)
        print(f"round {r}: ensemble on {len(dirs)} dataset(s), {len(data[0])} transitions "
              f"({time.perf_counter() - started:.0f}s)", flush=True)

        clone_data = load_train(ROOT / args.base_data)[0]
        if r == 0:
            init_path = round_dir / f"{ACTOR}_init.pt"
            if not init_path.exists():
                torch.manual_seed(args.seed)
                actor = Actor(feature_dim(STATE_DIM), ACTION_DIM)
                behavior_clone(actor, ensemble[0], clone_data[0], clone_data[2], clone_data[4], args.bc_steps, args.seed)
                save_actor(actor, ensemble[0], init_path)
            policies = [("expert", None), ("ppo", None), (f"{ACTOR}_init", str(init_path))]
            print("baseline evaluation", flush=True)
            results["baselines"] = summarize(evaluate(policies, seeds, pool, round_dir / "eval_baselines.json"))
            results_path.write_text(json.dumps(results, indent=2))

        path = round_dir / f"{ACTOR}.pt"
        if not path.exists():
            previous = (out / f"round_{r - 1}" / f"{ACTOR}.pt") if r > 0 else (round_dir / f"{ACTOR}_init.pt")
            actor = Actor(feature_dim(STATE_DIM), ACTION_DIM)
            actor.load_state_dict(torch.load(previous)["actor"])
            bc_coef = args.bc_coef_schedule[min(r, len(args.bc_coef_schedule) - 1)]
            history = imagine(actor, ensemble, data, args.imagine_steps, args.seed + r, args, f"round_{r}",
                              bc_coef=bc_coef, anchor_data=clone_data)
            save_actor(actor, ensemble[0], path)
            (round_dir / f"{ACTOR}_imagination.json").write_text(json.dumps(history))

        print(f"round {r} evaluation", flush=True)
        results[f"round_{r}"] = summarize(evaluate([(ACTOR, str(path))], seeds, pool, round_dir / "eval.json"))
        results_path.write_text(json.dumps(results, indent=2))

        if r < args.rounds:
            target = out / f"collect_{r}"
            collect_with_actor(path, target, args.collect_per_round, args.collect_seed_base + r * 10000, pool)
            collected.append(target)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["calibrate", "run"], default="run")
    ap.add_argument("--base-data", default="outputs/cloth_angles/fold_state_v2")
    ap.add_argument("--output", default="outputs/cloth_angles/loop_experiment")
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--members", type=int, default=4)
    ap.add_argument("--wm-steps", type=int, default=10000)
    ap.add_argument("--bc-steps", type=int, default=3000)
    ap.add_argument("--imagine-steps", type=int, default=3000)
    ap.add_argument("--horizon", type=int, default=10)
    ap.add_argument("--imagine-batch", type=int, default=256)
    ap.add_argument("--disagreement-coef", type=float, default=1.0)
    ap.add_argument("--ood-limit", type=float, default=12.0)
    ap.add_argument("--bc-coef-schedule", type=float, nargs="+", default=[0.0],
                    help="Behavior-anchor weight per round toward the cloning data")
    ap.add_argument("--collect-per-round", type=int, default=50)
    ap.add_argument("--collect-seed-base", type=int, default=80000)
    ap.add_argument("--eval-seeds", type=int, default=40)
    ap.add_argument("--eval-seed-base", type=int, default=70000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()
    torch.set_num_threads(4)
    with mp.get_context("spawn").Pool(args.workers) as pool:
        if args.stage == "calibrate":
            calibrate(args, pool)
        else:
            run(args, pool)


if __name__ == "__main__":
    main()
