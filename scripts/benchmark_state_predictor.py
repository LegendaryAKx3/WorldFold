"""Residual state predictors on fold-environment data, scored against persistence
and constant-velocity extrapolation.

    .venv/bin/python scripts/benchmark_state_predictor.py --data outputs/cloth_angles/fold_state_v2

Variants: state_l1 (363 vertex coordinates + 15 robot values, L1 loss),
vertex_l1 (vertices only), state_mse. Every variant sees the current and
previous state and the current action, and predicts the change.

Evaluation uses fixed windows: ten observed transitions, then 1, 5 and 10
predicted steps fed back autoregressively, stride ten. Metrics are in
millimetres: mean absolute vertex error and the moving corner's position
error, on all windows and on windows where the corner is grasped at the
prediction origin. Results are averaged per seed with episode-window weights,
then mean and standard deviation are taken across seeds.
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

from cloth_angles.data.state_episode import GRASP_INDEX, StateEpisodeStore
from cloth_angles.model.state_predictor import ResidualStatePredictor

BURN_IN, HORIZON, STRIDE = 10, 10, 10
HORIZONS = (1, 5, 10)
VERTEX_DIM, ACTION_DIM = 363, 6
MOVING_CORNER = 10  # cloth_10, the vertex welded to the left gripper
VARIANTS = ("state_l1", "vertex_l1", "state_mse")
METRICS = ("vertex_mae", "corner_err")
BASELINES = ("persistence_vertex_mae", "persistence_corner_err", "const_velocity_vertex_mae")


def load_split(data, train_per_kind=None):
    episodes = StateEpisodeStore(data).load_all()
    train = [ep for ep in episodes if ep.metadata["split"] == "train"]
    test = [ep for ep in episodes if ep.metadata["split"] == "test"]
    if train_per_kind is not None:
        counts, kept = {}, []
        for ep in train:
            kind = ep.metadata["kind"]
            if counts.get(kind, 0) < train_per_kind:
                kept.append(ep)
                counts[kind] = counts.get(kind, 0) + 1
        train = kept
    return train, test


def sequence(ep, variant):
    states = ep.states()
    return states[:, :VERTEX_DIM] if variant.startswith("vertex") else states


def transitions(episodes, variant):
    cur, prev, act, nxt = [], [], [], []
    for ep in episodes:
        seq = sequence(ep, variant)
        cur.append(seq[:-1])
        prev.append(np.concatenate([seq[:1], seq[:-2]]))
        act.append(ep.actions)
        nxt.append(seq[1:])
    return tuple(torch.as_tensor(np.concatenate(x)) for x in (cur, prev, act, nxt))


@torch.no_grad()
def evaluate(model, variant, episodes):
    rows = []
    for index, ep in enumerate(episodes):
        starts = list(range(0, len(ep) - BURN_IN - HORIZON + 1, STRIDE))
        if not starts:
            continue
        seq = torch.as_tensor(sequence(ep, variant))
        origin = torch.stack([seq[s + BURN_IN] for s in starts])
        prev = torch.stack([seq[s + BURN_IN - 1] for s in starts])
        actions = torch.as_tensor(np.stack([ep.actions[s + BURN_IN:s + BURN_IN + HORIZON] for s in starts]))
        predicted = model.rollout(origin, prev, actions)[..., :VERTEX_DIM].reshape(len(starts), HORIZON, -1, 3)
        vertices = torch.as_tensor(ep.vertices)
        v_origin = torch.stack([vertices[s + BURN_IN] for s in starts])
        v_prev = torch.stack([vertices[s + BURN_IN - 1] for s in starts])
        grasped = torch.as_tensor(np.array([ep.robot[s + BURN_IN, GRASP_INDEX] > 0.5 for s in starts]))
        for h in HORIZONS:
            target = torch.stack([vertices[s + BURN_IN + h] for s in starts])
            err = predicted[:, h - 1] - target
            persist = v_origin - target
            metrics = {"vertex_mae": err.abs().mean((-1, -2)), "corner_err": err[:, MOVING_CORNER].norm(dim=-1),
                       "persistence_vertex_mae": persist.abs().mean((-1, -2)),
                       "persistence_corner_err": persist[:, MOVING_CORNER].norm(dim=-1),
                       "const_velocity_vertex_mae": (v_origin + h * (v_origin - v_prev) - target).abs().mean((-1, -2))}
            row = {"episode": index, "kind": ep.metadata["kind"], "horizon": h,
                   "windows": len(starts), "grasped_windows": int(grasped.sum())}
            for name, values in metrics.items():
                row[name] = values.mean().item()
                row["grasped_" + name] = values[grasped].mean().item() if grasped.any() else None
            rows.append(row)
    return rows


def train_job(job):
    variant, seed, steps, data, out, train_per_kind = job
    torch.set_num_threads(1)
    train, test = load_split(Path(data), train_per_kind)
    cur, prev, act, nxt = transitions(train, variant)
    torch.manual_seed(seed)
    model = ResidualStatePredictor(cur.shape[1], ACTION_DIM, loss=variant.split("_")[1])
    model.fit_normalizer(cur.numpy(), nxt.numpy())
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    rng = np.random.default_rng(seed)
    started = time.perf_counter()
    losses = []
    for step in range(1, steps + 1):
        idx = torch.as_tensor(rng.integers(0, len(cur), 512))
        optimizer.zero_grad(set_to_none=True)
        loss = model.loss(cur[idx], prev[idx], act[idx], nxt[idx])
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    torch.save({"state_dict": model.state_dict(), "variant": variant, "steps": steps},
               Path(out) / f"{variant}_seed{seed}.pt")
    model.eval()
    return {"variant": variant, "seed": seed, "steps": steps, "transitions": len(cur), "train_episodes": len(train),
            "elapsed_seconds": time.perf_counter() - started, "final_loss": float(np.mean(losses[-100:])),
            "test": evaluate(model, variant, test), "train": evaluate(model, variant, train)}


def weighted(rows, name, count):
    return sum(r[name] * r[count] for r in rows) / sum(r[count] for r in rows)


def summarize(results, variants):
    summary = []
    for split in ("test", "train"):
        for subset in ("all", "grasped"):
            prefix = "grasped_" if subset == "grasped" else ""
            count = "grasped_windows" if subset == "grasped" else "windows"
            for h in HORIZONS:
                row = {"split": split, "subset": subset, "horizon": h}
                for variant in variants:
                    per_seed = {m: [] for m in METRICS}
                    for run in results:
                        if run["variant"] != variant:
                            continue
                        rows = [r for r in run[split] if r["horizon"] == h and r[count] > 0]
                        if not rows:
                            continue
                        row["windows"] = sum(r[count] for r in rows)
                        for b in BASELINES:
                            row[b] = weighted(rows, prefix + b, count)
                        for m in METRICS:
                            per_seed[m].append(weighted(rows, prefix + m, count))
                    for m, values in per_seed.items():
                        if values:
                            row[f"{variant}_{m}_mean"] = float(np.mean(values))
                            row[f"{variant}_{m}_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
                summary.append(row)
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="outputs/cloth_angles/fold_state_v2")
    ap.add_argument("--output", default="outputs/cloth_angles/state_benchmark")
    ap.add_argument("--steps", type=int, default=10000)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--variants", nargs="+", default=["state_l1"], choices=VARIANTS)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--train-per-kind", type=int, default=None,
                    help="Use only the first N training episodes of each kind")
    args = ap.parse_args()
    out = ROOT / args.output
    out.mkdir(parents=True, exist_ok=True)
    jobs = [(v, s, args.steps, str(ROOT / args.data), str(out), args.train_per_kind)
            for v in args.variants for s in args.seeds]
    results = []
    with mp.get_context("spawn").Pool(min(args.workers, len(jobs))) as pool:
        for result in pool.imap_unordered(train_job, jobs):
            results.append(result)
            print(f"done {result['variant']} seed={result['seed']} loss={result['final_loss']:.4f} "
                  f"elapsed={result['elapsed_seconds']:.0f}s", flush=True)
            (out / "results.json").write_text(json.dumps(results, indent=2))
    summary = summarize(results, args.variants)
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    for row in summary:
        if "windows" not in row:
            continue
        cells = [f"{v}={1000 * row[f'{v}_vertex_mae_mean']:.3f}/{1000 * row[f'{v}_corner_err_mean']:.2f}"
                 for v in args.variants if f"{v}_vertex_mae_mean" in row]
        print(row["split"], row["subset"], f"h={row['horizon']}", f"n={row['windows']}", "vertex/corner mm:", *cells,
              f"persist={1000 * row['persistence_vertex_mae']:.3f}/{1000 * row['persistence_corner_err']:.2f}",
              f"constvel={1000 * row['const_velocity_vertex_mae']:.3f}")


if __name__ == "__main__":
    main()
