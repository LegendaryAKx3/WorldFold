"""Train the cloth-angle RSSM world model on stored episodes.

Usage:
    python -m cloth_angles.train --config cloth_angles/config.yaml

No actor, critic, reward, or termination prediction: this trains only the
world model's ability to predict next-step angle fields from action-
conditioned dynamics, per cloth_folding_dreamerv4_world_model_spec.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cloth_angles.data.episode_store import EpisodeStore
from cloth_angles.data.sequence_replay import SequenceReplay
from cloth_angles.model.checkpoint import save_checkpoint
from cloth_angles.model.world_model import WorldModel


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to config.yaml")
    return parser.parse_args()


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def split_episodes(episodes, held_out_fraction: float, seed: int):
    rng = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(episodes), generator=rng).tolist()
    n_held_out = max(1, int(round(len(episodes) * held_out_fraction))) if len(episodes) > 1 else 0
    held_out_idx = set(perm[:n_held_out])
    train = [ep for i, ep in enumerate(episodes) if i not in held_out_idx]
    held_out = [ep for i, ep in enumerate(episodes) if i in held_out_idx]
    if not train:
        raise ValueError("held_out_fraction leaves no training episodes; lower it or add episodes")
    return train, held_out


def to_tensors(batch, device):
    obs, actions, next_obs, mask, is_first = batch
    return (
        torch.as_tensor(obs, device=device),
        torch.as_tensor(actions, device=device),
        torch.as_tensor(next_obs, device=device),
        torch.as_tensor(mask, device=device),
        torch.as_tensor(is_first, device=device),
    )


def main():
    args = parse_args()
    config = load_config(args.config)
    data_cfg = config["data"]
    model_cfg = config["model"]
    train_cfg = config["train"]

    torch.manual_seed(train_cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    store = EpisodeStore(
        data_cfg["episode_dir"],
        grid_size=data_cfg["grid_size"],
        action_dim=data_cfg["action_dim"],
        angle_unit=data_cfg["angle_unit"],
        angle_convention=data_cfg["angle_convention"],
    )
    episodes = store.load_all()
    if not episodes:
        raise RuntimeError(f"no episodes found under {data_cfg['episode_dir']}")

    # Split by episode (never by transition), so held-out data tests
    # generalization to unseen trajectories, not just unseen timesteps.
    train_episodes, held_out_episodes = split_episodes(
        episodes, data_cfg["held_out_fraction"], train_cfg["seed"]
    )
    train_replay = SequenceReplay(train_episodes, seq_len=data_cfg["seq_len"], seed=train_cfg["seed"])
    held_out_replay = (
        SequenceReplay(held_out_episodes, seq_len=data_cfg["seq_len"], seed=train_cfg["seed"] + 1)
        if held_out_episodes else None
    )

    model = WorldModel(
        grid_size=data_cfg["grid_size"],
        action_dim=data_cfg["action_dim"],
        angle_convention=data_cfg["angle_convention"],
        encoder_hidden=tuple(model_cfg["encoder_hidden"]),
        h_dim=model_cfg["h_dim"],
        n_categoricals=model_cfg["n_categoricals"],
        n_classes=model_cfg["n_classes"],
        mlp_hidden=model_cfg["mlp_hidden"],
        kl_free_bits=model_cfg["kl_free_bits"],
        kl_weight=model_cfg["kl_weight"],
        huber_delta=model_cfg["huber_delta"],
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=train_cfg["learning_rate"])

    checkpoint_dir = Path(train_cfg["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    for step in range(1, train_cfg["train_steps"] + 1):
        batch = to_tensors(train_replay.sample(train_cfg["batch_size"]), device)
        output = model.train_step(optimizer, *batch, grad_clip=train_cfg["grad_clip"])

        if step % train_cfg["eval_every"] == 0 or step == 1:
            msg = (
                f"step {step}/{train_cfg['train_steps']} "
                f"loss={output.loss.item():.4f} "
                f"angle={output.angle_loss.item():.4f} "
                f"kl={output.kl_loss.item():.4f} "
                f"mae_rad={output.mae_radians.item():.4f}"
            )
            if held_out_replay is not None:
                with torch.no_grad():
                    held_out_batch = to_tensors(held_out_replay.sample(train_cfg["batch_size"]), device)
                    held_out_output = model.loss(*held_out_batch)
                msg += f" | held_out_mae_rad={held_out_output.mae_radians.item():.4f}"
            print(msg)

            save_checkpoint(
                checkpoint_dir / f"checkpoint_{step:06d}.pt",
                model=model,
                step=step,
                grid_size=data_cfg["grid_size"],
                action_dim=data_cfg["action_dim"],
                angle_unit=data_cfg["angle_unit"],
                angle_convention=data_cfg["angle_convention"],
                model_config=model_cfg,
            )

    print(f"\ntraining complete. checkpoints in {checkpoint_dir}")


if __name__ == "__main__":
    main()
