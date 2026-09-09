"""Behavior-clone the scripted expert into an SB3 PPO policy.

Produces a .zip that `train.py --init-from` can pick up, so PPO fine-tunes a
policy that already folds instead of rediscovering grasping from scratch.

    python -m cloth_fold_rl.bc --epochs 30
    python -m cloth_fold_rl.train --run-dir outputs/cloth_fold_rl/run2 \
        --init-from outputs/cloth_fold_rl/bc.zip

Two details that matter:

* Only the ACTOR is cloned. PPO relearns the critic, which is cheap -- but it
  means the first fine-tuning round will show a large value_loss. That is
  expected, not a regression.
* log_std is shrunk from its default (std 1.0) to `--init-std`. At std 1.0 the
  exploration noise on a 6-D action completely swamps the cloned mean, and PPO
  destroys the imitated behaviour within one update.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from cloth_fold_rl.fold_env import SingleCornerFoldEnv


def action_mean(policy, obs):
    """Mean of the policy's action distribution -- what BC regresses on."""
    feats = policy.extract_features(obs)
    if isinstance(feats, tuple):     # unshared feature extractors
        feats = feats[0]
    latent_pi = policy.mlp_extractor.forward_actor(feats)
    return policy.action_net(latent_pi)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demos", default="outputs/cloth_fold_rl/demos.npz")
    ap.add_argument("--out", default="outputs/cloth_fold_rl/bc.zip")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--init-std", type=float, default=0.25)
    ap.add_argument("--eval-episodes", type=int, default=6)
    args = ap.parse_args()

    d = np.load(args.demos)
    obs_np, act_np = d["obs"], d["actions"]
    print(f"loaded {len(obs_np):,} transitions  obs{obs_np.shape} act{act_np.shape}")

    n_val = max(1, int(0.1 * len(obs_np)))
    perm = np.random.default_rng(0).permutation(len(obs_np))
    val_idx, train_idx = perm[:n_val], perm[n_val:]

    obs = torch.as_tensor(obs_np, dtype=torch.float32)
    acts = torch.as_tensor(act_np, dtype=torch.float32)

    # same architecture/hyperparams as train.py, so the zip loads cleanly there
    venv = DummyVecEnv([lambda: SingleCornerFoldEnv()])
    model = PPO("MlpPolicy", venv, verbose=0, seed=0,
                learning_rate=3e-4, n_steps=512, batch_size=256, n_epochs=10,
                gamma=0.99, gae_lambda=0.95, clip_range=0.2, ent_coef=0.005,
                policy_kwargs=dict(net_arch=[256, 256]))
    policy = model.policy
    opt = torch.optim.Adam(policy.parameters(), lr=args.lr)

    print(f"\n{'epoch':>6} {'train mse':>11} {'val mse':>11}")
    for ep in range(args.epochs):
        policy.train()
        idx = train_idx[torch.randperm(len(train_idx)).numpy()]
        tot, nb = 0.0, 0
        for s in range(0, len(idx), args.batch_size):
            b = idx[s:s + args.batch_size]
            loss = torch.nn.functional.mse_loss(action_mean(policy, obs[b]), acts[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss); nb += 1
        policy.eval()
        with torch.no_grad():
            v = float(torch.nn.functional.mse_loss(
                action_mean(policy, obs[val_idx]), acts[val_idx]))
        if ep % 5 == 0 or ep == args.epochs - 1:
            print(f"{ep:>6} {tot/nb:>11.5f} {v:>11.5f}")

    # shrink exploration noise so PPO fine-tuning does not wash out the clone
    with torch.no_grad():
        policy.log_std.data.fill_(float(np.log(args.init_std)))

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    model.save(args.out)
    print(f"\nsaved -> {args.out} (log_std set to std={args.init_std})")

    # does the cloned policy actually fold?
    from cloth_fold_rl.train import evaluate
    ev = evaluate(model, n_episodes=args.eval_episodes)
    print(f"\n=== BC policy, varied starts ===")
    print(f"success {ev['success_rate']:.0%}  grasp {ev['grasp_rate']:.0%}  "
          f"fold_score {ev['mean_fold_score']:.3f}  reward {ev['mean_reward']:.2f}")


if __name__ == "__main__":
    main()
