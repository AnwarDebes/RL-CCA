"""Wreath-equivariant self-play training on Chinese Checkers.

Trains a WreathCCNetwork via vanilla AlphaZero-style self-play. The
contribution is the *equivariant architecture* - comparison vs
non-equivariant baselines is the killer experiment.

Headline experiments (run after training):
  1. Sample efficiency vs MLPEncoder + same-size non-equivariant network.
  2. Zero-shot N-generalisation: train on N=2 and N=3 only, evaluate
     on N=4 and N=6 with NO retraining.
  3. Bit-identical seat-permutation logits (sanity check).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_HERE))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from equivariant_net.src.cc_runner import play_one_wreath_cc_game
from equivariant_net.src.wreath_network import WreathCCNetwork


def trajectories_to_batch(trajs):
    flat = [e for t in trajs for e in t]
    if not flat:
        return None
    feats_2d = np.stack([e["features_2d"] for e in flat])
    seat_feats = np.stack([e["seat_features"] for e in flat])
    legal_mask = np.stack([e["legal_mask"] for e in flat])
    target_pol = np.stack([e["target_policy"] for e in flat])
    target_v = np.array([e["target_scalar_value"] for e in flat], dtype=np.float32)
    return dict(
        features=torch.from_numpy(feats_2d).float(),
        seat_features=torch.from_numpy(seat_feats).float(),
        legal_mask=torch.from_numpy(legal_mask),
        target_policy=torch.from_numpy(target_pol),
        target_scalar_value=torch.from_numpy(target_v),
    )


def wreath_loss(
    network: WreathCCNetwork,
    features: torch.Tensor,
    seat_features: torch.Tensor,
    target_policy: torch.Tensor,
    legal_mask: torch.Tensor,
    target_scalar_value: torch.Tensor,
):
    policy_logits, scalar_v = network(features, seat_features)
    masked_logits = policy_logits.masked_fill(~legal_mask, -1e9)
    log_probs = F.log_softmax(masked_logits, dim=-1)
    policy_loss = -(target_policy * log_probs).sum(dim=-1).mean()
    value_loss = F.mse_loss(scalar_v, target_scalar_value)
    total = policy_loss + value_loss
    return total, dict(
        total=total.item(), policy=policy_loss.item(), value=value_loss.item()
    )


def wreath_iter_cc(
    network, optimizer, iter_seed,
    games_per_iter, train_steps, num_simulations, num_players, max_moves,
):
    network.eval()
    rng = np.random.default_rng(iter_seed)
    trajs = []
    successes = 0
    t0 = time.time()
    for g in range(games_per_iter):
        seed = int(rng.integers(0, 2**31))
        result = play_one_wreath_cc_game(
            network=network,
            num_players=num_players,
            num_simulations=num_simulations,
            seed=seed,
            max_moves=max_moves,
        )
        if result["terminated"]:
            trajs.append(result["trajectory"])
            successes += 1
    gen_sec = time.time() - t0
    print(f"  [gen] {successes}/{games_per_iter} terminated in {gen_sec:.1f}s")
    batch = trajectories_to_batch(trajs)
    if batch is None:
        return dict(num_states=0, gen_sec=gen_sec)
    network.train()
    losses = []
    t1 = time.time()
    for _ in range(train_steps):
        optimizer.zero_grad()
        loss, comps = wreath_loss(
            network,
            features=batch["features"],
            seat_features=batch["seat_features"],
            target_policy=batch["target_policy"],
            legal_mask=batch["legal_mask"],
            target_scalar_value=batch["target_scalar_value"],
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(network.parameters(), 1.0)
        optimizer.step()
        losses.append(comps)
    train_sec = time.time() - t1
    avg = {k: float(np.mean([d[k] for d in losses])) for k in losses[0]}
    return dict(
        num_states=batch["features"].shape[0],
        gen_sec=gen_sec, train_sec=train_sec,
        **{f"avg_{k}": v for k, v in avg.items()},
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-iterations", type=int, default=20)
    ap.add_argument("--games-per-iter", type=int, default=4)
    ap.add_argument("--train-steps", type=int, default=16)
    ap.add_argument("--num-simulations", type=int, default=12)
    ap.add_argument("--num-players", type=int, default=2)
    ap.add_argument("--max-moves", type=int, default=300)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--spatial-channels", type=int, default=16)
    ap.add_argument("--spatial-blocks", type=int, default=3)
    ap.add_argument("--spatial-out", type=int, default=128)
    ap.add_argument("--seat-hidden", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--checkpoint-dir", default="wreath_cc_checkpoints")
    ap.add_argument("--save-every", type=int, default=5)
    ap.add_argument("--resume-from", default=None)
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    print("=== Wreath-equivariant self-play training on CC ===")
    net = WreathCCNetwork(
        spatial_channels=args.spatial_channels,
        spatial_blocks=args.spatial_blocks,
        spatial_out=args.spatial_out,
        seat_hidden=args.seat_hidden,
    )
    n_params = sum(p.numel() for p in net.parameters())
    print(f"Network params: {n_params:,}")
    opt = optim.Adam(net.parameters(), lr=args.lr)
    start_iter = 0
    history = []
    if args.resume_from is not None:
        from flagship_coalition_mcts.src.checkpoint import load_checkpoint
        bundle = load_checkpoint(args.resume_from, net, optimizer=opt, strict=False)
        start_iter = bundle.iter_idx
        history = (bundle.metadata or {}).get("history", [])
        print(f"Resumed from {args.resume_from} at iter {start_iter}")
    for it in range(start_iter, args.num_iterations):
        print(f"\n[iter {it+1}/{args.num_iterations}]")
        stats = wreath_iter_cc(
            net, opt, args.seed * 1000 + it,
            args.games_per_iter, args.train_steps,
            args.num_simulations, args.num_players, args.max_moves,
        )
        history.append(dict(iter=it, **stats))
        if "avg_total" in stats:
            print(f"  loss={stats['avg_total']:.3f} (p={stats['avg_policy']:.3f} "
                  f"v={stats['avg_value']:.3f}) train={stats['train_sec']:.1f}s")
        if (it + 1) % args.save_every == 0:
            path = os.path.join(args.checkpoint_dir, f"iter_{it+1:04d}.pt")
            torch.save({"state_dict": net.state_dict(), "iter": it+1, "history": history}, path)
            print(f"  saved {path}")
    final = os.path.join(args.checkpoint_dir, "final.pt")
    torch.save({"state_dict": net.state_dict(), "iter": args.num_iterations, "history": history}, final)
    with open(os.path.join(args.checkpoint_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)
    print(f"\nFinal checkpoint: {final}")


if __name__ == "__main__":
    main()
