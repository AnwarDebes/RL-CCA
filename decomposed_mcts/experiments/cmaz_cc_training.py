"""CMAZ self-play training on real Chinese Checkers.

Trains a CMAZ network on CC via self-play with the 4-component
score-decomposed targets matching the teacher's tournament scoring:

    components = [pin_goal/1000, distance/200, time/100, move/1]

The mixer is trained jointly with the network. After training, the
mixer can be overridden at inference for utility re-weighting (the
killer property - same network, different objectives, no retraining).

Compute: ~2x slower than CD-MCTS training on the same network size,
because we run more MCTS sims per move (CMAZ doesn't have the EXP-IX
no-regret guarantee that lets CD-MCTS get away with fewer sims).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.optim as optim

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_HERE))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from decomposed_mcts.src.cc_adapter import (
    build_cmaz_cc_network,
    play_one_cmaz_cc_game,
)
from decomposed_mcts.src.network import cmaz_loss


def trajectory_to_cmaz_batch(trajs):
    flat = [e for traj in trajs for e in traj]
    if not flat:
        return None
    feats_2d = np.stack([e["features_2d"] for e in flat])
    legal_mask = np.stack([e["legal_mask"] for e in flat])
    target_pol = np.stack([e["target_policy"] for e in flat])
    target_comp = np.stack([e["target_components"] for e in flat]).astype(np.float32)
    target_util = np.array(
        [e.get("target_total_utility", target_comp[i].mean())
         for i, e in enumerate(flat)],
        dtype=np.float32,
    )
    return dict(
        features=torch.from_numpy(feats_2d).float(),
        legal_mask=torch.from_numpy(legal_mask),
        target_policy=torch.from_numpy(target_pol),
        target_components=torch.from_numpy(target_comp),
        target_total_utility=torch.from_numpy(target_util),
    )


def cmaz_iter_cc(
    network, optimizer, iter_seed, games_per_iter, train_steps,
    num_simulations, num_players, max_moves,
):
    network.eval()
    rng = np.random.default_rng(iter_seed)
    trajectories = []
    successes = 0
    t0 = time.time()
    for g in range(games_per_iter):
        seed = int(rng.integers(0, 2**31))
        result = play_one_cmaz_cc_game(
            network=network,
            num_players=num_players,
            num_simulations=num_simulations,
            seed=seed,
            max_moves=max_moves,
        )
        if result["terminated"]:
            trajectories.append(result["trajectory"])
            successes += 1
    gen_sec = time.time() - t0
    print(f"  [gen] {successes}/{games_per_iter} terminated in {gen_sec:.1f}s")

    batch = trajectory_to_cmaz_batch(trajectories)
    if batch is None:
        return dict(num_states=0, gen_sec=gen_sec)
    network.train()
    losses = []
    t1 = time.time()
    for _ in range(train_steps):
        optimizer.zero_grad()
        loss, comps = cmaz_loss(
            network,
            features=batch["features"],
            target_policy=batch["target_policy"],
            legal_mask=batch["legal_mask"],
            target_components=batch["target_components"],
            target_total_utility=batch["target_total_utility"],
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
    ap.add_argument("--channels", type=int, default=64)
    ap.add_argument("--num-blocks", type=int, default=4)
    ap.add_argument("--hidden-dim", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--checkpoint-dir", default="cmaz_cc_checkpoints")
    ap.add_argument("--save-every", type=int, default=5)
    ap.add_argument("--resume-from", default=None)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    print("=== CMAZ self-play training on Chinese Checkers ===")
    net = build_cmaz_cc_network(
        channels=args.channels, num_blocks=args.num_blocks,
        hidden_dim=args.hidden_dim,
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
        stats = cmaz_iter_cc(
            net, opt, args.seed * 1000 + it,
            args.games_per_iter, args.train_steps,
            args.num_simulations, args.num_players, args.max_moves,
        )
        history.append(dict(iter=it, **stats))
        if "avg_total" in stats:
            print(f"  [train] {args.train_steps} steps in {stats['train_sec']:.1f}s "
                  f"loss={stats['avg_total']:.3f} "
                  f"(p={stats['avg_policy']:.3f} comps={stats['avg_components']:.3f})")
        if (it + 1) % args.save_every == 0:
            path = os.path.join(args.checkpoint_dir, f"iter_{it+1:04d}.pt")
            torch.save({"state_dict": net.state_dict(), "iter": it + 1, "history": history}, path)
            print(f"  saved {path}")
    final = os.path.join(args.checkpoint_dir, "final.pt")
    torch.save({"state_dict": net.state_dict(), "iter": args.num_iterations, "history": history}, final)
    with open(os.path.join(args.checkpoint_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)
    print(f"\nFinal checkpoint: {final}")


if __name__ == "__main__":
    main()
