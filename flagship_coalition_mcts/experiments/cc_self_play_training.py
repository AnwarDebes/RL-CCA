"""CD-MCTS self-play training on real Chinese Checkers.

Trains a CDMCTSNetwork on CC via self-play. This is the production-
quality training script that runs after Phase 2 v4 completes (when GPU
+ CPU are free).

Loop
----
For each iteration:
  1. Generate K self-play games (each ~50-200 moves on CC) using the
     current network.
  2. Collect (state, target_policy, observed_ranking,
              observed_coalition_idx, target_scalar_value) tuples.
  3. Train network for M steps on the batched data using cdmcts_loss.
  4. Periodically save checkpoint.

Compute considerations
----------------------
- One CC self-play game with N=2..6 and K=8 simulations per move ≈ 30s CPU.
- 32 games × 8 sim → ~16 min/iter.
- 100 iterations → ~26h CPU (unblocked). With GPU acceleration of the
  network forward, ~3-5h. We don't run on GPU until Phase 2 v4 completes.

Reproducibility
---------------
Seeds are explicit. Each iteration's RNG is derived from
(base_seed + iter_idx) so re-runs are bit-reproducible.
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

from flagship_coalition_mcts.src.cc_runner import (
    build_cc_evaluator,
    build_cc_network,
    play_one_cc_game,
)
from flagship_coalition_mcts.src.network import cdmcts_loss


def trajectory_to_batch(trajs):
    """Pack a list of game trajectories into a tensor batch."""
    flat = [e for traj in trajs for e in traj]
    if not flat:
        return None
    feats_2d = np.stack([e["features_2d"] for e in flat])
    legal_mask = np.stack([e["legal_mask"] for e in flat])
    target_pol = np.stack([e["target_policy"] for e in flat])
    obs_ranking = np.stack([e["observed_ranking"] for e in flat])
    n_players = np.array([e["num_players"] for e in flat], dtype=np.int64)
    cp = np.array([e["current_player"] for e in flat], dtype=np.int64)
    obs_coal = np.array([e["observed_coalition_index"] for e in flat], dtype=np.int64)
    target_v = np.array([e["target_scalar_value"] for e in flat], dtype=np.float32)
    return dict(
        features=torch.from_numpy(feats_2d).float(),
        legal_mask=torch.from_numpy(legal_mask),
        target_policy=torch.from_numpy(target_pol),
        observed_ranking=torch.from_numpy(obs_ranking),
        num_players=torch.from_numpy(n_players),
        current_player=torch.from_numpy(cp),
        observed_coalition_index=torch.from_numpy(obs_coal),
        target_scalar_value=torch.from_numpy(target_v),
    )


def cdmcts_iteration_cc(
    network,
    optimizer,
    iter_seed: int,
    games_per_iter: int,
    train_steps: int,
    num_simulations: int,
    num_players: int,
    coalition_weight: float,
    max_moves: int,
):
    network.eval()
    rng = np.random.default_rng(iter_seed)
    trajectories = []
    t0 = time.time()
    successes = 0
    for g in range(games_per_iter):
        seed = int(rng.integers(0, 2**31))
        result = play_one_cc_game(
            network=network,
            num_players=num_players,
            num_simulations=num_simulations,
            coalition_weight=coalition_weight,
            seed=seed,
            max_moves=max_moves,
        )
        if result["terminated"]:
            trajectories.append(result["trajectory"])
            successes += 1
    gen_sec = time.time() - t0
    print(f"  [gen] {successes}/{games_per_iter} terminated in {gen_sec:.1f}s")

    batch = trajectory_to_batch(trajectories)
    if batch is None:
        return dict(num_states=0, gen_sec=gen_sec)
    network.train()
    losses = []
    t1 = time.time()
    for step in range(train_steps):
        optimizer.zero_grad()
        loss, comps = cdmcts_loss(
            network,
            features=batch["features"],
            target_policy=batch["target_policy"],
            legal_mask=batch["legal_mask"],
            observed_ranking=batch["observed_ranking"],
            num_players_per=batch["num_players"],
            observed_coalition_index=batch["observed_coalition_index"],
            current_player_per=batch["current_player"],
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
    ap.add_argument("--coalition-weight", type=float, default=0.5)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--channels", type=int, default=64)
    ap.add_argument("--num-blocks", type=int, default=4)
    ap.add_argument("--hidden-dim", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--checkpoint-dir", default="cdmcts_cc_checkpoints")
    ap.add_argument("--save-every", type=int, default=5)
    ap.add_argument(
        "--resume-from", default=None,
        help="Path to a checkpoint to resume from. Loads network + optimizer + iter index.",
    )
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    print("=== CD-MCTS self-play training on Chinese Checkers ===")
    net, _ev = build_cc_evaluator(
        num_players_max=6,
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
        stats = cdmcts_iteration_cc(
            network=net, optimizer=opt,
            iter_seed=args.seed * 1000 + it,
            games_per_iter=args.games_per_iter,
            train_steps=args.train_steps,
            num_simulations=args.num_simulations,
            num_players=args.num_players,
            coalition_weight=args.coalition_weight,
            max_moves=args.max_moves,
        )
        history.append(dict(iter=it, **stats))
        if "avg_total" in stats:
            print(f"  [train] {args.train_steps} steps in {stats['train_sec']:.1f}s, "
                  f"loss={stats['avg_total']:.3f} "
                  f"(p={stats['avg_policy']:.3f} pl={stats['avg_pl']:.3f} "
                  f"c={stats['avg_coalition']:.3f} v={stats['avg_value']:.3f})")
        if (it + 1) % args.save_every == 0:
            path = os.path.join(args.checkpoint_dir, f"iter_{it+1:04d}.pt")
            from flagship_coalition_mcts.src.checkpoint import save_checkpoint
            save_checkpoint(
                path, net, iter_idx=it + 1, optimizer=opt,
                metadata={"history": history, "args": vars(args)},
            )
            print(f"  saved {path}")

    final = os.path.join(args.checkpoint_dir, "final.pt")
    from flagship_coalition_mcts.src.checkpoint import save_checkpoint
    save_checkpoint(
        final, net, iter_idx=args.num_iterations, optimizer=opt,
        metadata={"history": history, "args": vars(args)},
    )
    with open(os.path.join(args.checkpoint_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)
    print(f"\nFinal checkpoint: {final}")


if __name__ == "__main__":
    main()
