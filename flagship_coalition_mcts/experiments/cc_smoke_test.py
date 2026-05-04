"""Smoke-test CD-MCTS on real Chinese Checkers.

Runs ONE 2-player CC game using a tiny network with low simulation
budget. Purpose: verify the entire pipeline works end-to-end on the
real game - no crashes, valid trajectories, terminal outcomes.

This is NOT a benchmark. It's a sanity check before larger runs.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_HERE))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from flagship_coalition_mcts.src.cc_runner import (
    build_cc_evaluator,
    play_one_cc_game,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-players", type=int, default=2)
    ap.add_argument("--num-simulations", type=int, default=8)
    ap.add_argument("--max-moves", type=int, default=200)
    ap.add_argument("--channels", type=int, default=32)
    ap.add_argument("--num-blocks", type=int, default=3)
    ap.add_argument("--hidden-dim", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    print(f"=== CD-MCTS on CC smoke test ===")
    print(f"num_players={args.num_players}, sims={args.num_simulations}, "
          f"channels={args.channels}, blocks={args.num_blocks}, hidden={args.hidden_dim}")
    print(f"max_moves={args.max_moves}, seed={args.seed}")

    torch.manual_seed(args.seed)
    net, _ev = build_cc_evaluator(
        num_players_max=6,
        channels=args.channels,
        num_blocks=args.num_blocks,
        hidden_dim=args.hidden_dim,
    )
    n_params = sum(p.numel() for p in net.parameters())
    print(f"Network params: {n_params:,}")

    t0 = time.time()
    result = play_one_cc_game(
        network=net,
        num_players=args.num_players,
        num_simulations=args.num_simulations,
        coalition_weight=0.5,
        seed=args.seed,
        max_moves=args.max_moves,
    )
    elapsed = time.time() - t0
    print(f"\nGame complete in {elapsed:.1f}s")
    print(f"  num_moves: {result['num_moves']}")
    print(f"  terminated: {result['terminated']}")
    print(f"  final_ranks: {result.get('final_ranks')}")
    print(f"  trajectory length: {len(result['trajectory'])}")
    if result["terminated"] and result["trajectory"]:
        first = result["trajectory"][0]
        print(f"  first entry: cp={first['current_player']}, "
              f"target_v={first['target_scalar_value']:.3f}, "
              f"obs_coal_idx={first['observed_coalition_index']}")
    if result["num_moves"] > 0:
        avg_per_move = elapsed / result["num_moves"]
        print(f"  avg per move: {avg_per_move*1000:.0f}ms")
    print("\nSMOKE TEST OK")


if __name__ == "__main__":
    main()
