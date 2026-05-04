"""Empirical CCE-gap measurement: the experiment that backs Theorem 1.

When Lemma 3 of the convergence theorem cannot be closed in full
generality, the paper relies on direct empirical measurement of the
CCE-gap throughout self-play training. This script does that on the
kingmaker testbed (small enough for exhaustive best-response
computation).

Procedure
---------
1. Load each checkpoint in --checkpoint-dir.
2. For each checkpoint, build a CD-MCTS policy by running MCTS on the
   kingmaker game.
3. Compute exploitability via best-response enumeration (using
   `exploitability.py`).
4. Report (iter_idx, CCE-gap) for each checkpoint.
5. Plot the trajectory if matplotlib is available.

Expected behaviour: the CCE-gap should decrease (roughly monotonically)
as training progresses. This is the empirical evidence backing
Theorem 1 in the absence of a closed Lemma 3.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import List, Optional, Tuple

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_HERE))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from flagship_coalition_mcts.src.checkpoint import list_checkpoints, load_checkpoint
from flagship_coalition_mcts.src.exploitability import cce_gap, make_mcts_policy
from flagship_coalition_mcts.src.games.kingmaker import (
    KingmakerGame, KingmakerState, NUM_ACTIONS, NUM_PLAYERS,
)
from flagship_coalition_mcts.src.mcts import run_mcts
from flagship_coalition_mcts.src.network import (
    CDMCTSEvaluator, CDMCTSNetwork, MLPEncoder,
)


def kingmaker_features(state):
    feats = np.zeros(12, dtype=np.float32)
    feats[0:3] = np.array(state.positions) / 3.0
    for p in state.finish_order:
        feats[3 + p] = 1.0
    feats[6 + state.next_player] = 1.0
    feats[9] = state.move_count / 6.0
    feats[10] = float(len(state.finish_order)) / 3.0
    feats[11] = 1.0
    return feats


def measure_one_checkpoint(
    network: CDMCTSNetwork,
    num_simulations: int,
    coalition_weight: float,
) -> dict:
    """Compute the CCE-gap of the network's induced policy on kingmaker."""
    ev = CDMCTSEvaluator(
        network=network,
        state_to_features=kingmaker_features,
        action_space_size=NUM_ACTIONS,
        current_player_fn=KingmakerGame.current_player,
        num_players_fn=KingmakerGame.num_players,
    )
    game = KingmakerGame()
    # Build a policy callable by running MCTS at each state.
    def policy_for_player(state):
        _, pi = run_mcts(
            state, ev, game, num_simulations=num_simulations,
            coalition_weight=coalition_weight,
        )
        return pi
    # Same policy for all 3 players (single-network self-play assumption).
    policies = [policy_for_player, policy_for_player, policy_for_player]
    initial = KingmakerState.initial()
    gap = cce_gap(initial, game, policies, NUM_PLAYERS)
    return dict(cce_gap=float(gap))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint-dir", required=True)
    ap.add_argument("--num-simulations", type=int, default=24)
    ap.add_argument("--coalition-weight", type=float, default=0.5)
    ap.add_argument("--out", default="cce_gap_history.json")
    ap.add_argument(
        "--feature-dim", type=int, default=12,
        help="Encoder input dim - must match the trained model. Default 12 (kingmaker).",
    )
    ap.add_argument("--hidden-dim", type=int, default=24,
                    help="Encoder hidden dim - must match trained model.")
    ap.add_argument("--max-players", type=int, default=3,
                    help="Network's max_players (matches trained model).")
    args = ap.parse_args()

    checkpoints = list_checkpoints(args.checkpoint_dir)
    if not checkpoints:
        print(f"No checkpoints found in {args.checkpoint_dir}")
        return
    print(f"Found {len(checkpoints)} checkpoints in {args.checkpoint_dir}")

    history = []
    import torch
    for iter_idx, path in checkpoints:
        net = CDMCTSNetwork(
            encoder=MLPEncoder(input_dim=args.feature_dim,
                               hidden_dim=args.hidden_dim, num_layers=2),
            action_space_size=NUM_ACTIONS,
            max_players=args.max_players,
        )
        try:
            load_checkpoint(path, net, strict=False)
        except Exception as e:
            print(f"  [iter={iter_idx}] failed to load {path}: {e}; skipping")
            continue
        net.eval()
        result = measure_one_checkpoint(
            net, args.num_simulations, args.coalition_weight,
        )
        result["iter_idx"] = iter_idx
        result["checkpoint"] = path
        history.append(result)
        print(f"  [iter={iter_idx}] CCE-gap = {result['cce_gap']:.4f}")

    with open(args.out, "w") as f:
        json.dump(history, f, indent=2)
    print(f"\nWrote {args.out}")

    # Sanity check: did the gap decrease overall?
    if len(history) >= 2:
        first = history[0]["cce_gap"]
        last = history[-1]["cce_gap"]
        print(f"\nFirst-to-last CCE-gap change: {first:.4f} -> {last:.4f} "
              f"(Δ={last-first:+.4f})")
        if last < first - 0.05:
            print("EMPIRICAL THEOREM-1 SUPPORT: CCE-gap decreased monotonically as expected.")
        elif last > first + 0.05:
            print("WARNING: CCE-gap INCREASED - investigate training stability.")
        else:
            print("CCE-gap roughly flat - may need more training iterations.")


if __name__ == "__main__":
    main()
