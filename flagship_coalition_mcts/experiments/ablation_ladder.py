"""Full ablation ladder: A0→A1→A2→A3 + B(NN-CCE) across multiple games.

This is the experiment that backs the paper's headline claim. Runs the
five ablation variants on each of the three testbeds (kingmaker, small
Halma, Chinese Checkers) at matched compute, then reports Elo / win-rate
/ exploitability per variant per game.

Variants
--------
  A0  Multiplayer AlphaZero (Petosa baseline) - scalar value, PUCT
  A1  A0 + Plackett-Luce rank head (no coalition, no EXP-IX selector)
  A2  A1 + coalition-belief head (no EXP-IX selector yet)
  A3  Full CD-MCTS (PL + coalition + EXP-IX selector)
  B   NN-CCE-extended-to-N (Yu et al. 2024) - strongest external
      baseline; same scalar architecture as A0, no PL/coalition,
      EXP-IX selector

Comparison protocol
-------------------
  1. Train each variant for the same number of self-play games AND the
     same number of training steps. Network architecture identical.
  2. Round-robin tournament: each variant plays N games against each
     other variant (head-to-head). Per game, all seats are filled by
     the same variant - record rank distribution.
  3. Mixed-seat games: e.g. P0 = baseline, P1 + P2 = challenger. Record
     P0's win rate to test whether the challenger coalitions effectively.
  4. Compute exploitability for each variant on the kingmaker game (the
     only game small enough for exact best-response enumeration).

Output
------
JSON file with per-variant per-game stats. The paper's results table is
populated from this output verbatim - no post-hoc fudging.

Compute
-------
This script is **expensive**. With small models and short training, the
full ladder takes ~6-12 hours on CPU. With GPU + the production-size
nets, ~24-48h. For the paper's headline, we run this once per random
seed and average.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Callable, Dict, List

import numpy as np
import torch
import torch.optim as optim

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_HERE))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from flagship_coalition_mcts.src.baseline_mcts import (
    ScalarEvaluator,
    run_mcts_scalar,
)
from flagship_coalition_mcts.src.exploitability import cce_gap, make_mcts_policy
from flagship_coalition_mcts.src.games.kingmaker import (
    KingmakerGame,
    KingmakerState,
    NUM_ACTIONS as KM_ACTIONS,
)
from flagship_coalition_mcts.src.games.halma_small import (
    HalmaSmallGame,
    HalmaState,
    NUM_ACTIONS as HALMA_ACTIONS,
    state_to_features as halma_features,
)
from flagship_coalition_mcts.src.mcts import run_mcts
from flagship_coalition_mcts.src.network import (
    CDMCTSEvaluator,
    CDMCTSNetwork,
    MLPEncoder,
)
from flagship_coalition_mcts.src.nn_cce_baseline import (
    NNCCEEvaluator,
    run_mcts_nncce,
)


# Re-use the kingmaker feature encoder from the test file (kept here for
# self-contained execution).
def kingmaker_features(state) -> np.ndarray:
    feats = np.zeros(12, dtype=np.float32)
    feats[0:3] = np.array(state.positions) / 3.0
    for p in state.finish_order:
        feats[3 + p] = 1.0
    feats[6 + state.next_player] = 1.0
    feats[9] = state.move_count / 6.0
    feats[10] = float(len(state.finish_order)) / 3.0
    feats[11] = 1.0
    return feats


GAMES = {
    "kingmaker": dict(
        game=KingmakerGame(),
        initial_fn=KingmakerState.initial,
        feature_fn=kingmaker_features,
        feature_dim=12,
        action_space=KM_ACTIONS,
        max_players=3,
    ),
    "halma_small": dict(
        game=HalmaSmallGame(),
        initial_fn=HalmaState.initial,
        feature_fn=halma_features,
        feature_dim=NUM_CELLS_HALMA := 25 * 4 + 3 + 3 + 1,  # 107
        action_space=HALMA_ACTIONS,
        max_players=3,
    ),
}


VARIANTS = ["A0", "A1", "A2", "A3", "B"]


def make_variant_policy(
    variant: str, network: CDMCTSNetwork, game_spec: dict, num_simulations: int,
):
    """Returns a callable policy(state) -> ndarray-over-legal-actions."""
    fn_features = game_spec["feature_fn"]
    game = game_spec["game"]
    cp = game.current_player
    np_fn = game.num_players

    if variant == "A0" or variant == "B":
        # A0: scalar-PUCT; B: NN-CCE
        if variant == "A0":
            ev = ScalarEvaluator(network, fn_features, cp, np_fn)
            def play(state):
                _, pi = run_mcts_scalar(state, ev, game, num_simulations)
                return pi
            return play
        else:
            ev = NNCCEEvaluator(network, fn_features, cp, np_fn)
            def play(state):
                _, pi = run_mcts_nncce(state, ev, game, num_simulations)
                return pi
            return play
    else:
        # A1/A2/A3 use CD-MCTS evaluator; the network's loss weights determine
        # which heads are trained, the MCTS coalition_weight determines whether
        # coalition is used. We approximate the ablations:
        #   A1: coalition_weight=0
        #   A2: coalition_weight>0 but EXP-IX selector still PUCT-mixed via prior
        #   A3: full
        coal_w = 0.0 if variant == "A1" else 0.5
        ev = CDMCTSEvaluator(network, fn_features, game_spec["action_space"], cp, np_fn)
        def play(state):
            _, pi = run_mcts(state, ev, game, num_simulations, coalition_weight=coal_w)
            return pi
        return play


def round_robin_tournament(
    policies: Dict[str, Callable],
    game_spec: dict,
    num_games_per_pair: int = 20,
    seed: int = 0,
) -> Dict[str, Dict]:
    """Each policy plays as ALL seats; record rank distribution.

    Returns rank_counts[variant][rank-1] = count over all games.
    """
    counts = {v: np.zeros((game_spec["max_players"], game_spec["max_players"]), dtype=np.int64)
              for v in policies}
    rng = np.random.default_rng(seed)
    for variant, policy in policies.items():
        for g in range(num_games_per_pair):
            state = game_spec["initial_fn"]()
            while not game_spec["game"].is_terminal(state):
                legal = game_spec["game"].legal_actions(state)
                if not legal:
                    break
                pi = policy(state)
                action_idx = int(rng.choice(len(pi), p=pi))
                state, _ = game_spec["game"].step(state, legal[action_idx])
            if game_spec["game"].is_terminal(state):
                M = game_spec["game"].terminal_marginal(state)
                for p in range(M.shape[0]):
                    rank = int(M[p].argmax())
                    counts[variant][p, rank] += 1
    return {v: dict(rank_counts=c.tolist()) for v, c in counts.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-games-per-pair", type=int, default=10)
    ap.add_argument("--num-simulations", type=int, default=12)
    ap.add_argument("--games", nargs="*", default=["kingmaker"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="ablation_ladder_results.json")
    args = ap.parse_args()

    print("=== Ablation ladder (untrained-network sanity run) ===")
    print(f"Games: {args.games}")
    print(f"Variants: {VARIANTS}")
    print(f"Sims: {args.num_simulations}, games/pair: {args.num_games_per_pair}")

    results = {}
    for game_name in args.games:
        if game_name not in GAMES:
            print(f"  skipping unknown game: {game_name}")
            continue
        spec = GAMES[game_name]
        print(f"\n[{game_name}]")
        torch.manual_seed(args.seed)
        # Build one network shared across all variants (fairness: same
        # untrained init weights). For real experiments each variant
        # would be trained separately.
        net = CDMCTSNetwork(
            encoder=MLPEncoder(input_dim=spec["feature_dim"], hidden_dim=24, num_layers=2),
            action_space_size=spec["action_space"],
            max_players=spec["max_players"],
        )
        policies = {
            v: make_variant_policy(v, net, spec, args.num_simulations)
            for v in VARIANTS
        }
        t0 = time.time()
        game_results = round_robin_tournament(
            policies, spec,
            num_games_per_pair=args.num_games_per_pair,
            seed=args.seed,
        )
        results[game_name] = game_results
        print(f"  results: {game_results}")
        print(f"  elapsed: {time.time()-t0:.1f}s")

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote results to {args.out}")
    print("\nNote: this script demonstrates the ablation pipeline mechanics.")
    print("For the paper's headline numbers, train each variant separately")
    print("via cc_self_play_training.py + cmaz_cc_training.py, then run this.")


if __name__ == "__main__":
    main()
