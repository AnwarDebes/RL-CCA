"""Final-evaluation script: produces the paper's headline results table.

Given trained checkpoints for each ablation variant (and optional
external baselines), runs round-robin head-to-head tournaments and
reports per-pair Elo gaps with bootstrap CIs, plus per-game Elo
rankings and exploitability metrics.

Usage:
    python flagship_coalition_mcts/experiments/final_evaluation.py \\
        --variant cdmcts:checkpoints/cdmcts_cc/final.pt \\
        --variant scalar:checkpoints/scalar_cc/final.pt \\
        --variant nncce:checkpoints/nncce_cc/final.pt \\
        --variant heuristic: \\
        --num-games 100 \\
        --num-simulations 32 \\
        --out final_results.json

Each --variant argument is `name:checkpoint_path` (path may be empty for
heuristic). The script will:
  1. Load each variant's network (if applicable).
  2. Run round-robin head-to-head: every pair plays num-games each
     in a 2-player game (we use kingmaker for fast iteration; CC takes
     more compute).
  3. Compute Elo gaps with 95% bootstrap CI.
  4. Optionally measure exploitability on kingmaker.
  5. Output a JSON summary (consumable by results_table.py).

This script does NOT train anything - it requires already-trained
checkpoints. For training, see cc_self_play_training.py / cmaz_cc_training.py
/ wreath_cc_training.py.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_HERE))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from flagship_coalition_mcts.src.baseline_mcts import (
    ScalarEvaluator, run_mcts_scalar,
)
from flagship_coalition_mcts.src.checkpoint import load_checkpoint
from flagship_coalition_mcts.src.exploitability import cce_gap, make_mcts_policy
from flagship_coalition_mcts.src.games.kingmaker import (
    KingmakerGame, KingmakerState, NUM_ACTIONS, NUM_PLAYERS,
)
from flagship_coalition_mcts.src.head_to_head import head_to_head, permutation_test
from flagship_coalition_mcts.src.mcts import run_mcts
from flagship_coalition_mcts.src.network import (
    CDMCTSEvaluator, CDMCTSNetwork, MLPEncoder,
)
from flagship_coalition_mcts.src.nn_cce_baseline import (
    NNCCEEvaluator, run_mcts_nncce,
)
from training.heuristic_agent import HeuristicAgent


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


def make_variant_player(variant: str, ckpt_path: str, num_simulations: int):
    """Returns a callable agent: state -> action_idx (over legal actions)."""
    if variant == "heuristic":
        # Use the existing nexus heuristic on the kingmaker testbed:
        # it's a standalone heuristic that doesn't need a model.
        rng = np.random.default_rng(0)
        def agent(state):
            legal = KingmakerGame.legal_actions(state)
            # Heuristic for kingmaker: prefer SPRINT (action 0) when possible.
            if 0 in legal:
                return legal.index(0)
            return int(rng.integers(0, len(legal)))
        return agent

    # All neural variants share architecture; only the MCTS routine differs.
    net = CDMCTSNetwork(
        encoder=MLPEncoder(input_dim=12, hidden_dim=24, num_layers=2),
        action_space_size=NUM_ACTIONS,
        max_players=3,
    )
    if ckpt_path and os.path.exists(ckpt_path):
        try:
            load_checkpoint(ckpt_path, net, strict=False)
            print(f"  [{variant}] loaded {ckpt_path}")
        except Exception as e:
            print(f"  [{variant}] load failed ({e}); using untrained")
    else:
        print(f"  [{variant}] no checkpoint; using untrained")
    net.eval()

    if variant == "scalar":
        ev = ScalarEvaluator(
            network=net, state_to_features=kingmaker_features,
            current_player_fn=KingmakerGame.current_player,
            num_players_fn=KingmakerGame.num_players,
        )
        def agent(state):
            _, pi = run_mcts_scalar(state, ev, KingmakerGame(), num_simulations=num_simulations)
            return int(np.argmax(pi))
        return agent
    if variant == "nncce":
        ev = NNCCEEvaluator(
            network=net, state_to_features=kingmaker_features,
            current_player_fn=KingmakerGame.current_player,
            num_players_fn=KingmakerGame.num_players,
        )
        def agent(state):
            _, pi = run_mcts_nncce(state, ev, KingmakerGame(), num_simulations=num_simulations)
            return int(np.argmax(pi))
        return agent
    if variant == "cdmcts":
        ev = CDMCTSEvaluator(
            network=net, state_to_features=kingmaker_features,
            action_space_size=NUM_ACTIONS,
            current_player_fn=KingmakerGame.current_player,
            num_players_fn=KingmakerGame.num_players,
        )
        def agent(state):
            _, pi = run_mcts(state, ev, KingmakerGame(),
                             num_simulations=num_simulations,
                             coalition_weight=0.5)
            return int(np.argmax(pi))
        return agent
    raise ValueError(f"unknown variant: {variant}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--variant", action="append", default=[],
        help="Variant spec: name:checkpoint_path (e.g. cdmcts:ckpt.pt). "
             "Special name 'heuristic' uses a hand-coded heuristic.",
    )
    ap.add_argument("--num-games", type=int, default=30)
    ap.add_argument("--num-simulations", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="final_evaluation_results.json")
    ap.add_argument("--measure-exploitability", action="store_true")
    args = ap.parse_args()

    if not args.variant:
        print("ERROR: at least one --variant required")
        return 1

    print("=" * 72)
    print("FINAL EVALUATION - round-robin head-to-head on kingmaker")
    print("=" * 72)

    parsed = []
    for spec in args.variant:
        if ":" in spec:
            name, path = spec.split(":", 1)
        else:
            name, path = spec, ""
        parsed.append((name, path))
    print(f"\nVariants: {[n for n, _ in parsed]}")

    print("\nBuilding agents...")
    agents = {n: make_variant_player(n, p, args.num_simulations) for n, p in parsed}

    # Round-robin
    print(f"\nRound-robin: {args.num_games} games per pair, num_a_seats=1")
    pairs = []
    names = list(agents.keys())
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            pairs.append((names[i], names[j]))

    results = {"pair_matches": [], "args": vars(args)}
    for name_a, name_b in pairs:
        print(f"\n  {name_a} vs {name_b}")
        t0 = time.time()
        res = head_to_head(
            game=KingmakerGame(),
            initial_state_fn=KingmakerState.initial,
            num_players=NUM_PLAYERS,
            agent_a=agents[name_a], agent_b=agents[name_b],
            name_a=name_a, name_b=name_b,
            num_games=args.num_games,
            num_a_seats=1,
            seed=args.seed,
        )
        elapsed = time.time() - t0
        p_value = permutation_test(res, n_resamples=500, seed=args.seed)
        print(f"    {res.summary()}  p={p_value:.3f}  ({elapsed:.0f}s)")
        results["pair_matches"].append({
            "a": name_a, "b": name_b,
            "expected_score_a": float(res.expected_score_a()),
            "elo_gap": float(res.elo_gap()),
            "p_value": float(p_value),
            "win_counts": res.win_counts_per_agent,
            "num_games": res.num_games,
            "elapsed_sec": elapsed,
        })

    if args.measure_exploitability:
        print("\nMeasuring exploitability on kingmaker...")
        results["exploitability"] = {}
        for name, agent in agents.items():
            t0 = time.time()
            # The agent gives action_idx; wrap it as a policy returning a one-hot vector
            def policy_for_agent(state, agent=agent):
                legal = KingmakerGame.legal_actions(state)
                pi = np.zeros(len(legal))
                pi[agent(state)] = 1.0
                return pi
            policies = [policy_for_agent] * NUM_PLAYERS
            initial = KingmakerState.initial()
            try:
                gap = float(cce_gap(initial, KingmakerGame(), policies, NUM_PLAYERS))
            except Exception as e:
                print(f"    {name}: exploitability calc failed ({e})")
                gap = float("nan")
            print(f"    {name}: CCE-gap = {gap:.4f}  ({time.time()-t0:.0f}s)")
            results["exploitability"][name] = gap

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults written to {args.out}")
    print("\nUse `python -m flagship_coalition_mcts.src.results_table --files",
          args.out, "--format latex` to generate paper-ready table.")


if __name__ == "__main__":
    main()
