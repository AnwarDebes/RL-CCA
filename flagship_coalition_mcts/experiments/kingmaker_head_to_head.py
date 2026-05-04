"""Killer experiment: CD-MCTS vs scalar-PUCT on the kingmaker testbed.

This is the experiment that defends the paper against the strongest
reviewer attack - "coalitions don't really matter in your real games."

Setup
-----
Train two agents to identical compute on the kingmaker game, both using
the SAME network architecture (CDMCTSNetwork), with the only difference
being:

  * SCALAR agent: trained with loss weights = {policy=1, pl=0, coalition=0,
    value=1}. At inference plays via run_mcts_scalar (PUCT, scalar V).
  * CD-MCTS agent: trained with full loss. At inference plays via
    run_mcts (vector backup, EXP-IX selection, coalition penalty).

Evaluation
----------
Three configurations:

  1. ALL_SCALAR: all 3 seats use the scalar agent. Record rank
     distribution (we expect P0 dominates because they have the head start).
  2. ALL_CD: all 3 seats use the CD agent. Record rank distribution.
     **Predicted:** P0 wins less often than in ALL_SCALAR because the
     CD-MCTS players in seats 1 and 2 coordinate to suppress the leader.
  3. MIXED: seat P0 = scalar, seats P1 and P2 = CD. Record P0's win rate.
     **Predicted:** lower than the ALL_SCALAR P0-win-rate.

Pre-registered prediction (binding)
-----------------------------------
We predict that in ALL_CD, P0's win rate is at least 10 percentage points
lower than in ALL_SCALAR. If this fails, the paper reports the failure
and discusses why - possible reasons include insufficient training, the
coalition head not converging, or the kingmaker game being too small to
elicit coordination via the loss signal.

This experiment is small (~minutes on CPU) and reproducible from a
single seed.
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

# Allow running this file directly (e.g. from project root) by adding the
# project root to sys.path so the package imports resolve.
_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_HERE))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from flagship_coalition_mcts.src.baseline_mcts import (
    ScalarEvaluator,
    run_mcts_scalar,
)
from flagship_coalition_mcts.src.games.kingmaker import (
    KingmakerGame,
    KingmakerState,
    NUM_ACTIONS,
    final_ranks,
    is_terminal,
)
from flagship_coalition_mcts.src.mcts import run_mcts
from flagship_coalition_mcts.src.network import (
    CDMCTSEvaluator,
    CDMCTSNetwork,
    MLPEncoder,
)
from flagship_coalition_mcts.src.self_play import self_play_iteration


# Re-export the feature encoder used in tests (kingmaker -> 12-d vector)
def kingmaker_to_features(state):
    feats = np.zeros(12, dtype=np.float32)
    feats[0:3] = np.array(state.positions) / 3.0
    for p in state.finish_order:
        feats[3 + p] = 1.0
    feats[6 + state.next_player] = 1.0
    feats[9] = state.move_count / 6.0
    feats[10] = float(len(state.finish_order)) / 3.0
    feats[11] = 1.0
    return feats


def make_network(seed: int) -> CDMCTSNetwork:
    torch.manual_seed(seed)
    return CDMCTSNetwork(
        encoder=MLPEncoder(input_dim=12, hidden_dim=32, num_layers=2),
        action_space_size=NUM_ACTIONS,
        max_players=3,
    )


# ----------------------------------------------------------------------
# Mixed-agent game runner: each seat has its own (evaluator, mcts) pair.
# ----------------------------------------------------------------------


def play_game_with_mixed_agents(
    seat_to_play_fn: list,  # length 3: each entry is a callable(state) -> action
    rng: np.random.Generator,
) -> tuple:
    state = KingmakerState.initial()
    while not is_terminal(state):
        cp = state.next_player
        action = seat_to_play_fn[cp](state)
        state, _ = KingmakerGame.step(state, action)
    return final_ranks(state)


def make_cd_player(network: CDMCTSNetwork, num_simulations: int, rng: np.random.Generator):
    ev = CDMCTSEvaluator(
        network=network,
        state_to_features=kingmaker_to_features,
        action_space_size=NUM_ACTIONS,
        current_player_fn=KingmakerGame.current_player,
        num_players_fn=KingmakerGame.num_players,
    )
    def play(state):
        _, pi = run_mcts(
            state=state, network=ev, game=KingmakerGame(),
            num_simulations=num_simulations,
            seed=int(rng.integers(0, 2**31)),
        )
        # Greedy at evaluation
        legal = KingmakerGame.legal_actions(state)
        return legal[int(np.argmax(pi))]
    return play


def make_scalar_player(network: CDMCTSNetwork, num_simulations: int, rng: np.random.Generator):
    ev = ScalarEvaluator(
        network=network,
        state_to_features=kingmaker_to_features,
        current_player_fn=KingmakerGame.current_player,
        num_players_fn=KingmakerGame.num_players,
    )
    def play(state):
        _, pi = run_mcts_scalar(
            state=state, network=ev, game=KingmakerGame(),
            num_simulations=num_simulations,
        )
        legal = KingmakerGame.legal_actions(state)
        return legal[int(np.argmax(pi))]
    return play


# ----------------------------------------------------------------------
# Main experiment
# ----------------------------------------------------------------------


def train_one_agent(
    agent_kind: str,  # "scalar" or "cd_mcts"
    num_iterations: int = 8,
    games_per_iter: int = 8,
    train_steps: int = 16,
    num_simulations: int = 16,
    lr: float = 5e-3,
    seed: int = 0,
    log_prefix: str = "",
) -> CDMCTSNetwork:
    if agent_kind == "scalar":
        weights = {"policy": 1.0, "pl": 0.0, "coalition": 0.0, "value": 1.0}
        coalition_weight = 0.0
    elif agent_kind == "cd_mcts":
        weights = {"policy": 1.0, "pl": 1.0, "coalition": 0.5, "value": 0.5}
        coalition_weight = 0.5
    else:
        raise ValueError(agent_kind)

    net = make_network(seed=seed)
    opt = optim.Adam(net.parameters(), lr=lr)
    print(f"{log_prefix}[train_{agent_kind}] starting {num_iterations} iters")
    t0 = time.time()
    for it in range(num_iterations):
        # Hack: monkey-patch self_play_iteration's loss weights via closure?
        # Simpler: use a subclassed iteration that takes weights.
        stats = self_play_iteration_with_weights(
            network=net,
            optimizer=opt,
            initial_state_fn=KingmakerState.initial,
            game=KingmakerGame(),
            state_to_features=kingmaker_to_features,
            games_per_iter=games_per_iter,
            train_steps=train_steps,
            num_simulations=num_simulations,
            coalition_weight=coalition_weight,
            action_space_size=NUM_ACTIONS,
            max_players=3,
            rng_seed=it,
            loss_weights=weights,
        )
        print(
            f"{log_prefix}[{agent_kind} iter {it+1}/{num_iterations}] "
            f"states={stats['num_states']} "
            f"loss={stats.get('avg_total', float('nan')):.3f} "
            f"(p={stats.get('avg_policy', 0):.3f} "
            f"pl={stats.get('avg_pl', 0):.3f} "
            f"c={stats.get('avg_coalition', 0):.3f} "
            f"v={stats.get('avg_value', 0):.3f}) "
            f"elapsed={time.time()-t0:.0f}s"
        )
    return net


def self_play_iteration_with_weights(loss_weights=None, **kw):
    """Wrapper that lets us override loss weights per-agent."""
    from flagship_coalition_mcts.src.self_play import (
        play_one_game,
        train_step,
        trajectory_to_batch,
    )

    network = kw["network"]
    optimizer = kw["optimizer"]
    network.eval()
    evaluator = CDMCTSEvaluator(
        network=network,
        state_to_features=kw["state_to_features"],
        action_space_size=kw["action_space_size"],
        current_player_fn=kw["game"].current_player,
        num_players_fn=kw["game"].num_players,
    )
    rng = np.random.default_rng(kw["rng_seed"])
    trajs = []
    for _ in range(kw["games_per_iter"]):
        traj = play_one_game(
            initial_state_fn=kw["initial_state_fn"],
            game=kw["game"],
            evaluator=evaluator,
            state_to_features=kw["state_to_features"],
            num_simulations=kw["num_simulations"],
            coalition_weight=kw["coalition_weight"],
            rng_seed=int(rng.integers(0, 2**31)),
            action_space_size=kw["action_space_size"],
            max_players=kw["max_players"],
        )
        trajs.append(traj)
    batch = trajectory_to_batch(trajs, kw["action_space_size"], kw["max_players"])
    if batch is None:
        return dict(num_games=kw["games_per_iter"], num_states=0)
    network.train()
    losses = []
    for _ in range(kw["train_steps"]):
        comps = train_step(network, optimizer, batch, weights=loss_weights)
        losses.append(comps)
    avg = {k: float(np.mean([d[k] for d in losses])) for k in losses[0]}
    return dict(
        num_games=kw["games_per_iter"],
        num_states=batch["features"].shape[0],
        **{f"avg_{k}": v for k, v in avg.items()},
    )


def evaluate_config(name, seat_makers, num_games, rng):
    """Play num_games games with the given seat configuration. Return rank
    distribution per seat (P0_rank_counts, P1_rank_counts, P2_rank_counts)."""
    win_counts = np.zeros((3, 3), dtype=np.int64)  # win_counts[player, rank-1]
    for g in range(num_games):
        ranks = play_game_with_mixed_agents(seat_makers, rng)
        for p, r in enumerate(ranks):
            win_counts[p, r - 1] += 1
    return win_counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-iterations", type=int, default=10)
    ap.add_argument("--games-per-iter", type=int, default=12)
    ap.add_argument("--train-steps", type=int, default=24)
    ap.add_argument("--num-simulations", type=int, default=16)
    ap.add_argument("--eval-games", type=int, default=60)
    ap.add_argument("--eval-simulations", type=int, default=24)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="kingmaker_h2h_results.json")
    args = ap.parse_args()

    print("=" * 72)
    print("KILLER EXPERIMENT: CD-MCTS vs Scalar-AZ on Kingmaker")
    print("=" * 72)

    scalar_net = train_one_agent(
        "scalar",
        num_iterations=args.num_iterations,
        games_per_iter=args.games_per_iter,
        train_steps=args.train_steps,
        num_simulations=args.num_simulations,
        seed=args.seed,
    )
    cd_net = train_one_agent(
        "cd_mcts",
        num_iterations=args.num_iterations,
        games_per_iter=args.games_per_iter,
        train_steps=args.train_steps,
        num_simulations=args.num_simulations,
        seed=args.seed + 1000,
    )

    rng = np.random.default_rng(args.seed + 9999)

    # ALL_SCALAR
    print("\n[evaluate] ALL_SCALAR")
    seats_scalar = [make_scalar_player(scalar_net, args.eval_simulations, rng) for _ in range(3)]
    counts_scalar = evaluate_config("ALL_SCALAR", seats_scalar, args.eval_games, rng)

    # ALL_CD
    print("[evaluate] ALL_CD")
    seats_cd = [make_cd_player(cd_net, args.eval_simulations, rng) for _ in range(3)]
    counts_cd = evaluate_config("ALL_CD", seats_cd, args.eval_games, rng)

    # MIXED: P0=scalar, P1=CD, P2=CD
    print("[evaluate] MIXED P0=scalar, P1+P2=CD")
    seats_mixed = [
        make_scalar_player(scalar_net, args.eval_simulations, rng),
        make_cd_player(cd_net, args.eval_simulations, rng),
        make_cd_player(cd_net, args.eval_simulations, rng),
    ]
    counts_mixed = evaluate_config("MIXED", seats_mixed, args.eval_games, rng)

    p0_winrate_scalar = counts_scalar[0, 0] / args.eval_games
    p0_winrate_cd = counts_cd[0, 0] / args.eval_games
    p0_winrate_mixed = counts_mixed[0, 0] / args.eval_games

    print("\n" + "=" * 72)
    print("RESULTS")
    print("=" * 72)
    print(f"ALL_SCALAR: P0 win-rate = {p0_winrate_scalar:.3f}")
    print(f"             rank counts (P0/P1/P2 across positions 1/2/3):")
    print(f"             {counts_scalar.tolist()}")
    print(f"ALL_CD:     P0 win-rate = {p0_winrate_cd:.3f}")
    print(f"             rank counts: {counts_cd.tolist()}")
    print(f"MIXED:      P0 win-rate = {p0_winrate_mixed:.3f}")
    print(f"             rank counts: {counts_mixed.tolist()}")

    drop_all = p0_winrate_scalar - p0_winrate_cd
    drop_mixed = p0_winrate_scalar - p0_winrate_mixed
    print(f"\nP0-winrate drop (ALL_CD vs ALL_SCALAR):  {drop_all:+.3f}")
    print(f"P0-winrate drop (MIXED  vs ALL_SCALAR):  {drop_mixed:+.3f}")

    pre_registered_threshold = 0.10
    passed = drop_all >= pre_registered_threshold
    print(f"\nPRE-REGISTERED CLAIM (drop_all >= {pre_registered_threshold}): "
          f"{'PASSED' if passed else 'FAILED'}")

    out = {
        "args": vars(args),
        "p0_winrate_scalar": float(p0_winrate_scalar),
        "p0_winrate_cd": float(p0_winrate_cd),
        "p0_winrate_mixed": float(p0_winrate_mixed),
        "drop_all": float(drop_all),
        "drop_mixed": float(drop_mixed),
        "passed_pre_registered": bool(passed),
        "rank_counts_scalar": counts_scalar.tolist(),
        "rank_counts_cd": counts_cd.tolist(),
        "rank_counts_mixed": counts_mixed.tolist(),
    }
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nResults written to {args.out}")


if __name__ == "__main__":
    main()
