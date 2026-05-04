#!/usr/bin/env python3
"""Comprehensive benchmark of a v3 checkpoint against three opponent classes
(Random, Greedy=Heuristic, Advanced=2-ply lookahead) across N=2..6.

Reports detailed per-game and aggregate stats:
  - final_score and score components (time, move, pin_goal, distance)
  - pins_in_goal at end, distance remaining
  - move_count (our agent's), total game moves
  - rank, win_rate
"""
from __future__ import annotations
import argparse
import os
import random
import statistics as stats
import sys
import time

NEXUS_DIR = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, NEXUS_DIR)

import numpy as np
import torch

from config import Config
from core.board import HexBoard
from core.game_env import GameEnv, COLOUR_ORDER
from core.action_space import get_legal_actions, decode_action
from core import teacher_score as ts
from network.model import NexusNet
from network.model_v3 import NexusNetV3
from training.heuristic_agent import HeuristicAgent


def _load_auto(path, device):
    sd = torch.load(path, map_location="cpu", weights_only=True)
    is_v3 = any(k.startswith("backbone.res_blocks_a.") for k in sd.keys())
    if is_v3:
        m = NexusNetV3().to(device)
        m.load_state_dict(sd)
        return m, True
    return NexusNet.load(path, str(device)), False


def _net_action(net, env, player, device, is_v3):
    state = env.get_state_tensor(player).unsqueeze(0).to(device)
    mask = env.get_legal_mask(player).unsqueeze(0).to(device)
    with torch.no_grad():
        if is_v3:
            seat_t = torch.tensor([player], device=device)
            out = net(state, mask, current_seat=seat_t)
        else:
            out = net(state, mask)
    pol = out["policy"][0].cpu().numpy()
    return int(np.argmax(pol))


# ── Opponent definitions ─────────────────────────────────────────────


def random_opponent(env, player, rng):
    legal = get_legal_actions(env.get_legal_mask(player))
    return rng.choice(legal) if legal else 0


def greedy_opponent_factory(board):
    """Wraps HeuristicAgent - the standard greedy-distance opponent."""
    h = HeuristicAgent(board)
    def f(env, player, rng):
        return h.choose_move(env, player)
    return f


def advanced_opponent_factory(board):
    """2-ply lookahead heuristic: pick the move that, after our (heuristic)
    next-best move, gives the lowest expected distance-to-goal.

    For each candidate move:
      1. simulate the move on a clone
      2. simulate the next mover (any player) doing their heuristic best
      3. score the resulting state from OUR perspective
    Pick highest-score candidate. Ties broken by 1-ply heuristic preference.
    """
    h = HeuristicAgent(board)

    def score_state(env, player):
        color = env.colors[player]
        dist = board.sum_distances_to_goal(env.pieces[player], color)
        pins = board.count_in_goal(env.pieces[player], color)
        # higher = better for us
        return -dist + 100 * pins

    def f(env, player, rng):
        legal = get_legal_actions(env.get_legal_mask(player))
        if not legal:
            return 0
        # Trim candidate set to top-K by 1-ply heuristic to keep cost bounded
        K = min(8, len(legal))
        # rank candidates by 1-ply state score
        ranked = []
        for a in legal:
            cand = env.clone()
            cand.step(a)
            s1 = score_state(cand, player)
            ranked.append((s1, a, cand))
        ranked.sort(reverse=True, key=lambda x: x[0])
        candidates = ranked[:K]

        best_a = candidates[0][1]
        best_s = -1e9
        for s1, a, cand in candidates:
            # 2-ply: simulate the next mover's heuristic move
            if not cand.is_done():
                next_p = cand.current_player
                try:
                    next_action = h.choose_move(cand, next_p)
                    cand2 = cand.clone()
                    cand2.step(next_action)
                    s2 = score_state(cand2, player)
                except Exception:
                    s2 = s1
            else:
                s2 = s1
            if s2 > best_s:
                best_s = s2
                best_a = a
        return best_a

    return f


# ── Eval driver ──────────────────────────────────────────────────────


def run_one_game(net, device, is_v3, board, N, opp_fn, net_seat, rng_seed):
    env = GameEnv(board, num_players=N)
    env.reset(random_colors=True, rng=random.Random(rng_seed))
    rng = random.Random(rng_seed + 1000)
    while not env.is_done():
        p = env.current_player
        if p == net_seat:
            action = _net_action(net, env, p, device, is_v3)
        else:
            action = opp_fn(env, p, rng)
        env.step(action)
    # Collect detailed per-player stats for our agent
    color = env.colors[net_seat]
    pieces = env.pieces[net_seat]
    pins_in_goal = board.count_in_goal(pieces, color)
    total_dist = board.sum_distances_to_goal(pieces, color)
    move_count = env.player_move_counts[net_seat]
    time_taken = env.player_time_taken[net_seat]
    final_score = env.compute_final_score(net_seat)
    components = ts.score_components(time_taken, move_count, pins_in_goal, total_dist)
    # Rank: 1 = highest final_score
    all_scores = [env.compute_final_score(p) for p in range(N)]
    sorted_desc = sorted(all_scores, reverse=True)
    rank = sorted_desc.index(final_score) + 1
    return {
        "N": N, "net_seat": net_seat, "winner": env.get_winner(),
        "won": env.get_winner() == net_seat,
        "final_score": final_score,
        "rank": rank,
        "pins_in_goal": pins_in_goal,
        "total_distance": total_dist,
        "move_count": move_count,
        "time_taken_sec": time_taken,
        "total_game_moves": env.move_count,
        "time_score": components["time_score"],
        "move_score": components["move_score"],
        "pin_goal_score": components["pin_goal_score"],
        "distance_score": components["distance_score"],
    }


def aggregate(results):
    """Compute mean/median/std/min/max for numeric fields."""
    if not results:
        return {}
    keys = ["final_score", "rank", "pins_in_goal", "total_distance",
            "move_count", "time_score", "move_score",
            "pin_goal_score", "distance_score", "total_game_moves"]
    out = {}
    for k in keys:
        vals = [r[k] for r in results]
        out[k] = {
            "mean": stats.mean(vals),
            "median": stats.median(vals),
            "std": stats.stdev(vals) if len(vals) > 1 else 0.0,
            "min": min(vals),
            "max": max(vals),
        }
    out["win_rate"] = sum(1 for r in results if r["won"]) / len(results)
    out["games"] = len(results)
    return out


def fmt_table(label, agg):
    s = agg
    print(f"\n  --- {label} ---")
    print(f"  Games: {s['games']}, Win rate: {s['win_rate']:.0%}")
    print(f"  {'metric':<22}{'mean':>9}{'median':>9}{'std':>9}{'min':>9}{'max':>9}")
    for k in ["final_score", "rank", "pins_in_goal", "total_distance",
              "move_count", "time_score", "move_score", "pin_goal_score",
              "distance_score", "total_game_moves"]:
        v = s[k]
        print(f"  {k:<22}{v['mean']:>9.1f}{v['median']:>9.1f}{v['std']:>9.1f}"
              f"{v['min']:>9.1f}{v['max']:>9.1f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--games-per-N", type=int, default=20)
    ap.add_argument("--Ns", default="2,3,4,5,6")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net, is_v3 = _load_auto(args.model, device)
    net.eval()
    Ns = [int(x) for x in args.Ns.split(",")]
    board = HexBoard()

    print(f"\n{'='*60}\n HEAVY V3 BENCHMARK: {args.model}")
    print(f" Detected: {'NexusNetV3' if is_v3 else 'NexusNet (v2)'}")
    print(f" Device: {device}, games/N: {args.games_per_N}")
    print(f"{'='*60}")

    opponents = [
        ("RANDOM", lambda: lambda env, p, rng: random_opponent(env, p, rng)),
        ("GREEDY (Heuristic)", lambda: greedy_opponent_factory(board)),
        ("ADVANCED (2-ply lookahead)", lambda: advanced_opponent_factory(board)),
    ]

    for opp_name, opp_factory_factory in opponents:
        print(f"\n\n{'#'*60}\n# vs {opp_name}\n{'#'*60}")
        opp_fn = opp_factory_factory()
        for N in Ns:
            t0 = time.time()
            results = []
            for g in range(args.games_per_N):
                seat = g % N
                r = run_one_game(net, device, is_v3, board, N, opp_fn, seat,
                                 rng_seed=20260502 + N * 1000 + g)
                results.append(r)
            agg = aggregate(results)
            fmt_table(f"N={N} ({int(time.time()-t0)}s)", agg)


if __name__ == "__main__":
    main()
