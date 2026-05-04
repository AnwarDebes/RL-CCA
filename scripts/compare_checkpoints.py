#!/usr/bin/env python3
"""Compare multiple v3 checkpoints with 100 games/N vs Heuristic across N=2..6.

The goal: answer "which checkpoint is actually best" with enough samples to
beat the noise that fooled the per-iter server eval (10 games/N).
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
from core.game_env import GameEnv
from core.action_space import get_legal_actions
from core import teacher_score as ts
from network.model_v3 import NexusNetV3
from training.heuristic_agent import HeuristicAgent


def load_v3(path, device):
    sd = torch.load(path, map_location="cpu", weights_only=True)
    m = NexusNetV3().to(device)
    m.load_state_dict(sd)
    m.eval()
    return m


def net_action(net, env, player, device):
    state = env.get_state_tensor(player).unsqueeze(0).to(device)
    mask = env.get_legal_mask(player).unsqueeze(0).to(device)
    seat_t = torch.tensor([player], device=device)
    with torch.no_grad():
        out = net(state, mask, current_seat=seat_t)
    return int(out["policy"][0].argmax().item())


def eval_one(net, device, board, N, n_games, heuristic):
    """Play n_games. Net rotates seats fairly. Return list of per-game records."""
    rng = random.Random(20260503 + N)
    records = []
    for g in range(n_games):
        env = GameEnv(board, num_players=N)
        env.reset(random_colors=True, rng=random.Random(42 + g))
        net_seat = g % N
        while not env.is_done():
            p = env.current_player
            if p == net_seat:
                a = net_action(net, env, p, device)
            else:
                a = heuristic.choose_move(env, p)
            env.step(a)
        score = env.compute_final_score(net_seat)
        all_scores = [env.compute_final_score(pp) for pp in range(N)]
        rank = sorted(all_scores, reverse=True).index(score) + 1
        color = env.colors[net_seat]
        pins = board.count_in_goal(env.pieces[net_seat], color)
        dist = board.sum_distances_to_goal(env.pieces[net_seat], color)
        records.append({
            "won": env.get_winner() == net_seat,
            "score": score,
            "rank": rank,
            "pins": pins,
            "dist": dist,
        })
    return records


def summarize(records):
    n = len(records)
    if n == 0:
        return {}
    scores = [r["score"] for r in records]
    ranks = [r["rank"] for r in records]
    pins = [r["pins"] for r in records]
    return {
        "win_rate": sum(1 for r in records if r["won"]) / n,
        "rank1_rate": sum(1 for r in records if r["rank"] == 1) / n,
        "mean_score": stats.mean(scores),
        "median_score": stats.median(scores),
        "score_std": stats.stdev(scores) if n > 1 else 0,
        "mean_rank": stats.mean(ranks),
        "mean_pins": stats.mean(pins),
        "n": n,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoints", nargs="+", required=True)
    ap.add_argument("--games-per-N", type=int, default=100)
    ap.add_argument("--Ns", default="2,3,4,5,6")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    board = HexBoard()
    heuristic = HeuristicAgent(board)
    Ns = [int(x) for x in args.Ns.split(",")]

    print(f"\n{'#'*70}")
    print(f"# Checkpoint comparison: {len(args.checkpoints)} ckpts x {args.games_per_N} games/N x {len(Ns)} N values")
    print(f"# Opponent: HeuristicAgent (the strong non-RL baseline)")
    print(f"# Device: {device}")
    print(f"{'#'*70}\n")

    all_results = {}
    for ckpt in args.checkpoints:
        name = os.path.basename(ckpt)
        print(f"\n=== Loading {name} ===")
        net = load_v3(ckpt, device)
        per_N = {}
        for N in Ns:
            t0 = time.time()
            recs = eval_one(net, device, board, N, args.games_per_N, heuristic)
            s = summarize(recs)
            per_N[N] = s
            print(f"  N={N}: win_rate={s['win_rate']:.1%}  rank1={s['rank1_rate']:.1%}  "
                  f"mean_score={s['mean_score']:.0f}  mean_rank={s['mean_rank']:.2f}  "
                  f"mean_pins={s['mean_pins']:.1f}/10  ({int(time.time()-t0)}s)")
        all_results[name] = per_N
        # Mean across N
        mean_score = stats.mean([per_N[N]["mean_score"] for N in Ns])
        mean_winrate = stats.mean([per_N[N]["win_rate"] for N in Ns])
        print(f"  SUMMARY {name}: mean_score={mean_score:.0f}, mean_winrate={mean_winrate:.1%}")

    # Final comparison
    print(f"\n\n{'='*70}\n FINAL COMPARISON\n{'='*70}")
    print(f"{'checkpoint':<45} {'mean_score':>11} {'mean_win%':>10} {'mean_rank':>10}")
    for name, per_N in all_results.items():
        ms = stats.mean([per_N[N]["mean_score"] for N in Ns])
        mw = stats.mean([per_N[N]["win_rate"] for N in Ns])
        mr = stats.mean([per_N[N]["mean_rank"] for N in Ns])
        print(f"{name:<45} {ms:>11.0f} {mw:>10.1%} {mr:>10.2f}")


if __name__ == "__main__":
    main()
