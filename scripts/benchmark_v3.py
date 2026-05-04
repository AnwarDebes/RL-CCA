#!/usr/bin/env python3
"""Quick v3 benchmark - evaluates a checkpoint against Random and Heuristic
opponents across N=2..6. Reports win rate, mean rank, and mean teacher final_score.

Usage:
  python scripts/benchmark_v3.py --model checkpoints_v3/phase1_v3.pt --games-per-N 20
"""
from __future__ import annotations
import argparse
import os
import random
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


def _net_action(net, env, player, device, is_v3, greedy=True):
    state = env.get_state_tensor(player).unsqueeze(0).to(device)
    mask = env.get_legal_mask(player).unsqueeze(0).to(device)
    with torch.no_grad():
        if is_v3:
            seat_t = torch.tensor([player], device=device)
            out = net(state, mask, current_seat=seat_t)
        else:
            out = net(state, mask)
    pol = out["policy"][0].cpu().numpy()
    if greedy:
        return int(np.argmax(pol))
    legal = get_legal_actions(env.get_legal_mask(player))
    if not legal:
        return 0
    p = pol.astype(np.float64)
    s = p.sum()
    return int(np.random.choice(len(p), p=p/s)) if s > 0 else int(np.argmax(pol))


def eval_vs_opponent(net, device, is_v3, N: int, n_games: int, opponent: str,
                     verbose: bool = False):
    """Play n_games of N-player games. The network plays one seat (rotating),
    other seats are filled with `opponent` (random | heuristic).
    Returns dict with mean_score, mean_rank, win_rate, avg_moves.
    """
    board = HexBoard()
    heuristic = HeuristicAgent(board) if opponent == "heuristic" else None
    rng = random.Random(20260501 + N)
    nexus_scores = []
    nexus_ranks = []
    move_lengths = []
    wins = 0
    for g in range(n_games):
        env = GameEnv(board, num_players=N)
        env.reset(random_colors=True, rng=rng)
        net_seat = g % N
        while not env.is_done():
            p = env.current_player
            if p == net_seat:
                action = _net_action(net, env, p, device, is_v3, greedy=True)
            else:
                if opponent == "heuristic":
                    action = heuristic.choose_move(env, p)
                else:  # random
                    legal = get_legal_actions(env.get_legal_mask(p))
                    action = rng.choice(legal) if legal else 0
            env.step(action)
        score = env.compute_final_score(net_seat)
        nexus_scores.append(score)
        # rank: 1 = highest final_score
        all_scores = [env.compute_final_score(p) for p in range(N)]
        sorted_scores = sorted(all_scores, reverse=True)
        rank = sorted_scores.index(score) + 1
        nexus_ranks.append(rank)
        move_lengths.append(env.move_count)
        if env.get_winner() == net_seat:
            wins += 1
        if verbose:
            print(f"    game {g}: seat={net_seat} score={score:.0f} rank={rank} moves={env.move_count}")
    return {
        "N": N, "opponent": opponent, "games": n_games,
        "mean_score": float(np.mean(nexus_scores)),
        "median_score": float(np.median(nexus_scores)),
        "mean_rank": float(np.mean(nexus_ranks)),
        "win_rate": wins / n_games,
        "avg_moves": float(np.mean(move_lengths)),
    }


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

    print(f"\n=== v3 Benchmark: {args.model} ===")
    print(f"Detected: {'NexusNetV3' if is_v3 else 'NexusNet (v2)'}")
    print(f"Device: {device}, games/N: {args.games_per_N}")
    print()

    for opponent in ["random", "heuristic"]:
        print(f"--- vs {opponent.upper()} ---")
        print(f"{'N':>3}  {'mean_score':>10}  {'rank':>5}  {'win_rate':>9}  {'avg_moves':>10}  {'sec':>5}")
        for N in Ns:
            t0 = time.time()
            r = eval_vs_opponent(net, device, is_v3, N, args.games_per_N, opponent)
            print(f"{N:>3}  {r['mean_score']:>10.0f}  {r['mean_rank']:>5.2f}  "
                  f"{r['win_rate']:>9.0%}  {r['avg_moves']:>10.1f}  {int(time.time()-t0):>5}")
        print()


if __name__ == "__main__":
    main()
