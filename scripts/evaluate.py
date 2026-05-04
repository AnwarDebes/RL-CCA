#!/usr/bin/env python3
"""Evaluate agent strength vs baselines with fair color alternation."""
import sys
sys.path.insert(0, '/home/coder/nexus')

import argparse
import random

import torch

from config import Config
from core.board import HexBoard
from core.game_env import GameEnv
from core.action_space import get_legal_actions
from network.model import NexusNet
from training.heuristic_agent import HeuristicAgent
from mcts.gumbel_mcts import GumbelMCTS


def evaluate_vs_random(network, device, board, num_games=100):
    """Network vs random, alternating colors for fairness."""
    network.eval()
    wins = 0
    total_moves = 0
    with torch.no_grad():
        for g in range(num_games):
            env = GameEnv(board)
            env.reset()
            net_player = g % 2

            while not env.is_done():
                p = env.current_player
                if p == net_player:
                    state = env.get_state_tensor(p).unsqueeze(0).to(device)
                    mask = env.get_legal_mask(p).unsqueeze(0).to(device)
                    out = network(state, mask)
                    action = out['policy'][0].argmax().item()
                else:
                    legal = get_legal_actions(env.get_legal_mask(p))
                    action = random.choice(legal)
                env.step(action)

            if env.get_winner() == net_player:
                wins += 1
            total_moves += env.move_count
    return wins / num_games, total_moves / num_games


def evaluate_vs_heuristic(network, device, board, num_games=100):
    """Network vs heuristic, alternating colors for fairness."""
    network.eval()
    heuristic = HeuristicAgent(board)
    wins = 0
    total_moves = 0
    with torch.no_grad():
        for g in range(num_games):
            env = GameEnv(board)
            env.reset()
            net_player = g % 2

            while not env.is_done():
                p = env.current_player
                if p == net_player:
                    state = env.get_state_tensor(p).unsqueeze(0).to(device)
                    mask = env.get_legal_mask(p).unsqueeze(0).to(device)
                    out = network(state, mask)
                    action = out['policy'][0].argmax().item()
                else:
                    action = heuristic.choose_move(env, p)
                env.step(action)

            if env.get_winner() == net_player:
                wins += 1
            total_moves += env.move_count
    return wins / num_games, total_moves / num_games


def evaluate_with_mcts(network, device, board, num_games=20, num_sims=32):
    """Network+MCTS vs heuristic, alternating colors for fairness."""
    network.eval()
    heuristic = HeuristicAgent(board)
    wins = 0
    total_moves = 0
    with torch.no_grad():
        for g in range(num_games):
            env = GameEnv(board)
            env.reset()
            mcts = GumbelMCTS(network, device, num_simulations=num_sims, add_noise=False)
            net_player = g % 2

            while not env.is_done():
                p = env.current_player
                if p == net_player:
                    action, _, _ = mcts.search(env)
                else:
                    action = heuristic.choose_move(env, p)
                env.step(action)

            if env.get_winner() == net_player:
                wins += 1
            total_moves += env.move_count
    return wins / num_games, total_moves / num_games


def main():
    parser = argparse.ArgumentParser(description='NEXUS v2 Evaluation')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--num-games', type=int, default=100)
    parser.add_argument('--mcts-games', type=int, default=20)
    parser.add_argument('--mcts-sims', type=int, default=32)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    board = HexBoard()
    network = NexusNet.load(args.checkpoint, str(device))

    print(f"Evaluating {args.checkpoint} on {device}")
    print(f"All evaluations alternate colors for fairness")
    print(f"{'='*50}")

    wr_random, moves_random = evaluate_vs_random(network, device, board, args.num_games)
    print(f"vs Random:    {wr_random:.1%} win rate, {moves_random:.0f} avg moves")

    wr_heur, moves_heur = evaluate_vs_heuristic(network, device, board, args.num_games)
    print(f"vs Heuristic: {wr_heur:.1%} win rate, {moves_heur:.0f} avg moves")

    wr_mcts, moves_mcts = evaluate_with_mcts(
        network, device, board, args.mcts_games, args.mcts_sims
    )
    print(f"vs Heuristic (MCTS {args.mcts_sims} sims): {wr_mcts:.1%} win rate, "
          f"{moves_mcts:.0f} avg moves")


if __name__ == '__main__':
    main()
