#!/usr/bin/env python3
"""Ablation: does MCTS recover if aggregate_value uses only v_win?

Hypothesis: v_pins / v_moves / v_dist are trained with terminal-state-derived
targets that produce noisy correlated duplicates of v_win. If true, masking
them out in MCTS aggregation should restore (or surpass) greedy-policy strength.
"""
import sys
sys.path.insert(0, '/home/coder/nexus')

import argparse
import torch

from core.board import HexBoard
from core.game_env import GameEnv
from network.model import NexusNet
from training.heuristic_agent import HeuristicAgent
from mcts.gumbel_mcts import GumbelMCTS


def make_aggregate(weights):
    """Build a replacement aggregate_value method with given fixed weights."""
    w = torch.tensor(weights, dtype=torch.float32)

    def aggregate_value(self, value):
        ww = w.to(value.device).to(value.dtype)
        return (value * ww).sum(dim=-1)

    return aggregate_value


def eval_mcts(network, device, board, num_games, num_sims):
    network.eval()
    heuristic = HeuristicAgent(board)
    wins = 0
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
    return wins / num_games


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--num-games', type=int, default=30)
    parser.add_argument('--num-sims', type=int, default=32)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    board = HexBoard()
    network = NexusNet.load(args.checkpoint, str(device))

    print(f"Checkpoint: {args.checkpoint}")
    print(f"Device: {device}, MCTS sims: {args.num_sims}, games per condition: {args.num_games}")
    print("=" * 60)

    configs = [
        ('default (0.45/0.25/0.00/0.30)', [0.45, 0.25, 0.0, 0.30]),
        ('v_win only   (1.00/0.00/0.00/0.00)', [1.0, 0.0, 0.0, 0.0]),
        ('v_win heavy  (0.90/0.05/0.00/0.05)', [0.9, 0.05, 0.0, 0.05]),
    ]

    for label, weights in configs:
        NexusNet.aggregate_value = make_aggregate(weights)
        wr = eval_mcts(network, device, board, args.num_games, args.num_sims)
        print(f"  {label:42s} -> MCTS vs heuristic win rate: {wr:.1%}")


if __name__ == '__main__':
    main()
