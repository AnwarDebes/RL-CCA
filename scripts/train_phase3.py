#!/usr/bin/env python3
"""Phase 3: Population training with Elo matchmaking."""
import sys
sys.path.insert(0, '/home/coder/nexus')

import argparse
import copy
import os
import random

import torch

from config import Config
from core.board import HexBoard
from core.game_env import GameEnv
from core.action_space import get_legal_actions
from network.model import NexusNet
from training.elo import EloTracker
from training.self_play import generate_self_play_game
from training.replay_buffer import ReplayBuffer
from training.losses import nexus_loss


def play_match(net_a, net_b, board, device, num_games=50):
    """Play a match between two networks. Returns score for A (wins/games)."""
    wins_a = 0
    for g in range(num_games):
        env = GameEnv(board)
        env.reset()
        nets = [net_a, net_b] if g % 2 == 0 else [net_b, net_a]
        mapping = [0, 1] if g % 2 == 0 else [1, 0]

        while not env.is_done():
            p = env.current_player
            net = nets[p]
            net.eval()
            with torch.no_grad():
                state = env.get_state_tensor(p).unsqueeze(0).to(device)
                mask = env.get_legal_mask(p).unsqueeze(0).to(device)
                out = net(state, mask)
                action = out['policy'][0].argmax().item()
            env.step(action)

        winner = env.get_winner()
        if winner is not None:
            actual_winner = mapping[winner]
            if actual_winner == 0:
                wins_a += 1

    return wins_a / num_games


def main():
    parser = argparse.ArgumentParser(description='NEXUS v2 Phase 3: Population Training')
    parser.add_argument('--rounds', type=int, default=100)
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints')
    parser.add_argument('--match-games', type=int, default=50)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    board = HexBoard()

    # Load population from Phase 2 checkpoints
    population = {}
    elo_tracker = EloTracker(k=Config.PHASE3_ELO_K)

    for f in sorted(os.listdir(args.checkpoint_dir)):
        if f.startswith('phase2_') and f.endswith('.pt'):
            path = os.path.join(args.checkpoint_dir, f)
            net = NexusNet.load(path, str(device))
            net.eval()
            agent_id = f.replace('.pt', '')
            population[agent_id] = net
            elo_tracker.register(agent_id)
            print(f"Loaded {agent_id}")

    if len(population) < 2:
        print("Need at least 2 checkpoints for population training.")
        return

    current_id = max(population.keys())
    current_net = population[current_id]

    for round_idx in range(args.rounds):
        # Select opponent
        opp_id = elo_tracker.get_opponent_in_range(current_id, 200.0)
        if opp_id is None:
            opp_id = random.choice([k for k in population if k != current_id])

        opp_net = population[opp_id]
        score = play_match(current_net, opp_net, board, device, args.match_games)
        elo_tracker.record_match(current_id, opp_id, score)

        print(f"Round {round_idx}: {current_id} vs {opp_id} - "
              f"score={score:.2f}, Elo={elo_tracker.get_rating(current_id):.0f}")

        # Add snapshot every 20 rounds
        if round_idx % 20 == 0 and round_idx > 0:
            snap_id = f"phase3_r{round_idx}"
            population[snap_id] = copy.deepcopy(current_net)
            elo_tracker.register(snap_id)

    # Save final
    path = os.path.join(args.checkpoint_dir, 'phase3_final.pt')
    current_net.save(path)
    print(f"\nFinal Elo ratings:")
    for aid, rating in sorted(elo_tracker.ratings.items(), key=lambda x: -x[1]):
        print(f"  {aid}: {rating:.0f}")


if __name__ == '__main__':
    main()
