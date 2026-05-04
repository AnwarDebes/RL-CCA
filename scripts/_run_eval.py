"""Final evaluation runner - importable by train_all.py."""
import os
import torch

from core.board import HexBoard
from network.model import NexusNet
from scripts.evaluate import evaluate_vs_random, evaluate_vs_heuristic, evaluate_with_mcts


def run_eval(checkpoint_dir='checkpoints'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    board = HexBoard()

    # Evaluate the best available checkpoint
    for name in ['phase2_best.pt', 'phase3_final.pt', 'phase2_final.pt']:
        path = os.path.join(checkpoint_dir, name)
        if os.path.exists(path):
            print(f"\n  Evaluating: {name}")
            network = NexusNet.load(path, str(device))

            wr_random, moves_random = evaluate_vs_random(network, device, board, 200)
            print(f"    vs Random:             {wr_random:.1%} win rate, {moves_random:.0f} avg moves")

            wr_heur, moves_heur = evaluate_vs_heuristic(network, device, board, 200)
            print(f"    vs Heuristic:          {wr_heur:.1%} win rate, {moves_heur:.0f} avg moves")

            wr_mcts, moves_mcts = evaluate_with_mcts(network, device, board, 50, 64)
            print(f"    vs Heuristic (MCTS64): {wr_mcts:.1%} win rate, {moves_mcts:.0f} avg moves")
