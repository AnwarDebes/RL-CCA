#!/usr/bin/env python3
"""Phase 1 v3: heuristic bootstrap with v3 aux targets."""
import argparse
import os
import sys

NEXUS_DIR = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, NEXUS_DIR)

from config import Config
from training.trainer_v3 import TrainerV3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-games", type=int, default=Config.PHASE1_NUM_GAMES)
    ap.add_argument("--epochs", type=int, default=Config.PHASE1_EPOCHS)
    ap.add_argument("--lr", type=float, default=Config.PHASE1_LR)
    ap.add_argument("--checkpoint-dir", default="checkpoints_v3")
    ap.add_argument("--run-name", default="phase1_v3")
    args = ap.parse_args()

    trainer = TrainerV3(
        checkpoint_dir=args.checkpoint_dir,
        run_name=args.run_name,
    )
    trainer.run_phase1(num_games=args.num_games, epochs=args.epochs, lr=args.lr)
    print(f"[OK] Phase1 v3 complete. Logs: {trainer.run_dir}")


if __name__ == "__main__":
    main()
