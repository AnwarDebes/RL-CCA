#!/usr/bin/env python3
"""Phase 1: heuristic bootstrap. Generates mixed-N games + supervised pretrain.

Run from /home/coder/nexus:
  ./venv/bin/python scripts/train_phase1.py [--games 50000] [--epochs 50]
"""
import sys
import os
import argparse

NEXUS_DIR = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, NEXUS_DIR)

from config import Config
from training.trainer import Trainer


def main():
    ap = argparse.ArgumentParser(description="NEXUS v3 Phase 1: heuristic bootstrap")
    ap.add_argument("--games", type=int, default=Config.PHASE1_NUM_GAMES)
    ap.add_argument("--epochs", type=int, default=Config.PHASE1_EPOCHS)
    ap.add_argument("--batch-size", type=int, default=Config.BATCH_SIZE)
    ap.add_argument("--lr", type=float, default=Config.PHASE1_LR)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--checkpoint-dir", default="checkpoints_v2")
    args = ap.parse_args()

    trainer = Trainer(checkpoint_dir=args.checkpoint_dir, run_name="phase1")
    trainer.run_phase1(
        num_games=args.games,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        num_workers=args.workers,
    )
    print(f"\n[OK] Phase 1 complete. Checkpoint: {args.checkpoint_dir}/phase1.pt")
    print(f"     Logs: {trainer.run_dir}")


if __name__ == "__main__":
    main()
