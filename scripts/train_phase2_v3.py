#!/usr/bin/env python3
"""Phase 2 v3: N-player self-play with full v3 stack."""
import argparse
import os
import sys

NEXUS_DIR = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, NEXUS_DIR)

import torch
from config import Config
from training.trainer_v3 import TrainerV3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iterations", type=int, default=400)
    ap.add_argument("--games-per-iter", type=int, default=Config.GAMES_PER_ITERATION)
    ap.add_argument("--training-steps", type=int, default=Config.TRAINING_STEPS_PER_ITER)
    ap.add_argument("--lr", type=float, default=Config.LEARNING_RATE)
    ap.add_argument("--resume", type=str, default=None)
    ap.add_argument("--start-iter", type=int, default=0)
    ap.add_argument("--checkpoint-dir", default="checkpoints_v3")
    ap.add_argument("--run-name", default="phase2_v3")
    args = ap.parse_args()

    trainer = TrainerV3(
        checkpoint_dir=args.checkpoint_dir,
        run_name=args.run_name,
    )
    if args.resume and os.path.exists(args.resume):
        print(f"[Phase2-v3] Loading checkpoint: {args.resume}")
        sd = torch.load(args.resume, map_location=trainer.device, weights_only=True)
        trainer.network.load_state_dict(sd)

    trainer.run_phase2(
        num_iterations=args.iterations,
        games_per_iter=args.games_per_iter,
        training_steps=args.training_steps,
        lr=args.lr,
        start_iter=args.start_iter,
    )
    print(f"\n[OK] Phase2-v3 complete. Best mean_score={trainer.best_mean_score:.1f} "
          f"at iter={trainer.best_iter}")
    print(f"     Logs: {trainer.run_dir}")
    print(f"     Best: {args.checkpoint_dir}/phase2_best_v3.pt")


if __name__ == "__main__":
    main()
