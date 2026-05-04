#!/usr/bin/env python3
"""Phase 2: N-player self-play training with curriculum + heuristic mix.

Run from /home/coder/nexus (after Phase 1 has saved checkpoints_v2/phase1.pt):
  ./venv/bin/python scripts/train_phase2.py
  ./venv/bin/python scripts/train_phase2.py --iterations 500 --resume checkpoints_v2/phase2_latest.pt
"""
import sys
import os
import argparse

NEXUS_DIR = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, NEXUS_DIR)

import torch
from config import Config
from training.trainer import Trainer


def main():
    ap = argparse.ArgumentParser(description="NEXUS v3 Phase 2: N-player self-play")
    ap.add_argument("--iterations", type=int, default=600)
    ap.add_argument("--games-per-iter", type=int, default=Config.GAMES_PER_ITERATION)
    ap.add_argument("--training-steps", type=int, default=Config.TRAINING_STEPS_PER_ITER)
    ap.add_argument("--lr", type=float, default=Config.LEARNING_RATE)
    ap.add_argument("--use-mcts", action="store_true",
                    help="Use sequential MCTS self-play (slower, higher quality)")
    ap.add_argument("--resume", type=str, default=None,
                    help="Checkpoint to load (e.g., checkpoints_v2/phase1.pt)")
    ap.add_argument("--start-iter", type=int, default=0)
    ap.add_argument("--checkpoint-dir", default="checkpoints_v2")
    ap.add_argument("--run-name", default=None,
                    help="Logger run name (default: timestamped)")
    args = ap.parse_args()

    trainer = Trainer(
        checkpoint_dir=args.checkpoint_dir,
        run_name=args.run_name or "phase2",
    )
    if args.resume and os.path.exists(args.resume):
        print(f"[Phase2] Loading checkpoint: {args.resume}")
        sd = torch.load(args.resume, map_location=trainer.device, weights_only=True)
        trainer.network.load_state_dict(sd)

    trainer.run_phase2(
        num_iterations=args.iterations,
        games_per_iter=args.games_per_iter,
        training_steps=args.training_steps,
        lr=args.lr,
        start_iter=args.start_iter,
        use_mcts=args.use_mcts,
    )
    print(f"\n[OK] Phase 2 complete. Best mean_score={trainer.best_mean_score:.1f} "
          f"at iter={trainer.best_iter}")
    print(f"     Logs: {trainer.run_dir}")
    print(f"     Best: {args.checkpoint_dir}/phase2_best.pt")


if __name__ == "__main__":
    main()
