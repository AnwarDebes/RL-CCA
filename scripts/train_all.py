#!/usr/bin/env python3
"""NEXUS v2 - Full training pipeline (Phase 1 → 2 → 3 → Evaluate).

Run: python scripts/train_all.py
All output saved to checkpoints/ directory.
"""
import sys
sys.path.insert(0, '/home/coder/nexus')

import multiprocessing as mp
import time

from config import Config
from training.trainer import Trainer


def main():
    start = time.time()
    mp.set_start_method('spawn', force=True)

    trainer = Trainer(checkpoint_dir='checkpoints')
    print(f"Device: {trainer.device}")
    print(f"Network params: {sum(p.numel() for p in trainer.network.parameters()):,}")
    print(f"{'='*60}")

    # ── Phase 1: Heuristic Bootstrap ─────────────────────────────
    import os
    phase1_path = os.path.join('checkpoints', 'phase1.pt')
    if os.path.exists(phase1_path):
        print("\n" + "="*60)
        print(f"PHASE 1: SKIPPED - found existing {phase1_path}")
        print("="*60)
        p1_time = 0.0
    else:
        print("\n" + "="*60)
        print("PHASE 1: Heuristic Bootstrap (50k games, 30 epochs)")
        print("="*60)
        p1_start = time.time()

        trainer.run_phase1(
            num_games=Config.PHASE1_NUM_GAMES,
            epochs=Config.PHASE1_EPOCHS,
            batch_size=Config.BATCH_SIZE,
            lr=Config.PHASE1_LR,
            num_workers=Config.NUM_WORKERS,
        )

        p1_time = time.time() - p1_start
        print(f"\nPhase 1 finished in {p1_time/60:.1f} min")

    # ── Phase 2: Self-Play with Gumbel MCTS ──────────────────────
    print("\n" + "="*60)
    print("PHASE 2: Self-Play with MCTS (1000 iterations, 16 games/iter)")
    print("="*60)
    p2_start = time.time()

    trainer.run_phase2(
        num_iterations=1000,
        games_per_iter=Config.GAMES_PER_ITERATION,
        training_steps=Config.TRAINING_STEPS_PER_ITER,
        num_workers=Config.NUM_WORKERS,
        checkpoint_path='checkpoints/phase1.pt',
    )

    p2_time = time.time() - p2_start
    print(f"\nPhase 2 finished in {p2_time/3600:.1f} hours")

    # ── Phase 3: Population Elo Training ─────────────────────────
    print("\n" + "="*60)
    print("PHASE 3: Population Elo Training (200 rounds)")
    print("="*60)
    p3_start = time.time()

    from scripts._run_phase3 import run_phase3
    run_phase3(checkpoint_dir='checkpoints', rounds=200, match_games=50)

    p3_time = time.time() - p3_start
    print(f"\nPhase 3 finished in {p3_time/60:.1f} min")

    # ── Final Evaluation ─────────────────────────────────────────
    print("\n" + "="*60)
    print("FINAL EVALUATION")
    print("="*60)

    from scripts._run_eval import run_eval
    run_eval(checkpoint_dir='checkpoints')

    # ── Summary ──────────────────────────────────────────────────
    total = time.time() - start
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print(f"  Phase 1: {p1_time/60:.1f} min")
    print(f"  Phase 2: {p2_time/3600:.1f} hours")
    print(f"  Phase 3: {p3_time/60:.1f} min")
    print(f"  Total:   {total/3600:.1f} hours")
    print(f"\nCheckpoints saved in: checkpoints/")
    print(f"  phase1.pt          - heuristic bootstrap")
    print(f"  phase2_best.pt     - best self-play model (USE THIS)")
    print(f"  phase2_final.pt    - final self-play model")
    print(f"  phase3_final.pt    - population-trained model")
    print("="*60)


if __name__ == '__main__':
    main()
