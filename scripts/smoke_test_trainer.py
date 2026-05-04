#!/usr/bin/env python3
"""Smoke test: run 1 iter of Phase 2 with tiny settings (no server eval).

Verifies the full training loop wires together end-to-end:
- Self-play generates trajectories with N-player curriculum
- Trajectories add to replay buffer
- Training step computes loss without errors
- Logger writes iter_metrics.jsonl + status.txt
"""
import sys
import os
NEXUS_DIR = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, NEXUS_DIR)

from config import Config
# Minimize all knobs for the smoke test
Config.GAMES_PER_ITERATION = 4
Config.TRAINING_STEPS_PER_ITER = 5
Config.BATCH_SIZE = 32
Config.EVAL_INPROC_EVERY = 1
Config.EVAL_INPROC_GAMES = 2
Config.EVAL_SERVER_EVERY = 999    # skip server eval in smoke
Config.RULE_ALIGNMENT_EVERY = 999  # skip alignment in smoke
Config.NUM_PLAYERS_CURRICULUM = {0: {2: 0.4, 3: 0.3, 4: 0.3}}
Config.VS_HEURISTIC_FRACTION = 0.50

from training.trainer import Trainer

trainer = Trainer(checkpoint_dir="/tmp/nexus_smoke_ckpt", run_name="smoke")
trainer.run_phase2(num_iterations=1, games_per_iter=4, training_steps=5)
print("\n[SMOKE OK]")
print(f"  run dir: {trainer.run_dir}")
print(f"  status.txt:")
with open(os.path.join(trainer.run_dir, "status.txt")) as f:
    print("  " + f.read().replace("\n", "\n  "))
