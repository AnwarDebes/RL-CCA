#!/usr/bin/env python3
"""Phase 1 v4: heuristic bootstrap with all v4 aux targets."""
import argparse
import os
import sys
import time

NEXUS_DIR = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, NEXUS_DIR)

import torch
import torch.optim as optim

from config import Config
from training.trainer_v4 import TrainerV4, _split_param_groups
from training.phase1_v4 import play_games_parallel_v4


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-games", type=int, default=Config.PHASE1_NUM_GAMES)
    ap.add_argument("--epochs", type=int, default=Config.PHASE1_EPOCHS)
    ap.add_argument("--lr", type=float, default=Config.PHASE1_LR)
    ap.add_argument("--checkpoint-dir", default="checkpoints_v4")
    ap.add_argument("--run-name", default="phase1_v4")
    ap.add_argument("--num-workers", type=int, default=None)
    args = ap.parse_args()

    trainer = TrainerV4(
        checkpoint_dir=args.checkpoint_dir,
        run_name=args.run_name,
    )

    BATCH_GAMES = 2000
    num_batches = max(1, (args.num_games + BATCH_GAMES - 1) // BATCH_GAMES)
    STEPS_PER_FULL_EPOCH = max(1, Config.REPLAY_BUFFER_SIZE // Config.BATCH_SIZE)
    steps_per_data_batch = max(1, (STEPS_PER_FULL_EPOCH * args.epochs) // num_batches)

    trainer.network.train()
    trainer.optimizer = optim.AdamW(_split_param_groups(trainer.network), lr=args.lr)

    print(f"[Phase1-v4] {num_batches} batches × {steps_per_data_batch} train steps")

    for batch_idx in range(num_batches):
        t0 = time.time()
        print(f"[Phase1-v4] Batch {batch_idx+1}/{num_batches} - generating "
              f"{BATCH_GAMES} mixed-N heuristic games")
        trajs = play_games_parallel_v4(
            num_games=BATCH_GAMES, num_workers=args.num_workers,
        )
        gen_sec = time.time() - t0
        print(f"[Phase1-v4]   gen complete: {gen_sec:.1f}s, {len(trajs)} valid trajectories", flush=True)
        t_buf = time.time()
        for traj in trajs:
            for entry in traj:
                trainer._add_to_buffer(entry)
        print(f"[Phase1-v4]   buffer fill: {time.time()-t_buf:.1f}s, buf={len(trainer.replay_buffer)}", flush=True)

        tr_t0 = time.time()
        loss_sum = 0.0
        for step in range(steps_per_data_batch):
            s = trainer._train_step(Config.BATCH_SIZE)
            loss_sum += s["loss_total"]
            if step == 0 or step == 9 or step == 49 or step % 100 == 99:
                print(f"[Phase1-v4]   step {step+1}/{steps_per_data_batch}: loss={s['loss_total']:.3f} ({(time.time()-tr_t0)/(step+1)*1000:.0f}ms/step)", flush=True)
        avg = loss_sum / max(1, steps_per_data_batch)
        train_sec = time.time() - tr_t0
        print(f"[Phase1-v4] Batch {batch_idx+1}: gen={gen_sec:.1f}s "
              f"train={train_sec:.1f}s ({steps_per_data_batch} steps) "
              f"loss={avg:.3f} buf={len(trainer.replay_buffer)}")

    out_path = os.path.join(trainer.checkpoint_dir, "phase1_v4.pt")
    trainer.network.save(out_path)
    print(f"[Phase1-v4] Saved {out_path}")


if __name__ == "__main__":
    main()
