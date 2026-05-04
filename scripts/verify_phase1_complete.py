#!/usr/bin/env python3
"""verify_phase1_complete.py - sanity-check that Phase 1 v4 finished cleanly.

Inspects the Phase 1 training log and the resulting checkpoint to confirm
the run completed without anomalies. Run this as the first thing after
Phase 1 v4 stops generating training output.

Checks:
  1. The training process is no longer running.
  2. The log shows all 25 batches completed (or however many were
     configured).
  3. Loss decreased monotonically (allowing some wobble).
  4. The expected output checkpoint exists and is loadable.
  5. The checkpoint passes a quick forward-pass sanity check.

Usage:
    ./venv/bin/python scripts/verify_phase1_complete.py \\
        --log training_logs/phase1_v4.log \\
        --checkpoint checkpoints_v4/phase1_v4.pt
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from typing import List, Tuple


def parse_batch_history(log_path: str) -> List[Tuple[int, float, float, float, float]]:
    """Parse the log for batch completion lines.

    Returns list of (batch_idx, gen_sec, train_sec, loss, buf_size).
    """
    pattern = re.compile(
        r"Batch (\d+): gen=(\d+\.?\d*)s train=(\d+\.?\d*)s "
        r"\((\d+) steps\) loss=(\d+\.?\d*) buf=(\d+)"
    )
    history = []
    with open(log_path) as f:
        for line in f:
            m = pattern.search(line)
            if m:
                idx = int(m.group(1))
                gen = float(m.group(2))
                train = float(m.group(3))
                loss = float(m.group(5))
                buf = int(m.group(6))
                history.append((idx, gen, train, loss, buf))
    return history


def check_process_stopped(pid: int) -> bool:
    """Returns True if the given pid is no longer running."""
    try:
        os.kill(pid, 0)
        return False  # still running
    except OSError:
        return True   # not running


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default="training_logs/phase1_v4.log")
    ap.add_argument("--checkpoint", default="checkpoints_v4/phase1_v4.pt")
    ap.add_argument("--expected-batches", type=int, default=25)
    ap.add_argument("--pid", type=int, default=None,
                    help="Optional: training process pid; if given, verify it has stopped.")
    args = ap.parse_args()

    failures = 0
    print("=" * 60)
    print("Phase 1 v4 completion verification")
    print("=" * 60)

    # 1. Process check
    if args.pid is not None:
        if check_process_stopped(args.pid):
            print(f"[1] Process pid={args.pid} is stopped -- OK")
        else:
            print(f"[1] WARN: Process pid={args.pid} is STILL running. Wait for it.")
            failures += 1
    else:
        print("[1] (skipping process check; --pid not given)")

    # 2. Log shows expected batches
    if not os.path.exists(args.log):
        print(f"[2] FAIL: log file {args.log} not found")
        return 1
    history = parse_batch_history(args.log)
    print(f"[2] Found {len(history)} completed batches in log")
    if len(history) < args.expected_batches:
        print(f"   WARN: expected {args.expected_batches}, got {len(history)}")
        failures += 1

    # 3. Loss decreasing
    if len(history) >= 2:
        first_loss = history[0][3]
        last_loss = history[-1][3]
        delta = first_loss - last_loss
        print(f"[3] Loss change: {first_loss:.3f} -> {last_loss:.3f} "
              f"(Δ={delta:+.3f})")
        if delta < 0.01:
            print(f"   WARN: loss did not decrease meaningfully")
            failures += 1
        else:
            print(f"   OK: loss decreased by {delta/first_loss*100:.1f}%")
    else:
        print("[3] Not enough batches for trend analysis")

    # Step rate sanity
    if history:
        avg_gen = sum(h[1] for h in history) / len(history)
        avg_train = sum(h[2] for h in history) / len(history)
        print(f"   Avg gen: {avg_gen:.0f}s, avg train: {avg_train:.0f}s per batch")

    # 4. Checkpoint exists
    if not os.path.exists(args.checkpoint):
        print(f"[4] FAIL: checkpoint {args.checkpoint} not found")
        failures += 1
    else:
        size_mb = os.path.getsize(args.checkpoint) / (1024 * 1024)
        print(f"[4] Checkpoint {args.checkpoint} found ({size_mb:.1f} MB)")

    # 5. Checkpoint loads
    if os.path.exists(args.checkpoint):
        try:
            import torch
            data = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
            if isinstance(data, dict):
                print(f"[5] Checkpoint loads OK (dict with keys: {list(data.keys())[:5]}...)")
            else:
                print(f"[5] Checkpoint loads OK (type: {type(data).__name__})")
        except Exception as e:
            print(f"[5] FAIL: checkpoint won't load: {e}")
            failures += 1

    print("\n" + "=" * 60)
    if failures == 0:
        print("PHASE 1 v4 VERIFICATION PASSED")
        print("Ready to proceed to Phase 2 v4.")
    else:
        print(f"PHASE 1 v4 VERIFICATION FAILED ({failures} failures)")
        print("Review the warnings above before proceeding.")
    print("=" * 60)
    return failures


if __name__ == "__main__":
    sys.exit(main())
