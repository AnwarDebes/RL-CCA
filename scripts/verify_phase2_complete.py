#!/usr/bin/env python3
"""verify_phase2_complete.py - sanity-check that Phase 2 v4 finished cleanly.

Mirrors verify_phase1_complete.py but for the longer Phase 2 (250 iters
of self-play + MCTS-improved targets). Phase 2 produces the final
tournament agent at `checkpoints_v4/phase2_best_v4.pt`.

Usage:
    ./venv/bin/python scripts/verify_phase2_complete.py
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from typing import List, Tuple


def parse_phase2_history(log_path: str) -> List[dict]:
    """Phase 2 logs an iteration line per training iteration."""
    pattern = re.compile(
        r"Phase2-v4.*\[iter (\d+)/(\d+)\].*"
        r"loss=([\d.]+)"
    )
    history = []
    with open(log_path) as f:
        for line in f:
            m = pattern.search(line)
            if m:
                history.append(dict(
                    iter=int(m.group(1)),
                    total=int(m.group(2)),
                    loss=float(m.group(3)),
                ))
    return history


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default="training_logs/phase2_v4.log")
    ap.add_argument("--checkpoint", default="checkpoints_v4/phase2_best_v4.pt")
    ap.add_argument("--expected-iters", type=int, default=250)
    args = ap.parse_args()

    failures = 0
    print("=" * 60)
    print("Phase 2 v4 completion verification")
    print("=" * 60)

    if not os.path.exists(args.log):
        print(f"[1] FAIL: log {args.log} not found")
        return 1

    history = parse_phase2_history(args.log)
    print(f"[1] Found {len(history)} iteration entries in log")
    if len(history) < args.expected_iters:
        print(f"   WARN: expected {args.expected_iters}, got {len(history)}")
        failures += 1

    if len(history) >= 2:
        first_loss = history[0]["loss"]
        last_loss = history[-1]["loss"]
        delta = first_loss - last_loss
        print(f"[2] Loss change: {first_loss:.3f} -> {last_loss:.3f} (Δ={delta:+.3f})")
        if delta < 0.01:
            print(f"   WARN: loss did not decrease meaningfully")
            failures += 1
        else:
            pct = delta / first_loss * 100
            print(f"   OK: loss decreased by {pct:.1f}%")

    if not os.path.exists(args.checkpoint):
        print(f"[3] FAIL: final checkpoint {args.checkpoint} not found")
        failures += 1
    else:
        size_mb = os.path.getsize(args.checkpoint) / (1024 * 1024)
        print(f"[3] Final checkpoint OK ({size_mb:.1f} MB)")
        try:
            import torch
            data = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
            print(f"[4] Loads OK ({type(data).__name__})")
        except Exception as e:
            print(f"[4] FAIL: load error: {e}")
            failures += 1

    print("\n" + "=" * 60)
    if failures == 0:
        print("PHASE 2 v4 VERIFICATION PASSED")
        print("Tournament agent ready at:", args.checkpoint)
        print()
        print("Next: run the research-subproject experiments.")
        print("  See: WHAT_NEXT.md")
    else:
        print(f"PHASE 2 v4 VERIFICATION FAILED ({failures} failures)")
    print("=" * 60)
    return failures


if __name__ == "__main__":
    sys.exit(main())
