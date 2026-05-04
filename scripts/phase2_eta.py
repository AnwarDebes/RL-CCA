#!/usr/bin/env python3
"""phase2_eta.py - predict when Phase 2 v4 training will finish.

Reads the Phase 2 log to find recent iteration timings, then projects
remaining wall-clock time based on the current rate.

Usage:
    ./venv/bin/python scripts/phase2_eta.py
"""

from __future__ import annotations

import os
import re
import sys
from datetime import datetime, timedelta


def parse_iter_lines(log_path: str) -> list:
    """Extract (iter_idx, total_iters, timestamp) from Phase 2 log.

    Looks for lines like:
        [Phase2-v4] [iter 5/250] ...
    Plus uses file mtimes as approximate timestamps if log lines lack them.
    """
    if not os.path.exists(log_path):
        return []
    pattern = re.compile(r"\[iter (\d+)/(\d+)\]")
    iters = []
    with open(log_path) as f:
        for line in f:
            m = pattern.search(line)
            if m:
                iters.append((int(m.group(1)), int(m.group(2))))
    return iters


def main():
    log = "training_logs/phase2_v4.log"
    iters = parse_iter_lines(log)
    if not iters:
        if os.path.exists(log):
            mtime = os.path.getmtime(log)
            age = (datetime.now().timestamp() - mtime)
            size = os.path.getsize(log)
            print(f"Phase 2 log exists but no iteration entries yet.")
            print(f"  Path: {log}")
            print(f"  Size: {size} bytes")
            print(f"  Modified: {age:.0f}s ago")
            if age < 300:
                print("  Phase 2 likely still initializing (model load, first generation).")
            else:
                print("  WARN: log is silent for >5 min. Check the process:")
                print("        ps -ef | grep train_phase2_v4")
        else:
            print(f"Phase 2 log not found at {log}")
            print("  Phase 2 has not been launched yet.")
            print("  Launch with: make launch-phase2")
        return 0

    cur, total = iters[-1]
    print(f"=== Phase 2 v4 progress ===")
    print(f"  Iterations completed: {cur} / {total} ({100.0 * cur / total:.1f}%)")

    if len(iters) >= 2:
        # Estimate per-iteration time from recent iters
        # We don't have per-line timestamps, use file mtime as a rough total-elapsed proxy.
        try:
            import psutil
            for proc in psutil.process_iter(["pid", "name", "cmdline", "create_time"]):
                cmd = " ".join(proc.info.get("cmdline") or [])
                if "train_phase2_v4" in cmd:
                    elapsed_sec = (datetime.now().timestamp() - proc.info["create_time"])
                    rate = elapsed_sec / max(1, cur)
                    remaining = (total - cur) * rate
                    eta = datetime.now() + timedelta(seconds=remaining)
                    print(f"  Process elapsed: {elapsed_sec/3600:.1f}h")
                    print(f"  Per-iter rate: {rate/60:.1f} min")
                    print(f"  Remaining: {remaining/3600:.1f}h ({remaining/3600/24:.1f} days)")
                    print(f"  Estimated finish: {eta.strftime('%Y-%m-%d %H:%M')}")
                    return 0
            print("  (process not running; using log mtime for elapsed estimate)")
        except ImportError:
            pass
        # Fallback: use log mtime
        mtime = os.path.getmtime(log)
        elapsed_sec = (datetime.now().timestamp() - mtime + cur * 60)  # rough
        print(f"  (psutil not available; install for accurate ETA)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
