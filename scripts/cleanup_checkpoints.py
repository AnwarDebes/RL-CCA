#!/usr/bin/env python3
"""cleanup_checkpoints.py - prune old checkpoints to save disk.

Strategy: keep the latest checkpoint + every Nth milestone (default
every 10) + the explicitly-named final.pt. Delete the rest.

Safety: dry-run by default. Pass --apply to actually delete.

Usage:
    # See what would be deleted:
    python scripts/cleanup_checkpoints.py --dir checkpoints/cdmcts_cc_seed0

    # Actually delete:
    python scripts/cleanup_checkpoints.py --dir checkpoints/cdmcts_cc_seed0 --apply
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from typing import List, Tuple


def _parse_iter(filename: str) -> int:
    """Extract iteration number from `iter_NNNN.pt`. Returns -1 if not matched."""
    m = re.match(r"iter_(\d+)\.pt$", filename)
    return int(m.group(1)) if m else -1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="Checkpoint directory to clean")
    ap.add_argument("--keep-every", type=int, default=10,
                    help="Keep every Nth iter checkpoint as a milestone")
    ap.add_argument("--apply", action="store_true",
                    help="Actually delete (dry-run by default)")
    args = ap.parse_args()

    if not os.path.isdir(args.dir):
        print(f"FAIL: {args.dir} is not a directory")
        return 1

    # List all .pt files
    pt_files = []
    for f in sorted(os.listdir(args.dir)):
        path = os.path.join(args.dir, f)
        if f.endswith(".pt") and os.path.isfile(path):
            pt_files.append((f, path, _parse_iter(f), os.path.getsize(path)))

    if not pt_files:
        print(f"No .pt files in {args.dir}")
        return 0

    # Decide what to keep
    to_keep = []
    to_delete = []
    iter_files = [f for f in pt_files if f[2] >= 0]
    iter_files.sort(key=lambda x: x[2])
    if iter_files:
        latest_iter = iter_files[-1][2]
        for f, path, it, size in iter_files:
            keep = (it == latest_iter) or (it % args.keep_every == 0)
            (to_keep if keep else to_delete).append((f, path, it, size))
    # Always keep non-iter named files (e.g. final.pt, phase1_v4.pt)
    for f, path, it, size in pt_files:
        if it < 0:
            to_keep.append((f, path, it, size))

    total_size_mb = sum(s for _, _, _, s in pt_files) / (1024 * 1024)
    keep_size_mb = sum(s for _, _, _, s in to_keep) / (1024 * 1024)
    delete_size_mb = sum(s for _, _, _, s in to_delete) / (1024 * 1024)

    print(f"Directory: {args.dir}")
    print(f"  Total .pt files: {len(pt_files)} ({total_size_mb:.1f} MB)")
    print(f"  Will keep:       {len(to_keep)} ({keep_size_mb:.1f} MB)")
    print(f"  Will delete:     {len(to_delete)} ({delete_size_mb:.1f} MB)")
    print()
    print("Keep:")
    for f, _, _, size in sorted(to_keep, key=lambda x: x[2]):
        print(f"  {f:<24} ({size/1024/1024:.1f} MB)")
    print()
    print("Delete:")
    for f, _, _, size in sorted(to_delete, key=lambda x: x[2]):
        print(f"  {f:<24} ({size/1024/1024:.1f} MB)")

    if not args.apply:
        print(f"\nDRY RUN - no files deleted. Pass --apply to delete.")
        return 0

    if not to_delete:
        print("\nNothing to delete.")
        return 0

    print()
    n_deleted = 0
    for f, path, _, _ in to_delete:
        try:
            os.unlink(path)
            n_deleted += 1
        except Exception as e:
            print(f"  FAIL to delete {f}: {e}")
    print(f"Deleted {n_deleted} files, freed ~{delete_size_mb:.1f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
