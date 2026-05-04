"""Multi-seed sweep utility: run any experiment script with multiple seeds.

For paper-quality results, every experiment should be repeated with
≥3 seeds for variance estimation. This script automates that and
collects the JSON outputs for results_table.py to aggregate.

Usage:
    python -m flagship_coalition_mcts.experiments.seed_sweep \\
        --script flagship_coalition_mcts.experiments.kingmaker_head_to_head \\
        --seeds 0 1 2 3 4 \\
        --extra "--num-iterations 20 --games-per-iter 16" \\
        --out-pattern "results/kingmaker_h2h_seed{seed}.json"

The {seed} placeholder in --out-pattern is replaced per run.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--script", required=True,
        help="Module path to the experiment script (e.g. flagship_coalition_mcts.experiments.kingmaker_head_to_head)",
    )
    ap.add_argument(
        "--seeds", nargs="+", type=int, required=True,
        help="List of seeds (e.g. 0 1 2)",
    )
    ap.add_argument(
        "--extra", default="",
        help="Extra args to pass through verbatim (e.g. \"--num-iterations 20\")",
    )
    ap.add_argument(
        "--out-pattern", default="results/sweep_seed{seed}.json",
        help="Output path with {seed} placeholder",
    )
    ap.add_argument("--continue-on-error", action="store_true",
                    help="If one seed fails, continue with the rest")
    args = ap.parse_args()

    venv = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..",
                        "venv", "bin", "python")
    if not os.path.exists(venv):
        venv = sys.executable

    print(f"=== Seed sweep ===")
    print(f"Script: {args.script}")
    print(f"Seeds: {args.seeds}")
    print(f"Extra args: {args.extra}")

    results = []
    for seed in args.seeds:
        out_path = args.out_pattern.format(seed=seed)
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        cmd = [
            venv, "-m", args.script,
            "--seed", str(seed),
            "--out", out_path,
        ]
        # Append extra args
        if args.extra:
            cmd.extend(args.extra.split())
        print(f"\n[seed={seed}] {' '.join(cmd)}")
        t0 = time.time()
        try:
            subprocess.run(cmd, check=True)
            elapsed = time.time() - t0
            print(f"[seed={seed}] OK in {elapsed:.0f}s -> {out_path}")
            results.append((seed, out_path, "ok"))
        except subprocess.CalledProcessError as e:
            elapsed = time.time() - t0
            print(f"[seed={seed}] FAILED after {elapsed:.0f}s: {e}")
            results.append((seed, out_path, "failed"))
            if not args.continue_on_error:
                print("Aborting sweep; use --continue-on-error to skip failures.")
                sys.exit(1)

    print("\n=== Sweep complete ===")
    for seed, path, status in results:
        print(f"  seed={seed:>3}: {status:<7} {path}")
    print(f"\nAggregate with: python -m flagship_coalition_mcts.src.results_table "
          f"--files {' '.join(p for _, p, s in results if s == 'ok')}")


if __name__ == "__main__":
    main()
