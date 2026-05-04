"""Pick the best (and median, and worst) seed from a sweep.

After running an experiment with multiple seeds, this picks the
"best" / "median" / "worst" by a chosen metric. CRITICAL: the paper
reports the *median* and *std*, not the *best* - using only the best
seed is cherry-picking and we explicitly call it out.

Usage:
    python -m flagship_coalition_mcts.src.find_best_seed \\
        --files results/seed_*.json \\
        --metric drop_all \\
        --report all
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, List, Tuple


def _extract_metric(data: Any, metric: str) -> float:
    """Extract a single metric value from a result dict."""
    if isinstance(data, dict):
        if metric in data:
            v = data[metric]
            if isinstance(v, (int, float)):
                return float(v)
            if isinstance(v, dict) and "mean" in v:
                return float(v["mean"])
        # Try nested look-up: metric="game.variant.metric"
        if "." in metric:
            parts = metric.split(".")
            cur = data
            for p in parts:
                if isinstance(cur, dict) and p in cur:
                    cur = cur[p]
                else:
                    return float("nan")
            if isinstance(cur, (int, float)):
                return float(cur)
    return float("nan")


def _report(name: str, path: str, value: float) -> None:
    print(f"  {name:<8} {value:>+.4f}  ({os.path.basename(path)})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", nargs="+", required=True)
    ap.add_argument("--metric", required=True,
                    help="Field name in the JSON (e.g. 'drop_all'). "
                         "Use dotted path for nested: 'kingmaker.A3.elo_gap'.")
    ap.add_argument("--higher-is-better", action="store_true", default=True)
    ap.add_argument("--lower-is-better", action="store_true", default=False)
    ap.add_argument("--report", choices=["all", "best", "median", "worst"], default="all")
    args = ap.parse_args()
    if args.lower_is_better:
        args.higher_is_better = False

    rows: List[Tuple[float, str, dict]] = []
    for path in args.files:
        with open(path) as f:
            data = json.load(f)
        v = _extract_metric(data, args.metric)
        rows.append((v, path, data))
    if not rows:
        print("No valid files.")
        return 1

    rows.sort(key=lambda r: r[0])
    if args.higher_is_better:
        rows.reverse()  # best first

    n = len(rows)
    median_idx = n // 2
    print(f"=== Seed sweep summary ({n} seeds, metric={args.metric}, "
          f"{'higher' if args.higher_is_better else 'lower'} is better) ===")

    if args.report in ("all", "best"):
        v, p, _ = rows[0]
        _report("BEST", p, v)
    if args.report in ("all", "median"):
        v, p, _ = rows[median_idx]
        _report("MEDIAN", p, v)
    if args.report in ("all", "worst"):
        v, p, _ = rows[-1]
        _report("WORST", p, v)

    if args.report == "all":
        # Compute summary statistics
        values = [r[0] for r in rows if r[0] == r[0]]  # filter NaN
        if values:
            mean = sum(values) / len(values)
            std = (sum((v - mean) ** 2 for v in values) / len(values)) ** 0.5
            print(f"\n  mean ± std = {mean:+.4f} ± {std:.4f}")
            print(f"  range      = [{min(values):+.4f}, {max(values):+.4f}]")
            print()
            print("  IMPORTANT: For the paper, report MEAN ± STD, not just BEST.")
            print("  Reporting only BEST is cherry-picking and reviewers will catch it.")


if __name__ == "__main__":
    main()
