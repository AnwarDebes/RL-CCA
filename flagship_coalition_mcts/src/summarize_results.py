"""One-line-per-file results browsing.

When you have many seed runs (`results/seed_0.json`, `seed_1.json`, ...),
this gives a compact text summary of each. Useful for spot-checking a
sweep before running results_table aggregation.

Usage:
    python -m flagship_coalition_mcts.src.summarize_results results/*.json
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict


def _short_summary(data: Dict[str, Any], path: str) -> str:
    """One-line summary of a result JSON. Format depends on schema."""
    base = os.path.basename(path)

    # Kingmaker H2H format
    if "p0_winrate_scalar" in data:
        s = data["p0_winrate_scalar"]
        c = data["p0_winrate_cd"]
        m = data["p0_winrate_mixed"]
        passed = data.get("passed_pre_registered", False)
        marker = "PASS" if passed else "FAIL"
        return (
            f"  [{marker}] {base:<40} "
            f"P0 win-rate scalar={s:.2f} cd={c:.2f} mixed={m:.2f}"
        )

    # Ablation ladder format
    if isinstance(data, dict) and any(isinstance(v, dict) for v in data.values()):
        # Dict of game -> variant -> stats
        games = list(data.keys())
        variants = set()
        for game in games:
            v = data[game]
            if isinstance(v, dict):
                variants.update(v.keys())
        return (
            f"  [abl] {base:<40} "
            f"{len(games)} game(s): {games}, "
            f"{len(variants)} variant(s): {sorted(variants)}"
        )

    # CCE-gap history format (list of {iter, cce_gap, ...})
    if isinstance(data, list) and data and "cce_gap" in data[0]:
        first = data[0]["cce_gap"]
        last = data[-1]["cce_gap"]
        delta = first - last
        return (
            f"  [cce]  {base:<40} "
            f"{len(data)} checkpoints, gap {first:.3f} -> {last:.3f} "
            f"(Δ={delta:+.3f})"
        )

    # Generic
    return f"  [?]    {base:<40} (unrecognised schema)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    args = ap.parse_args()

    print(f"=== Results summary: {len(args.files)} file(s) ===")
    for path in sorted(args.files):
        if not os.path.exists(path):
            print(f"  [?]    {os.path.basename(path):<40} (file not found)")
            continue
        try:
            with open(path) as f:
                data = json.load(f)
        except Exception as e:
            print(f"  [?]    {os.path.basename(path):<40} (parse error: {e})")
            continue
        print(_short_summary(data, path))


if __name__ == "__main__":
    main()
