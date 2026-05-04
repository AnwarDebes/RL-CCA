"""Publication-quality results aggregator.

Reads the JSON output of ablation experiments and head-to-head runs,
produces:

  * A clean summary table (text + markdown).
  * A LaTeX table ready to paste into the paper.
  * A simple matplotlib plot (Elo ± CI per variant per game).

Usage
-----
    python -m flagship_coalition_mcts.src.results_table \\
        --files exp1.json exp2.json --output paper_table.tex

Or programmatically:

    from results_table import load_run, build_summary_table
    rows = build_summary_table([load_run(p) for p in paths])
    print(format_markdown(rows))
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


# ----------------------------------------------------------------------
# Data structures
# ----------------------------------------------------------------------


@dataclass
class RunRecord:
    """One experimental run's normalised representation."""
    name: str          # human-readable label (e.g. "kingmaker_h2h_seed0")
    variant: str       # algorithm variant (e.g. "A0", "A3", "CD-MCTS")
    game: str          # game name (e.g. "kingmaker", "halma_small", "cc_2p")
    metrics: Dict[str, float]  # arbitrary numeric metrics
    raw_json: Dict[str, Any] = None


def load_run(path: str) -> List[RunRecord]:
    """Load a JSON file produced by an experiment and convert to records.

    Supports two output formats:
      1. kingmaker_head_to_head.py: single-run dict with p0_winrate_*,
         drop_*, rank_counts_*.
      2. ablation_ladder.py: nested dict {game: {variant: {...}}}.
    """
    with open(path, "r") as f:
        data = json.load(f)
    records = []
    if "p0_winrate_scalar" in data:
        # kingmaker_head_to_head format
        records.append(RunRecord(
            name=path,
            variant="scalar",
            game="kingmaker",
            metrics=dict(
                p0_winrate=data["p0_winrate_scalar"],
                drop_vs_self=0.0,
            ),
            raw_json=data,
        ))
        records.append(RunRecord(
            name=path,
            variant="cd_mcts",
            game="kingmaker",
            metrics=dict(
                p0_winrate=data["p0_winrate_cd"],
                drop_vs_scalar=data["drop_all"],
            ),
            raw_json=data,
        ))
        records.append(RunRecord(
            name=path,
            variant="mixed_p0scalar_p1p2cd",
            game="kingmaker",
            metrics=dict(
                p0_winrate=data["p0_winrate_mixed"],
                drop_vs_all_scalar=data["drop_mixed"],
                pre_registered_passed=int(data.get("passed_pre_registered", False)),
            ),
            raw_json=data,
        ))
    elif isinstance(data, dict):
        # ablation_ladder format: {game: {variant: {rank_counts: ...}}}
        for game_name, variants in data.items():
            if not isinstance(variants, dict):
                continue
            for variant_name, vd in variants.items():
                if not isinstance(vd, dict):
                    continue
                metrics = {}
                if "rank_counts" in vd:
                    rc = vd["rank_counts"]
                    # rc[seat][rank-1] = count.
                    n = sum(sum(row) for row in rc) // len(rc) if rc else 0
                    if n > 0:
                        for seat in range(len(rc)):
                            for rank in range(len(rc[seat])):
                                metrics[f"seat{seat}_rank{rank+1}"] = rc[seat][rank] / max(1, n)
                records.append(RunRecord(
                    name=f"{path}::{game_name}::{variant_name}",
                    variant=variant_name,
                    game=game_name,
                    metrics=metrics,
                    raw_json=vd,
                ))
    return records


# ----------------------------------------------------------------------
# Summarisation
# ----------------------------------------------------------------------


def aggregate(records: List[RunRecord]) -> Dict[str, Dict[str, Dict[str, float]]]:
    """Group records by (game, variant) and aggregate via mean / std.

    Returns: {game: {variant: {metric: {mean, std, n}}}}.
    """
    bucket: Dict = {}
    for r in records:
        bucket.setdefault(r.game, {}).setdefault(r.variant, []).append(r)
    out = {}
    for game, vmap in bucket.items():
        out[game] = {}
        for variant, rs in vmap.items():
            metric_names = set()
            for r in rs:
                metric_names.update(r.metrics.keys())
            agg = {}
            for m in sorted(metric_names):
                vals = [r.metrics[m] for r in rs if m in r.metrics]
                if not vals:
                    continue
                mean = sum(vals) / len(vals)
                std = math.sqrt(sum((v - mean) ** 2 for v in vals) / max(1, len(vals)))
                agg[m] = dict(mean=mean, std=std, n=len(vals))
            out[game][variant] = agg
    return out


# ----------------------------------------------------------------------
# Output formats
# ----------------------------------------------------------------------


def format_markdown(agg: Dict, decimals: int = 3) -> str:
    """Markdown table with one row per (game, variant), columns per metric."""
    lines = []
    for game, vmap in agg.items():
        all_metrics = sorted({m for v in vmap.values() for m in v})
        header = ["variant"] + all_metrics
        lines.append(f"\n### {game}")
        lines.append("| " + " | ".join(header) + " |")
        lines.append("|" + "|".join(["---"] * len(header)) + "|")
        for variant, metrics in vmap.items():
            row = [variant]
            for m in all_metrics:
                if m not in metrics:
                    row.append("-")
                else:
                    d = metrics[m]
                    if d["n"] == 1:
                        row.append(f"{d['mean']:.{decimals}f}")
                    else:
                        row.append(f"{d['mean']:.{decimals}f} ± {d['std']:.{decimals}f}")
            lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def format_latex(agg: Dict, decimals: int = 3) -> str:
    """LaTeX `tabular` ready for a paper."""
    lines = []
    for game, vmap in agg.items():
        all_metrics = sorted({m for v in vmap.values() for m in v})
        cols = "l" + "c" * len(all_metrics)
        lines.append(f"\\begin{{table}}[h]")
        lines.append(f"\\caption{{Results on \\textsc{{{game}}}.}}")
        lines.append(f"\\begin{{tabular}}{{{cols}}}")
        lines.append("\\toprule")
        lines.append(
            " variant & " + " & ".join(m.replace("_", "\\_") for m in all_metrics) + " \\\\"
        )
        lines.append("\\midrule")
        for variant, metrics in vmap.items():
            cells = [variant.replace("_", "\\_")]
            for m in all_metrics:
                if m not in metrics:
                    cells.append("--")
                else:
                    d = metrics[m]
                    if d["n"] == 1:
                        cells.append(f"{d['mean']:.{decimals}f}")
                    else:
                        cells.append(f"${d['mean']:.{decimals}f} \\pm {d['std']:.{decimals}f}$")
            lines.append(" & ".join(cells) + " \\\\")
        lines.append("\\bottomrule")
        lines.append("\\end{tabular}")
        lines.append("\\end{table}\n")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", nargs="+", required=True)
    ap.add_argument("--format", choices=["markdown", "latex"], default="markdown")
    ap.add_argument("--decimals", type=int, default=3)
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    records = []
    for path in args.files:
        records.extend(load_run(path))
    print(f"Loaded {len(records)} records from {len(args.files)} files.")
    agg = aggregate(records)
    if args.format == "markdown":
        out = format_markdown(agg, decimals=args.decimals)
    else:
        out = format_latex(agg, decimals=args.decimals)
    if args.output:
        with open(args.output, "w") as f:
            f.write(out + "\n")
        print(f"Wrote {args.output}")
    else:
        print(out)


if __name__ == "__main__":
    main()
