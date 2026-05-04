"""Tests for the results-table aggregator.

Uses synthetic JSON to validate parsing, aggregation, and formatting.
"""

from __future__ import annotations

import json
import os
import tempfile

import pytest

from flagship_coalition_mcts.src.results_table import (
    RunRecord,
    aggregate,
    format_latex,
    format_markdown,
    load_run,
)


def _write(path, data):
    with open(path, "w") as f:
        json.dump(data, f)


def test_load_kingmaker_h2h_format():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump({
            "p0_winrate_scalar": 0.6,
            "p0_winrate_cd": 0.4,
            "p0_winrate_mixed": 0.3,
            "drop_all": 0.2,
            "drop_mixed": 0.3,
            "passed_pre_registered": True,
            "rank_counts_scalar": [[3, 2, 0], [1, 2, 2], [1, 1, 3]],
            "rank_counts_cd": [[1, 2, 2], [2, 2, 1], [2, 1, 2]],
            "rank_counts_mixed": [[2, 1, 2], [2, 2, 1], [1, 2, 2]],
            "args": {},
        }, f)
        path = f.name
    try:
        records = load_run(path)
        assert len(records) == 3
        variants = {r.variant for r in records}
        assert variants == {"scalar", "cd_mcts", "mixed_p0scalar_p1p2cd"}
        scalar_rec = next(r for r in records if r.variant == "scalar")
        assert scalar_rec.metrics["p0_winrate"] == 0.6
        cd_rec = next(r for r in records if r.variant == "cd_mcts")
        assert cd_rec.metrics["drop_vs_scalar"] == 0.2
    finally:
        os.unlink(path)


def test_load_ablation_format():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump({
            "kingmaker": {
                "A0": {"rank_counts": [[3, 2, 0], [1, 2, 2], [1, 1, 3]]},
                "A3": {"rank_counts": [[1, 2, 2], [2, 2, 1], [2, 1, 2]]},
            },
            "halma_small": {
                "A0": {"rank_counts": [[2, 2, 1], [1, 2, 2], [2, 1, 2]]},
            },
        }, f)
        path = f.name
    try:
        records = load_run(path)
        assert len(records) == 3
        games = {r.game for r in records}
        assert games == {"kingmaker", "halma_small"}
    finally:
        os.unlink(path)


def test_aggregate_with_multiple_seeds():
    records = [
        RunRecord("a", "A3", "kingmaker", {"elo": 100.0}),
        RunRecord("b", "A3", "kingmaker", {"elo": 110.0}),
        RunRecord("c", "A3", "kingmaker", {"elo": 90.0}),
        RunRecord("d", "A0", "kingmaker", {"elo": 0.0}),
    ]
    agg = aggregate(records)
    assert "kingmaker" in agg
    assert "A3" in agg["kingmaker"]
    assert agg["kingmaker"]["A3"]["elo"]["mean"] == 100.0
    assert agg["kingmaker"]["A3"]["elo"]["n"] == 3
    assert agg["kingmaker"]["A3"]["elo"]["std"] > 0
    assert agg["kingmaker"]["A0"]["elo"]["std"] == 0.0


def test_format_markdown_contains_variants():
    records = [
        RunRecord("a", "A0", "kingmaker", {"x": 1.0}),
        RunRecord("b", "A3", "kingmaker", {"x": 2.0}),
    ]
    out = format_markdown(aggregate(records))
    assert "A0" in out
    assert "A3" in out
    assert "kingmaker" in out


def test_format_latex_has_tabular():
    records = [RunRecord("a", "A0", "kingmaker", {"x": 1.0})]
    out = format_latex(aggregate(records))
    assert "\\begin{tabular}" in out
    assert "\\toprule" in out
    assert "\\end{tabular}" in out


def test_aggregate_handles_missing_metrics():
    records = [
        RunRecord("a", "A0", "kingmaker", {"x": 1.0}),
        RunRecord("b", "A0", "kingmaker", {"y": 2.0}),
    ]
    agg = aggregate(records)
    assert "x" in agg["kingmaker"]["A0"]
    assert "y" in agg["kingmaker"]["A0"]
    assert agg["kingmaker"]["A0"]["x"]["n"] == 1
    assert agg["kingmaker"]["A0"]["y"]["n"] == 1
