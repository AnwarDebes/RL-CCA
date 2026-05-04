"""Tests for the results-summary helper."""

from __future__ import annotations

import json
import os
import tempfile

import pytest

from flagship_coalition_mcts.src.summarize_results import _short_summary


def test_kingmaker_h2h_format():
    data = {
        "p0_winrate_scalar": 0.45,
        "p0_winrate_cd": 0.30,
        "p0_winrate_mixed": 0.20,
        "passed_pre_registered": True,
    }
    line = _short_summary(data, "/tmp/seed_0.json")
    assert "PASS" in line
    assert "0.45" in line
    assert "0.30" in line
    assert "0.20" in line


def test_kingmaker_h2h_format_failed():
    data = {
        "p0_winrate_scalar": 0.45,
        "p0_winrate_cd": 0.40,
        "p0_winrate_mixed": 0.42,
        "passed_pre_registered": False,
    }
    line = _short_summary(data, "/tmp/seed_0.json")
    assert "FAIL" in line


def test_ablation_format():
    data = {
        "kingmaker": {"A0": {"rank_counts": [[1, 2, 3]]}, "A3": {"rank_counts": [[3, 2, 1]]}},
        "halma_small": {"A0": {"rank_counts": [[2, 2, 2]]}},
    }
    line = _short_summary(data, "/tmp/abl.json")
    assert "abl" in line
    assert "kingmaker" in line or "halma_small" in line


def test_cce_gap_format():
    data = [
        {"iter_idx": 0, "cce_gap": 0.5},
        {"iter_idx": 5, "cce_gap": 0.3},
        {"iter_idx": 10, "cce_gap": 0.1},
    ]
    line = _short_summary(data, "/tmp/cce.json")
    assert "cce" in line
    assert "0.500" in line
    assert "0.100" in line
    assert "Δ=" in line


def test_generic_unknown_schema():
    data = {"unknown_key": "unknown_value"}
    line = _short_summary(data, "/tmp/foo.json")
    assert "?" in line
    assert "foo.json" in line
