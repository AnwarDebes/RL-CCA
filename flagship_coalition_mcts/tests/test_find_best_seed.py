"""Tests for the seed-picking utility."""

from __future__ import annotations

from flagship_coalition_mcts.src.find_best_seed import _extract_metric


def test_extract_top_level_scalar():
    data = {"drop_all": 0.42, "p0_winrate": 0.30}
    assert _extract_metric(data, "drop_all") == 0.42


def test_extract_nested_dotted():
    data = {"kingmaker": {"A3": {"elo_gap": 75.0}}}
    assert _extract_metric(data, "kingmaker.A3.elo_gap") == 75.0


def test_extract_mean_field():
    data = {"loss": {"mean": 0.5, "std": 0.1, "n": 3}}
    assert _extract_metric(data, "loss") == 0.5


def test_extract_missing_returns_nan():
    import math
    data = {"foo": 1.0}
    v = _extract_metric(data, "bar")
    assert math.isnan(v)


def test_extract_nested_missing_returns_nan():
    import math
    data = {"a": {"b": 1.0}}
    v = _extract_metric(data, "a.c.d")
    assert math.isnan(v)
