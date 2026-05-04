"""Tests for training visualisation utilities.

Tests the data-handling logic; doesn't require a GUI backend.
"""

from __future__ import annotations

import json
import os
import tempfile

import pytest

from flagship_coalition_mcts.src.training_visualize import (
    _load_history, text_summary,
)


def test_load_history_list_format():
    data = [{"iter": 0, "avg_total": 1.0}, {"iter": 1, "avg_total": 0.9}]
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(data, f)
        path = f.name
    try:
        h = _load_history(path)
        assert h == data
    finally:
        os.unlink(path)


def test_load_history_nested_format():
    data = {"history": [{"iter": 0, "avg_total": 1.0}], "args": {}}
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(data, f)
        path = f.name
    try:
        h = _load_history(path)
        assert h == data["history"]
    finally:
        os.unlink(path)


def test_load_history_unknown_format_raises():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump("not a list or dict-with-history", f)
        path = f.name
    try:
        with pytest.raises(ValueError):
            _load_history(path)
    finally:
        os.unlink(path)


def test_text_summary_includes_iteration_numbers():
    h = [
        {"iter": 0, "avg_total": 1.5, "avg_policy": 0.8, "gen_sec": 100, "train_sec": 200},
        {"iter": 1, "avg_total": 1.2, "avg_policy": 0.7, "gen_sec": 95, "train_sec": 210},
    ]
    s = text_summary(h)
    assert "0" in s and "1" in s
    assert "1.500" in s and "1.200" in s


def test_plot_history_emits_file_with_matplotlib():
    """If matplotlib is available, a plot file should be written."""
    try:
        import matplotlib  # noqa
    except ImportError:
        pytest.skip("matplotlib not installed")

    from flagship_coalition_mcts.src.training_visualize import plot_history

    history = [
        {"iter": i, "avg_total": 2.0 - 0.05 * i,
         "avg_policy": 1.0 - 0.025 * i, "gen_sec": 100 + i, "train_sec": 200 + i}
        for i in range(20)
    ]
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        out = f.name
    try:
        plot_history(history, output=out, title="test")
        assert os.path.exists(out)
        assert os.path.getsize(out) > 0
    finally:
        if os.path.exists(out):
            os.unlink(out)
