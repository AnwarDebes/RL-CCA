"""Tests for the MCTS tree-debug utility."""

from __future__ import annotations

import io
import sys

import numpy as np
import pytest

from flagship_coalition_mcts.src.games.kingmaker import (
    KingmakerGame, KingmakerState, NUM_ACTIONS,
)
from flagship_coalition_mcts.src.mcts import NetworkOutput, run_mcts
from flagship_coalition_mcts.src.tree_debug import (
    format_stats, print_tree, tree_stats,
)


class StubNet:
    def evaluate(self, state):
        return NetworkOutput(
            prior_policy=np.full(NUM_ACTIONS, 1.0 / NUM_ACTIONS),
            placement_marginals=np.full((3, 3), 1.0 / 3),
            coalition_alignment=np.zeros(3),
        )


def _build_root(num_simulations=20):
    return run_mcts(
        KingmakerState.initial(), StubNet(), KingmakerGame(),
        num_simulations=num_simulations, seed=0,
    )[0]


def test_tree_stats_returns_dict():
    root = _build_root(num_simulations=20)
    stats = tree_stats(root)
    assert isinstance(stats, dict)
    assert "total_nodes" in stats
    assert "max_depth" in stats
    assert stats["total_nodes"] >= 1


def test_tree_stats_root_visits_match_num_simulations():
    root = _build_root(num_simulations=37)
    stats = tree_stats(root)
    assert stats["root_visits"] == 37


def test_tree_stats_top_action_share_in_unit_interval():
    root = _build_root(num_simulations=30)
    stats = tree_stats(root)
    s = stats["top_action_visit_share"]
    assert 0.0 <= s <= 1.0


def test_tree_stats_max_depth_at_least_1():
    root = _build_root(num_simulations=20)
    stats = tree_stats(root)
    # Depth-0 = root only; with simulations > 0, depth >= 1.
    assert stats["max_depth"] >= 1


def test_print_tree_writes_to_stdout():
    root = _build_root(num_simulations=20)
    capture = io.StringIO()
    old_stdout = sys.stdout
    sys.stdout = capture
    try:
        print_tree(root, max_depth=2, max_children=3)
    finally:
        sys.stdout = old_stdout
    out = capture.getvalue()
    assert "ROOT" in out
    assert "visits=" in out


def test_format_stats_returns_multiline():
    root = _build_root(num_simulations=20)
    stats = tree_stats(root)
    s = format_stats(stats)
    assert "\n" in s
    assert "MCTS tree statistics" in s
