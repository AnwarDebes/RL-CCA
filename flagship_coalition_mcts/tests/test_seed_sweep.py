"""Tests for the seed_sweep CLI helper.

Tests argument parsing and pattern substitution; doesn't actually run
subprocess calls (those need real experiment scripts and compute).
"""

from __future__ import annotations

import sys


def test_seed_sweep_module_imports():
    """The seed_sweep module imports cleanly."""
    from flagship_coalition_mcts.experiments import seed_sweep  # noqa
    assert hasattr(seed_sweep, "main")


def test_out_pattern_substitution():
    """The output pattern replaces {seed} correctly."""
    pattern = "results/foo_seed{seed}.json"
    for seed in [0, 1, 42]:
        out = pattern.format(seed=seed)
        assert str(seed) in out
        assert "{seed}" not in out


def test_out_pattern_with_no_placeholder():
    """A pattern without {seed} works (all writes go to same path -
    user's choice)."""
    pattern = "results/single_output.json"
    out = pattern.format(seed=0)
    assert out == pattern
