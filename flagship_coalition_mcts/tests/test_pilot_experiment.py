"""Smoke test for the pilot experiment.

Doesn't actually run the experiment (would take 1-2 minutes), but
verifies the imports + helper functions work.
"""

from __future__ import annotations


def test_pilot_imports_cleanly():
    from flagship_coalition_mcts.experiments import pilot_experiment
    assert hasattr(pilot_experiment, "main")
    assert hasattr(pilot_experiment, "kingmaker_features")


def test_kingmaker_features_correct_dim():
    from flagship_coalition_mcts.experiments.pilot_experiment import kingmaker_features
    from flagship_coalition_mcts.src.games.kingmaker import KingmakerState
    s = KingmakerState.initial()
    feats = kingmaker_features(s)
    assert feats.shape == (12,)
