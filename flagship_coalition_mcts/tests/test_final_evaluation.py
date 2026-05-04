"""Tests for the final-evaluation script's helper functions.

Verifies the variant builders, since the script's main() requires
network checkpoints that don't exist during testing.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

_NEXUS_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)
if _NEXUS_ROOT not in sys.path:
    sys.path.insert(0, _NEXUS_ROOT)

try:
    from flagship_coalition_mcts.experiments.final_evaluation import (
        kingmaker_features, make_variant_player,
    )
    from flagship_coalition_mcts.src.games.kingmaker import (
        KingmakerGame, KingmakerState,
    )
    HAVE = True
except ImportError:
    HAVE = False

pytestmark = pytest.mark.skipif(
    not HAVE, reason="final_evaluation requires nexus core/* modules",
)


def test_kingmaker_features_correct_dim():
    s = KingmakerState.initial()
    feats = kingmaker_features(s)
    assert feats.shape == (12,)


def test_heuristic_variant_returns_valid_action_idx():
    agent = make_variant_player("heuristic", "", num_simulations=4)
    s = KingmakerState.initial()
    legal = KingmakerGame.legal_actions(s)
    a = agent(s)
    assert isinstance(a, int)
    assert 0 <= a < len(legal)


def test_scalar_variant_constructible_without_checkpoint():
    """Should fall back to untrained network gracefully."""
    agent = make_variant_player("scalar", "", num_simulations=2)
    s = KingmakerState.initial()
    a = agent(s)
    legal = KingmakerGame.legal_actions(s)
    assert 0 <= a < len(legal)


def test_cdmcts_variant_constructible_without_checkpoint():
    agent = make_variant_player("cdmcts", "", num_simulations=2)
    s = KingmakerState.initial()
    a = agent(s)
    legal = KingmakerGame.legal_actions(s)
    assert 0 <= a < len(legal)


def test_nncce_variant_constructible_without_checkpoint():
    agent = make_variant_player("nncce", "", num_simulations=2)
    s = KingmakerState.initial()
    a = agent(s)
    legal = KingmakerGame.legal_actions(s)
    assert 0 <= a < len(legal)


def test_unknown_variant_raises():
    with pytest.raises(ValueError):
        make_variant_player("nonsense_variant", "", num_simulations=2)
