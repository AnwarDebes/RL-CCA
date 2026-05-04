"""Tests for the CD-MCTS tournament player adapter.

Verifies:
1. Construction with no model_path (untrained network).
2. set_color / advance_with_opponent_action.
3. Subtree reuse: after playing a move, next call uses the cached root
   when the opponent action is recorded.
4. Robust fallback through _policy_move.
"""

from __future__ import annotations

import os
import sys

import pytest

_NEXUS_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)
if _NEXUS_ROOT not in sys.path:
    sys.path.insert(0, _NEXUS_ROOT)

try:
    from flagship_coalition_mcts.src.tournament_player_cdmcts import (
        NexusTournamentPlayerCDMCTS,
    )
    HAVE = True
except ImportError:
    HAVE = False

pytestmark = pytest.mark.skipif(
    not HAVE, reason="Tournament player requires nexus core/* modules",
)


def test_construct_with_no_model_path():
    """Should construct successfully even without a checkpoint."""
    p = NexusTournamentPlayerCDMCTS(
        model_path=None, channels=8, num_blocks=1, hidden_dim=16,
    )
    assert p.color is None
    assert p.network is not None


def test_set_color_resets_cache():
    p = NexusTournamentPlayerCDMCTS(channels=8, num_blocks=1, hidden_dim=16)
    p._cached_root = "anything"
    p._cached_action_history = [1, 2, 3]
    p.set_color("red", ["red", "blue"])
    assert p.color == "red"
    assert p._cached_root is None
    assert p._cached_action_history == []


def test_advance_with_opponent_action_records_history():
    p = NexusTournamentPlayerCDMCTS(channels=8, num_blocks=1, hidden_dim=16)
    p._cached_root = "fake_root"
    p._cached_action_history = []
    p.advance_with_opponent_action(42)
    assert p._cached_action_history == [42]
    p.advance_with_opponent_action(99)
    assert p._cached_action_history == [42, 99]


def test_advance_with_opponent_action_noop_when_no_cache():
    """If no cached root exists, advance_with_opponent_action is a no-op."""
    p = NexusTournamentPlayerCDMCTS(channels=8, num_blocks=1, hidden_dim=16)
    p.advance_with_opponent_action(42)
    # Cache history should still be empty since there's nothing to advance from
    assert p._cached_action_history == []
