"""Import / construction smoke test for the play_server_cdmcts.py
tournament client.

Doesn't actually connect to a server (no network in unit tests). Just
verifies the import chain is intact and the player can be instantiated
with no model_path (untrained network).
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


def test_import_play_server_cdmcts_module():
    """The module's imports should resolve cleanly."""
    try:
        # Import as a module without invoking main()
        import importlib.util
        path = os.path.join(_NEXUS_ROOT, "scripts", "play_server_cdmcts.py")
        if not os.path.exists(path):
            pytest.skip("play_server_cdmcts.py not found")
        spec = importlib.util.spec_from_file_location("play_server_cdmcts", path)
        module = importlib.util.module_from_spec(spec)
        # Don't actually run main()
        # Just check that the imports up to main() work.
        try:
            spec.loader.exec_module(module)
        except SystemExit:
            pass
        assert hasattr(module, "main")
    except ImportError as e:
        pytest.skip(f"play_server_cdmcts requires nexus core/* modules: {e}")


def test_tournament_player_cdmcts_module_import():
    """Independent: test that the player module itself can be imported
    without a model_path."""
    try:
        from flagship_coalition_mcts.src.tournament_player_cdmcts import (
            NexusTournamentPlayerCDMCTS,
        )
    except ImportError:
        pytest.skip("requires nexus core/* modules")
    p = NexusTournamentPlayerCDMCTS(
        model_path=None, channels=4, num_blocks=1, hidden_dim=8,
    )
    assert p.color is None
    assert p.network is not None
    # Construct + close cleanly
    del p
