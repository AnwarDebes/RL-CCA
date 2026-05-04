"""Tests for the public API surface (flagship subproject's __init__.py).

If any of these imports breaks, downstream users will see broken
imports. The test suite catches that early.
"""

from __future__ import annotations

import pytest


def test_main_classes_importable():
    """Top-level classes are importable from the package root."""
    from flagship_coalition_mcts.src import (
        CDMCTSNetwork,
        MLPEncoder,
        NetworkOutput,
        Node,
        PlackettLuceHead,
        CoalitionHead,
        SelectorState,
        ReplayBuffer,
        ScalarEvaluator,
        NNCCEEvaluator,
        HeadToHeadResult,
    )
    # Each is a class
    assert isinstance(CDMCTSNetwork, type)
    assert isinstance(PlackettLuceHead, type)
    assert isinstance(CoalitionHead, type)


def test_main_functions_importable():
    """Top-level functions are importable from the package root."""
    from flagship_coalition_mcts.src import (
        run_mcts,
        run_mcts_scalar,
        run_mcts_nncce,
        head_to_head,
        save_checkpoint,
        load_checkpoint,
        sample_ranking,
        winner_marginal,
        cce_gap,
    )
    assert callable(run_mcts)
    assert callable(head_to_head)
    assert callable(save_checkpoint)


def test_cc_runner_imports():
    """CC integration imports."""
    try:
        from flagship_coalition_mcts.src import (
            build_cc_evaluator, build_cc_network, play_one_cc_game,
        )
        assert callable(build_cc_evaluator)
        assert callable(play_one_cc_game)
    except ImportError:
        pytest.skip("CC runner needs nexus core/* modules")


def test_all_export_consistent():
    """The __all__ list should reference only actually-importable names."""
    from flagship_coalition_mcts import src as pkg
    for name in pkg.__all__:
        # Some attrs are deferred to None when nexus core/* is missing.
        attr = getattr(pkg, name, "MISSING")
        assert attr != "MISSING", f"__all__ lists {name!r} but it's not importable"


def test_version_metadata_present():
    """Package exposes __version__, __title__, __author__."""
    from flagship_coalition_mcts import src as pkg
    assert pkg.__version__
    assert pkg.__title__
    assert pkg.__author__
