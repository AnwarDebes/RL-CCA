"""Tests for the CMAZ public API surface."""

from __future__ import annotations

import pytest


def test_main_classes_importable():
    from decomposed_mcts.src import (
        MonotonicMixer,
        ComponentValueHead,
        CMAZNetwork,
        CMAZEncoder,
        CMAZEvaluator,
        CMAZNode,
        CMAZNetworkOutput,
    )
    assert isinstance(MonotonicMixer, type)
    assert isinstance(CMAZNetwork, type)


def test_main_functions_importable():
    from decomposed_mcts.src import (
        cmaz_loss, run_mcts_cmaz, run_simulation_cmaz,
    )
    assert callable(cmaz_loss)
    assert callable(run_mcts_cmaz)


def test_adapter_imports():
    try:
        from decomposed_mcts.src import (
            CMAZCCEvaluator, build_cmaz_cc_network,
            KingmakerCMAZEvaluator, build_cmaz_kingmaker_network,
            HalmaCMAZEvaluator, build_cmaz_halma_network,
        )
    except ImportError:
        pytest.skip("Adapters need nexus core/* modules")


def test_all_export_consistent():
    from decomposed_mcts import src as pkg
    for name in pkg.__all__:
        attr = getattr(pkg, name, "MISSING")
        assert attr != "MISSING", f"__all__ lists {name!r} but it's not importable"


def test_version_metadata_present():
    from decomposed_mcts import src as pkg
    assert pkg.__version__
    assert pkg.__title__
    assert pkg.__author__
