"""Tests for the CD-MCTS-on-Chinese-Checkers runner.

These tests verify:
1. build_cc_evaluator returns a CDMCTSNetwork + CCMCTSEvaluator pair.
2. Evaluator returns a valid NetworkOutput when given a real GameEnv state.
3. play_one_cc_game runs at least a few MCTS steps without crashing.
4. Targets are correctly populated when a game terminates.

These tests are heavier than the kingmaker tests because they instantiate
a real GameEnv. They are queued for after training completes.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest
import torch

_NEXUS_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)
if _NEXUS_ROOT not in sys.path:
    sys.path.insert(0, _NEXUS_ROOT)

try:
    from flagship_coalition_mcts.src.cc_runner import (
        CCMCTSEvaluator,
        build_cc_evaluator,
        build_cc_network,
        play_one_cc_game,
    )
    from flagship_coalition_mcts.src.games.chinese_checkers import make_cc_env
    from flagship_coalition_mcts.src.mcts import NetworkOutput
    HAVE_CC = True
except ImportError:
    HAVE_CC = False

pytestmark = pytest.mark.skipif(
    not HAVE_CC, reason="CC env requires nexus core/* modules",
)


def test_build_cc_network_shapes():
    net = build_cc_network(
        num_players_max=6, channels=8, num_blocks=2, hidden_dim=16,
    )
    x = torch.randn(2, 32, 17, 17)
    pl_logits, theta, A, beta, sv = net(x)
    assert pl_logits.shape == (2, 1210)
    assert theta.shape == (2, 6)
    assert A.shape == (2, 6, 6)


def test_evaluator_produces_valid_output():
    torch.manual_seed(0)
    net, ev = build_cc_evaluator(
        num_players_max=6, channels=8, num_blocks=2, hidden_dim=16,
    )
    state = make_cc_env(num_players=2, seed=0)
    out = ev.evaluate(state)
    assert isinstance(out, NetworkOutput)
    assert out.prior_policy.shape == (1210,)
    assert abs(out.prior_policy.sum() - 1.0) < 1e-5
    # Evaluator returns (N, N) matching active player count - MCTS
    # expects (num_players, num_players) shape for its vector backup,
    # not max_players-padded matrices. Verified by test_cc_runner smoke.
    assert out.placement_marginals.shape == (2, 2)
    assert np.allclose(out.placement_marginals.sum(axis=1), 1.0, atol=1e-5)
    assert out.coalition_alignment.shape == (2,)


def test_play_one_cc_game_short_smoke():
    """Run play_one_cc_game with very small budgets - verify no crash.

    Expensive in absolute terms (CC games are ~50-200 moves each), but
    with num_simulations=2 it's tractable in <30s.
    """
    torch.manual_seed(1)
    net = build_cc_network(channels=8, num_blocks=2, hidden_dim=16)
    result = play_one_cc_game(
        network=net,
        num_players=2,
        num_simulations=2,
        coalition_weight=0.5,
        seed=1,
        max_moves=20,  # truncate aggressively for the test
    )
    assert "trajectory" in result
    assert isinstance(result["num_moves"], int)
    assert result["num_moves"] <= 20
    if result["terminated"]:
        for entry in result["trajectory"]:
            # Targets must have been filled
            assert "observed_ranking" in entry
            assert "observed_coalition_index" in entry
            assert "target_scalar_value" in entry
            assert 0.0 <= entry["target_scalar_value"] <= 1.0
