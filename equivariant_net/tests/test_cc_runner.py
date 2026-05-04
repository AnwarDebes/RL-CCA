"""Tests for the wreath equivariant CC runner."""

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
    from equivariant_net.src.cc_runner import (
        WreathCCEvaluator, play_one_wreath_cc_game,
    )
    from equivariant_net.src.wreath_network import WreathCCNetwork
    from flagship_coalition_mcts.src.games.chinese_checkers import make_cc_env
    HAVE = True
except ImportError:
    HAVE = False

pytestmark = pytest.mark.skipif(
    not HAVE, reason="Requires nexus core/* modules",
)


def test_wreath_cc_evaluator_returns_valid_output():
    torch.manual_seed(0)
    net = WreathCCNetwork(
        spatial_channels=4, spatial_blocks=1, spatial_out=8,
        seat_hidden=4, seat_blocks=1,
    )
    ev = WreathCCEvaluator(net)
    state = make_cc_env(num_players=2, seed=0)
    out = ev.evaluate_scalar(state)
    assert out.prior_policy.shape == (1210,)
    assert abs(out.prior_policy.sum() - 1.0) < 1e-5
    assert out.per_player_value.shape == (2,)


def test_play_one_wreath_cc_game_short_smoke():
    """Run one truncated game to verify the pipeline works end-to-end."""
    torch.manual_seed(1)
    net = WreathCCNetwork(
        spatial_channels=4, spatial_blocks=1, spatial_out=8,
        seat_hidden=4, seat_blocks=1,
    )
    result = play_one_wreath_cc_game(
        network=net,
        num_players=2,
        num_simulations=2,
        seed=1,
        max_moves=10,
    )
    assert "trajectory" in result
    assert isinstance(result["num_moves"], int)
    assert result["num_moves"] <= 10
    if result["terminated"]:
        for entry in result["trajectory"]:
            assert "target_scalar_value" in entry
            assert 0.0 <= entry["target_scalar_value"] <= 1.0
            assert entry["features_2d"].shape == (32, 17, 17)
            assert entry["seat_features"].shape == (6, 8)
