"""Tests for the CMAZ Chinese Checkers adapter."""

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
    from decomposed_mcts.src.cc_adapter import (
        CMAZCCEvaluator, build_cmaz_cc_network, play_one_cmaz_cc_game,
    )
    from flagship_coalition_mcts.src.games.chinese_checkers import (
        cc_score_components, make_cc_env,
    )
    HAVE = True
except ImportError:
    HAVE = False

pytestmark = pytest.mark.skipif(
    not HAVE, reason="Requires nexus core/* modules",
)


def test_build_cmaz_cc_network_shape():
    net = build_cmaz_cc_network(channels=4, num_blocks=1, hidden_dim=8, num_components=4)
    n_params = sum(p.numel() for p in net.parameters())
    assert n_params > 0
    # 4 score components for CC (pin_goal, distance, time, move)
    assert net.num_components == 4


def test_evaluator_returns_valid_output():
    torch.manual_seed(0)
    net = build_cmaz_cc_network(channels=4, num_blocks=1, hidden_dim=8)
    ev = CMAZCCEvaluator(net)
    state = make_cc_env(num_players=2, seed=0)
    out = ev.evaluate_cmaz(state)
    assert out.prior_policy.shape == (1210,)
    assert abs(out.prior_policy.sum() - 1.0) < 1e-5
    assert out.component_values.shape == (4,)
    # Encoded features have hidden_dim length.
    assert out.encoder_features.shape == (8,)


def test_terminal_components_returns_4_score_decomposition():
    torch.manual_seed(1)
    net = build_cmaz_cc_network(channels=4, num_blocks=1, hidden_dim=8)
    ev = CMAZCCEvaluator(net)
    state = make_cc_env(num_players=2, seed=1)
    comps = ev.terminal_components(state)
    assert comps.shape == (4,)
    assert (comps >= 0).all()
    assert (comps <= 1.0 + 1e-6).all()


def test_inference_override_changes_q():
    """Different mixer overrides yield different scalar Q values."""
    torch.manual_seed(2)
    net = build_cmaz_cc_network(channels=4, num_blocks=1, hidden_dim=8)
    state = make_cc_env(num_players=2, seed=2)
    ev_a = CMAZCCEvaluator(net, override_weights=np.array([1.0, 0.0, 0.0, 0.0]))
    ev_b = CMAZCCEvaluator(net, override_weights=np.array([0.0, 0.0, 0.0, 1.0]))
    # Need encoded features to call mixer_apply; get them via evaluate_cmaz
    out = ev_a.evaluate_cmaz(state)
    v = np.array([0.8, 0.4, 0.1, 0.3])
    qa = ev_a.mixer_apply(v, out.encoder_features)
    qb = ev_b.mixer_apply(v, out.encoder_features)
    assert qa != qb, f"override should change Q: qa={qa}, qb={qb}"


def test_play_one_cmaz_cc_game_smoke():
    """Truncated CMAZ game on real CC, small budget."""
    torch.manual_seed(3)
    net = build_cmaz_cc_network(channels=4, num_blocks=1, hidden_dim=8)
    result = play_one_cmaz_cc_game(
        network=net,
        num_players=2,
        num_simulations=2,
        seed=3,
        max_moves=10,
    )
    assert "trajectory" in result
    if result["terminated"]:
        for entry in result["trajectory"]:
            assert "target_components" in entry
            assert entry["target_components"].shape == (4,)
