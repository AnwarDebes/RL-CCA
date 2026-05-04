"""Tests for the CMAZ kingmaker adapter."""

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

from decomposed_mcts.src.cmaz_mcts import run_mcts_cmaz
from decomposed_mcts.src.kingmaker_adapter import (
    KingmakerCMAZEvaluator,
    build_cmaz_kingmaker_network,
    kingmaker_features_for_cmaz,
    kingmaker_score_components,
)
from flagship_coalition_mcts.src.games.kingmaker import (
    KingmakerGame, KingmakerState, NUM_ACTIONS,
)


def test_features_correct_dim():
    s = KingmakerState.initial()
    feats = kingmaker_features_for_cmaz(s)
    assert feats.shape == (12,)


def test_score_components_one_hot():
    """At a terminal state with known ranks, score components are one-hot."""
    s = KingmakerState.initial()
    while not KingmakerGame.is_terminal(s):
        legal = KingmakerGame.legal_actions(s)
        if not legal:
            break
        s, _ = KingmakerGame.step(s, legal[0])
    for p in range(3):
        comp = kingmaker_score_components(s, p)
        assert comp.shape == (3,)
        assert comp.sum() == 1.0
        assert (comp >= 0).all()


def test_build_network_shape():
    torch.manual_seed(0)
    net = build_cmaz_kingmaker_network(feature_dim=12, hidden_dim=8, num_components=3)
    x = torch.randn(2, 12)
    pl, comps, q = net(x)
    assert pl.shape == (2, NUM_ACTIONS)
    assert comps.shape == (2, 3)
    assert q.shape == (2,)


def test_evaluator_returns_valid_output():
    torch.manual_seed(1)
    net = build_cmaz_kingmaker_network(hidden_dim=8)
    ev = KingmakerCMAZEvaluator(net)
    s = KingmakerState.initial()
    out = ev.evaluate_cmaz(s)
    assert out.prior_policy.shape == (NUM_ACTIONS,)
    assert abs(out.prior_policy.sum() - 1.0) < 1e-5
    assert out.component_values.shape == (3,)
    assert out.encoder_features.shape[0] > 0


def test_end_to_end_cmaz_mcts_on_kingmaker():
    torch.manual_seed(2)
    net = build_cmaz_kingmaker_network(hidden_dim=8)
    ev = KingmakerCMAZEvaluator(net)
    root, pi = run_mcts_cmaz(
        state=KingmakerState.initial(),
        network=ev,
        game=KingmakerGame(),
        mixer_apply=ev.mixer_apply,
        num_simulations=20,
    )
    assert pi.shape[0] == len(KingmakerGame.legal_actions(KingmakerState.initial()))
    assert abs(pi.sum() - 1.0) < 1e-9


def test_inference_override_changes_q():
    """Different override weights should give different Q values for the
    same per-component value vector. Note: mixer_apply expects ENCODED
    features (the hidden_dim representation produced by the encoder),
    not raw features - the evaluator encodes first."""
    torch.manual_seed(3)
    net = build_cmaz_kingmaker_network(hidden_dim=8)
    ev_a = KingmakerCMAZEvaluator(net, override_weights=np.array([1.0, 0.0, 0.0]))
    ev_b = KingmakerCMAZEvaluator(net, override_weights=np.array([0.0, 0.0, 1.0]))
    v = np.array([0.8, 0.4, 0.1])
    s = KingmakerState.initial()
    # Use the evaluator's evaluate_cmaz to get encoded features.
    out = ev_a.evaluate_cmaz(s)
    encoded_feats = out.encoder_features
    qa = ev_a.mixer_apply(v, encoded_feats)
    qb = ev_b.mixer_apply(v, encoded_feats)
    assert qa != qb, f"override should change Q: qa={qa}, qb={qb}"
