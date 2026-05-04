"""Tests for the CMAZ Halma adapter."""

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
from decomposed_mcts.src.halma_adapter import (
    HalmaCMAZEvaluator,
    build_cmaz_halma_network,
    halma_score_components,
)
from flagship_coalition_mcts.src.games.halma_small import (
    HalmaSmallGame, HalmaState, NUM_ACTIONS,
)


def test_halma_score_components_one_hot_terminal():
    s = HalmaState.initial()
    while not HalmaSmallGame.is_terminal(s):
        legal = HalmaSmallGame.legal_actions(s)
        if not legal:
            break
        s, _ = HalmaSmallGame.step(s, legal[0])
    for p in range(3):
        comp = halma_score_components(s, p)
        assert comp.shape == (3,)
        assert comp.sum() == 1.0


def test_halma_score_components_non_terminal_valid():
    s = HalmaState.initial()
    for p in range(3):
        comp = halma_score_components(s, p)
        assert comp.shape == (3,)
        assert comp.sum() == 1.0


def test_build_halma_network_shape():
    torch.manual_seed(0)
    net = build_cmaz_halma_network(hidden_dim=8, num_components=3)
    x = torch.randn(2, 107)
    pl, comps, q = net(x)
    assert pl.shape == (2, NUM_ACTIONS)
    assert comps.shape == (2, 3)
    assert q.shape == (2,)


def test_halma_evaluator_returns_valid_output():
    net = build_cmaz_halma_network(hidden_dim=8)
    ev = HalmaCMAZEvaluator(net)
    s = HalmaState.initial()
    out = ev.evaluate_cmaz(s)
    assert out.prior_policy.shape == (NUM_ACTIONS,)
    assert abs(out.prior_policy.sum() - 1.0) < 1e-5
    assert out.component_values.shape == (3,)


def test_end_to_end_cmaz_mcts_on_halma():
    torch.manual_seed(1)
    net = build_cmaz_halma_network(hidden_dim=8)
    ev = HalmaCMAZEvaluator(net)
    s = HalmaState.initial()
    legal = HalmaSmallGame.legal_actions(s)
    if not legal:
        pytest.skip("no legal actions at start (shouldn't happen)")
    root, pi = run_mcts_cmaz(
        state=s, network=ev, game=HalmaSmallGame(),
        mixer_apply=ev.mixer_apply,
        num_simulations=10,
    )
    assert pi.shape[0] == len(legal)
    assert abs(pi.sum() - 1.0) < 1e-9
