"""Tests for the scalar-PUCT baseline (ablation A0)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from flagship_coalition_mcts.src.baseline_mcts import (
    ScalarEvaluator,
    ScalarNetworkOutput,
    run_mcts_scalar,
)
from flagship_coalition_mcts.src.games.kingmaker import (
    KingmakerGame,
    KingmakerState,
    NUM_ACTIONS,
)
from flagship_coalition_mcts.src.network import CDMCTSNetwork, MLPEncoder
from flagship_coalition_mcts.tests.test_network import kingmaker_to_features


def test_scalar_mcts_smoke():
    torch.manual_seed(0)
    net = CDMCTSNetwork(
        encoder=MLPEncoder(input_dim=12, hidden_dim=24, num_layers=2),
        action_space_size=NUM_ACTIONS, max_players=3,
    )
    ev = ScalarEvaluator(
        network=net,
        state_to_features=kingmaker_to_features,
        current_player_fn=KingmakerGame.current_player,
        num_players_fn=KingmakerGame.num_players,
    )
    root, pi = run_mcts_scalar(
        state=KingmakerState.initial(),
        network=ev,
        game=KingmakerGame(),
        num_simulations=20,
    )
    assert pi.shape == (len(KingmakerGame.legal_actions(KingmakerState.initial())),)
    assert abs(pi.sum() - 1.0) < 1e-9
    assert (pi >= 0).all()


def test_scalar_visits_sum_to_simulations():
    torch.manual_seed(1)
    net = CDMCTSNetwork(
        encoder=MLPEncoder(input_dim=12, hidden_dim=24, num_layers=2),
        action_space_size=NUM_ACTIONS, max_players=3,
    )
    ev = ScalarEvaluator(
        network=net,
        state_to_features=kingmaker_to_features,
        current_player_fn=KingmakerGame.current_player,
        num_players_fn=KingmakerGame.num_players,
    )
    root, pi = run_mcts_scalar(
        state=KingmakerState.initial(),
        network=ev,
        game=KingmakerGame(),
        num_simulations=50,
    )
    assert root.child_visits.sum() == 50


def test_scalar_evaluator_returns_correct_shapes():
    net = CDMCTSNetwork(
        encoder=MLPEncoder(input_dim=12, hidden_dim=16, num_layers=1),
        action_space_size=NUM_ACTIONS, max_players=3,
    )
    ev = ScalarEvaluator(
        network=net,
        state_to_features=kingmaker_to_features,
        current_player_fn=KingmakerGame.current_player,
        num_players_fn=KingmakerGame.num_players,
    )
    out = ev.evaluate_scalar(KingmakerState.initial())
    assert isinstance(out, ScalarNetworkOutput)
    assert out.prior_policy.shape == (NUM_ACTIONS,)
    assert out.per_player_value.shape == (3,)
    assert abs(out.prior_policy.sum() - 1.0) < 1e-5
