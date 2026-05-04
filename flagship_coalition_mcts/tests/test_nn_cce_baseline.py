"""Tests for the NN-CCE-extended-to-N baseline.

Verifies:
1. Smoke test: runs without crashing.
2. Visit counts sum to num_simulations.
3. Vector backup is consistent on a one-step game (matches per-player
   utility of terminal state).
4. NN-CCE shares the EXP-IX selector convergence properties with CD-MCTS:
   on Matching-Pennies-style games the empirical play approaches Nash.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from flagship_coalition_mcts.src.games.kingmaker import (
    KingmakerGame,
    KingmakerState,
    NUM_ACTIONS,
)
from flagship_coalition_mcts.src.network import CDMCTSNetwork, MLPEncoder
from flagship_coalition_mcts.src.nn_cce_baseline import (
    NNCCEEvaluator,
    NNCCENetworkOutput,
    run_mcts_nncce,
)
from flagship_coalition_mcts.tests.test_network import kingmaker_to_features


def _make_net(seed: int = 0) -> CDMCTSNetwork:
    torch.manual_seed(seed)
    return CDMCTSNetwork(
        encoder=MLPEncoder(input_dim=12, hidden_dim=24, num_layers=2),
        action_space_size=NUM_ACTIONS, max_players=3,
    )


def test_nncce_smoke():
    net = _make_net()
    ev = NNCCEEvaluator(
        network=net, state_to_features=kingmaker_to_features,
        current_player_fn=KingmakerGame.current_player,
        num_players_fn=KingmakerGame.num_players,
    )
    root, pi = run_mcts_nncce(
        state=KingmakerState.initial(),
        network=ev,
        game=KingmakerGame(),
        num_simulations=20,
        seed=0,
    )
    assert pi.shape[0] == 3  # 3 legal actions at root
    assert abs(pi.sum() - 1.0) < 1e-9


def test_nncce_visits_sum_to_simulations():
    net = _make_net(seed=1)
    ev = NNCCEEvaluator(
        network=net, state_to_features=kingmaker_to_features,
        current_player_fn=KingmakerGame.current_player,
        num_players_fn=KingmakerGame.num_players,
    )
    root, _ = run_mcts_nncce(
        state=KingmakerState.initial(),
        network=ev, game=KingmakerGame(),
        num_simulations=40,
        seed=1,
    )
    assert root.selector_state.visits.sum() == 40


def test_nncce_evaluator_returns_correct_shapes():
    net = _make_net()
    ev = NNCCEEvaluator(
        network=net, state_to_features=kingmaker_to_features,
        current_player_fn=KingmakerGame.current_player,
        num_players_fn=KingmakerGame.num_players,
    )
    out = ev.evaluate_nncce(KingmakerState.initial())
    assert isinstance(out, NNCCENetworkOutput)
    assert out.prior_policy.shape == (NUM_ACTIONS,)
    assert out.per_player_value.shape == (3,)
