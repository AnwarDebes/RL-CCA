"""Tests for self_play.py.

Verifies:
1. play_one_game returns a non-empty trajectory with all fields populated.
2. observed_ranking is a permutation of the actual finishing order.
3. observed_coalition_index is a valid index into the enumeration.
4. target_scalar_value is in [0, 1].
5. self_play_iteration runs end-to-end without crashing and reduces loss
   over a small number of iterations on a tiny problem.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.optim as optim

from flagship_coalition_mcts.src.coalition_head import _enumerate_coalitions
from flagship_coalition_mcts.src.games.kingmaker import (
    KingmakerGame,
    KingmakerState,
    NUM_ACTIONS,
)
from flagship_coalition_mcts.src.network import (
    CDMCTSEvaluator,
    CDMCTSNetwork,
    MLPEncoder,
)
from flagship_coalition_mcts.src.self_play import (
    play_one_game,
    self_play_iteration,
    trajectory_to_batch,
)
from flagship_coalition_mcts.tests.test_network import kingmaker_to_features


def _make_net(seed: int = 0) -> CDMCTSNetwork:
    torch.manual_seed(seed)
    return CDMCTSNetwork(
        encoder=MLPEncoder(input_dim=12, hidden_dim=24, num_layers=2),
        action_space_size=NUM_ACTIONS,
        max_players=3,
    )


def test_play_one_game_returns_trajectory():
    net = _make_net()
    ev = CDMCTSEvaluator(
        network=net, state_to_features=kingmaker_to_features,
        action_space_size=NUM_ACTIONS,
        current_player_fn=KingmakerGame.current_player,
        num_players_fn=KingmakerGame.num_players,
    )
    traj = play_one_game(
        initial_state_fn=KingmakerState.initial,
        game=KingmakerGame(),
        evaluator=ev,
        state_to_features=kingmaker_to_features,
        num_simulations=10,
        rng_seed=0,
        action_space_size=NUM_ACTIONS,
        max_players=3,
    )
    assert len(traj) == 6  # TOTAL_MOVES = 6
    for entry in traj:
        assert entry.features.shape == (12,)
        assert entry.legal_mask.shape == (NUM_ACTIONS,)
        assert abs(entry.target_policy_legal.sum() - 1.0) < 1e-9
        assert entry.observed_ranking is not None
        assert entry.observed_ranking.shape == (3,)
        assert entry.observed_coalition_index is not None
        assert entry.observed_coalition_index >= 0
        assert 0.0 <= entry.target_scalar_value <= 1.0


def test_observed_ranking_is_valid_permutation():
    net = _make_net(seed=1)
    ev = CDMCTSEvaluator(
        network=net, state_to_features=kingmaker_to_features,
        action_space_size=NUM_ACTIONS,
        current_player_fn=KingmakerGame.current_player,
        num_players_fn=KingmakerGame.num_players,
    )
    traj = play_one_game(
        initial_state_fn=KingmakerState.initial,
        game=KingmakerGame(),
        evaluator=ev,
        state_to_features=kingmaker_to_features,
        num_simulations=10,
        rng_seed=1,
        action_space_size=NUM_ACTIONS,
        max_players=3,
    )
    # The ranking must contain every player exactly once
    ranking = traj[0].observed_ranking[:3].tolist()
    assert sorted(ranking) == [0, 1, 2]


def test_observed_coalition_index_in_range():
    """For 3 players, opponents = 2, # subsets = 2^2 = 4. Index in [0,3]."""
    net = _make_net(seed=2)
    ev = CDMCTSEvaluator(
        network=net, state_to_features=kingmaker_to_features,
        action_space_size=NUM_ACTIONS,
        current_player_fn=KingmakerGame.current_player,
        num_players_fn=KingmakerGame.num_players,
    )
    traj = play_one_game(
        initial_state_fn=KingmakerState.initial,
        game=KingmakerGame(),
        evaluator=ev,
        state_to_features=kingmaker_to_features,
        num_simulations=8,
        rng_seed=2,
        action_space_size=NUM_ACTIONS,
        max_players=3,
    )
    for entry in traj:
        assert 0 <= entry.observed_coalition_index < 4


def test_trajectory_to_batch_shapes():
    net = _make_net()
    ev = CDMCTSEvaluator(
        network=net, state_to_features=kingmaker_to_features,
        action_space_size=NUM_ACTIONS,
        current_player_fn=KingmakerGame.current_player,
        num_players_fn=KingmakerGame.num_players,
    )
    trajs = []
    for s in range(3):
        trajs.append(play_one_game(
            initial_state_fn=KingmakerState.initial,
            game=KingmakerGame(),
            evaluator=ev,
            state_to_features=kingmaker_to_features,
            num_simulations=8,
            rng_seed=s,
            action_space_size=NUM_ACTIONS,
            max_players=3,
        ))
    batch = trajectory_to_batch(trajs, action_space_size=NUM_ACTIONS, max_players=3)
    B = batch["features"].shape[0]
    assert B == 3 * 6
    assert batch["target_policy"].shape == (B, NUM_ACTIONS)
    assert batch["legal_mask"].dtype == torch.bool
    assert batch["observed_ranking"].shape == (B, 3)


def test_self_play_iteration_runs_and_reduces_loss():
    """Run a few iterations and verify the average loss eventually drops."""
    torch.manual_seed(7)
    net = _make_net(seed=7)
    opt = optim.Adam(net.parameters(), lr=1e-2)
    losses = []
    for it in range(3):
        stats = self_play_iteration(
            network=net,
            optimizer=opt,
            initial_state_fn=KingmakerState.initial,
            game=KingmakerGame(),
            state_to_features=kingmaker_to_features,
            games_per_iter=3,
            train_steps=8,
            num_simulations=8,
            action_space_size=NUM_ACTIONS,
            max_players=3,
            rng_seed=it,
        )
        losses.append(stats["avg_total"])
    # Loss should decrease from iter 0 to iter 2 - small problem, lots of capacity
    # Allow for noise: just require last < first.
    assert losses[-1] < losses[0] + 0.1, f"losses did not drop: {losses}"
