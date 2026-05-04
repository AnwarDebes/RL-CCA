"""Tests for the CD-MCTS network module.

Verifies:
1. Forward pass returns the expected shapes for all five outputs.
2. CDMCTSEvaluator returns a valid NetworkOutput compatible with mcts.py.
3. Joint loss has finite gradients on a synthetic batch.
4. Network can drive an end-to-end MCTS run (no torch errors).
5. PL theta passes through to placement marginals correctly.
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
from flagship_coalition_mcts.src.mcts import NetworkOutput, run_mcts
from flagship_coalition_mcts.src.network import (
    CDMCTSEvaluator,
    CDMCTSNetwork,
    MLPEncoder,
    cdmcts_loss,
)


def kingmaker_to_features(state: KingmakerState) -> np.ndarray:
    # 12-dim feature: 3 normalized positions + 3 one-hot finished + 3
    # one-hot turn_player + 3 normalized move_count (one-hot of 6 truncated to 3).
    feats = np.zeros(12, dtype=np.float32)
    feats[0:3] = np.array(state.positions) / 3.0
    for p in state.finish_order:
        feats[3 + p] = 1.0
    feats[6 + state.next_player] = 1.0
    feats[9] = state.move_count / 6.0
    feats[10] = float(len(state.finish_order)) / 3.0
    feats[11] = 1.0  # bias
    return feats


def make_test_network(seed: int = 0) -> CDMCTSNetwork:
    torch.manual_seed(seed)
    encoder = MLPEncoder(input_dim=12, hidden_dim=32, num_layers=2)
    return CDMCTSNetwork(
        encoder=encoder, action_space_size=NUM_ACTIONS, max_players=3
    )


def test_forward_shapes():
    net = make_test_network()
    x = torch.randn(5, 12)
    pl, theta, A, beta, sv = net(x)
    assert pl.shape == (5, NUM_ACTIONS)
    assert theta.shape == (5, 3)
    assert A.shape == (5, 3, 3)
    assert beta.shape == (5,)
    assert sv.shape == (5,)


def test_evaluator_returns_valid_output():
    net = make_test_network()
    ev = CDMCTSEvaluator(
        network=net,
        state_to_features=kingmaker_to_features,
        action_space_size=NUM_ACTIONS,
        current_player_fn=KingmakerGame.current_player,
        num_players_fn=KingmakerGame.num_players,
    )
    out = ev.evaluate(KingmakerState.initial())
    assert isinstance(out, NetworkOutput)
    assert out.prior_policy.shape == (NUM_ACTIONS,)
    assert abs(out.prior_policy.sum() - 1.0) < 1e-5
    assert out.placement_marginals.shape == (3, 3)
    # Each row of M should sum to 1 (full distribution over positions)
    assert np.allclose(out.placement_marginals.sum(axis=1), 1.0, atol=1e-5)
    assert out.coalition_alignment.shape == (3,)
    # Self-alignment is zero
    cp = KingmakerGame.current_player(KingmakerState.initial())
    assert abs(out.coalition_alignment[cp]) < 1e-6


def test_end_to_end_mcts_with_real_network():
    net = make_test_network(seed=42)
    ev = CDMCTSEvaluator(
        network=net,
        state_to_features=kingmaker_to_features,
        action_space_size=NUM_ACTIONS,
        current_player_fn=KingmakerGame.current_player,
        num_players_fn=KingmakerGame.num_players,
    )
    root, pi = run_mcts(
        state=KingmakerState.initial(),
        network=ev,
        game=KingmakerGame(),
        num_simulations=30,
        seed=0,
    )
    assert pi.shape[0] == len(KingmakerGame.legal_actions(KingmakerState.initial()))
    assert abs(pi.sum() - 1.0) < 1e-9
    assert root.selector_state.visits.sum() == 30


def test_loss_has_finite_gradients():
    net = make_test_network()
    B = 4
    feats = torch.randn(B, 12)
    target_policy = torch.softmax(torch.randn(B, NUM_ACTIONS), dim=-1)
    legal_mask = torch.ones(B, NUM_ACTIONS, dtype=torch.bool)
    legal_mask[:, 0] = False  # arbitrary illegal action for testing
    # Renormalise target_policy over legal actions
    target_policy = target_policy * legal_mask.float()
    target_policy = target_policy / target_policy.sum(dim=-1, keepdim=True).clamp(min=1e-9)
    # Random observed rankings
    obs_rank = torch.tensor(
        [[2, 1, 0], [0, 2, 1], [1, 0, 2], [2, 0, 1]], dtype=torch.long
    )
    nplayers = torch.full((B,), 3, dtype=torch.long)
    cp = torch.tensor([0, 1, 2, 0], dtype=torch.long)
    obs_coal_idx = torch.tensor([0, 1, 2, 0], dtype=torch.long)  # arbitrary
    target_v = torch.tensor([0.5, -0.5, 0.0, 0.7])

    loss, comps = cdmcts_loss(
        net, feats, target_policy, legal_mask,
        obs_rank, nplayers, obs_coal_idx, cp, target_v,
    )
    loss.backward()
    # All parameters should have finite gradients
    for n, p in net.named_parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all(), f"non-finite grad in {n}"
    # Components should be non-negative
    assert comps["policy"] >= 0
    assert comps["pl"] >= -1e-6  # PL log-lik can drift very slightly negative due to fp
    assert comps["coalition"] >= -1e-6
    assert comps["value"] >= 0


def test_pl_theta_drives_marginals():
    """If we manually overwrite theta to favour player 0, the placement
    marginal for player 0 in position 1 should be highest."""
    net = make_test_network()
    # Manually set the PL head's bias so theta = [10, 0, 0] regardless of input
    with torch.no_grad():
        net.pl_head.proj.weight.zero_()
        net.pl_head.proj.bias.copy_(torch.tensor([10.0, 0.0, 0.0]))
    ev = CDMCTSEvaluator(
        network=net,
        state_to_features=kingmaker_to_features,
        action_space_size=NUM_ACTIONS,
        current_player_fn=KingmakerGame.current_player,
        num_players_fn=KingmakerGame.num_players,
    )
    out = ev.evaluate(KingmakerState.initial())
    M = out.placement_marginals
    # P(player 0 in position 1) should dominate
    assert M[0, 0] > 0.99
    assert M[1, 0] < 0.01
    assert M[2, 0] < 0.01
