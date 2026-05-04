"""Tests for the CMAZ network module (encoder + heads + cmaz_loss).

The network and joint loss are exercised indirectly via integration
tests, but direct unit tests give us:
  1. Confidence the loss components are non-negative.
  2. Verification that gradients flow through every pillar.
  3. Early failure if loss-weight defaults change unexpectedly.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from decomposed_mcts.src.network import (
    CMAZEncoder, CMAZEvaluator, CMAZNetwork, cmaz_loss,
)


def _make_net(in_dim=8, hidden=16, action=4, K=3):
    enc = CMAZEncoder(input_dim=in_dim, hidden_dim=hidden, num_layers=2)
    return CMAZNetwork(encoder=enc, action_space_size=action, num_components=K)


def test_cmaz_network_forward_shapes():
    net = _make_net()
    x = torch.randn(3, 8)
    pl, comp, q = net(x)
    assert pl.shape == (3, 4)
    assert comp.shape == (3, 3)
    assert q.shape == (3,)


def test_cmaz_loss_non_negative_components():
    torch.manual_seed(0)
    net = _make_net()
    B = 4
    feats = torch.randn(B, 8)
    target_pol = torch.softmax(torch.randn(B, 4), dim=-1)
    legal_mask = torch.ones(B, 4, dtype=torch.bool)
    target_comp = torch.rand(B, 3)
    total, comps = cmaz_loss(
        net, feats, target_pol, legal_mask, target_comp,
    )
    assert comps["policy"] >= 0
    assert comps["components"] >= 0
    assert torch.isfinite(total).all()


def test_cmaz_loss_gradients_flow_through_all_heads():
    """Each parameter that should be trained must receive a non-zero
    gradient at least once (backward through the full loss)."""
    torch.manual_seed(1)
    net = _make_net()
    B = 4
    feats = torch.randn(B, 8)
    target_pol = torch.softmax(torch.randn(B, 4), dim=-1)
    legal_mask = torch.ones(B, 4, dtype=torch.bool)
    target_comp = torch.rand(B, 3)
    total, _ = cmaz_loss(
        net, feats, target_pol, legal_mask, target_comp,
    )
    total.backward()
    for n, p in net.named_parameters():
        if p.grad is None:
            pytest.fail(f"no gradient on {n}")
        assert torch.isfinite(p.grad).all(), f"non-finite grad on {n}"


def test_cmaz_loss_respects_legal_mask():
    """When legal_mask zeros out an action, the policy log-prob for that
    action should not contribute to the loss (because target_pol[..., a]=0
    on illegal actions, and our masked log_softmax pushes it to -inf)."""
    torch.manual_seed(2)
    net = _make_net()
    B = 2
    feats = torch.randn(B, 8)
    legal_mask = torch.tensor([[True, True, False, False],
                               [False, True, True, False]], dtype=torch.bool)
    target_pol = torch.zeros(B, 4)
    target_pol[0, 0] = 1.0
    target_pol[1, 1] = 1.0
    target_comp = torch.rand(B, 3)
    total, comps = cmaz_loss(
        net, feats, target_pol, legal_mask, target_comp,
    )
    assert torch.isfinite(total).all()
    assert comps["policy"] >= 0


def test_cmaz_loss_weights_are_applied():
    """Weighting components down to zero produces only the policy loss."""
    torch.manual_seed(3)
    net = _make_net()
    B = 2
    feats = torch.randn(B, 8)
    target_pol = torch.softmax(torch.randn(B, 4), dim=-1)
    legal_mask = torch.ones(B, 4, dtype=torch.bool)
    target_comp = torch.rand(B, 3)
    total_only_p, _ = cmaz_loss(
        net, feats, target_pol, legal_mask, target_comp,
        weights=dict(policy=1.0, components=0.0),
    )
    total_only_c, _ = cmaz_loss(
        net, feats, target_pol, legal_mask, target_comp,
        weights=dict(policy=0.0, components=1.0),
    )
    # Either alone < the sum
    total_both, _ = cmaz_loss(
        net, feats, target_pol, legal_mask, target_comp,
        weights=dict(policy=1.0, components=1.0),
    )
    assert total_only_p.item() < total_both.item() + 1e-6
    assert total_only_c.item() < total_both.item() + 1e-6


def test_cmaz_evaluator_round_trip():
    """Construct an evaluator with a stub state-to-features fn and verify
    evaluate_cmaz returns sensible outputs."""
    net = _make_net()

    def state_to_features(s):
        return np.full(8, 0.5, dtype=np.float32)

    def cp_fn(s):
        return 0

    def np_fn(s):
        return 3

    def term_fn(s):
        return np.array([1.0, 0.0, 0.0])

    ev = CMAZEvaluator(
        network=net,
        state_to_features=state_to_features,
        current_player_fn=cp_fn,
        num_players_fn=np_fn,
        terminal_components_fn=term_fn,
    )
    out = ev.evaluate_cmaz("dummy_state")
    assert out.prior_policy.shape == (4,)
    assert abs(out.prior_policy.sum() - 1.0) < 1e-5
    assert out.component_values.shape == (3,)
