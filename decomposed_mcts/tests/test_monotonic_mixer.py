"""Unit tests for the CMAZ monotonic mixer.

Tests verify the four properties a flagship-quality QMIX-style mixer must satisfy:

1. Mixer weights w(s) sum to 1 (softmax property).
2. Monotonicity: increasing any v_k strictly increases Q (when w_k > 0).
3. Inference-time override: passing an override_weights tensor produces
   exactly that linear combination plus bias.
4. Gradient signs: dQ/dv_k > 0 for all k (when w_k > 0).
5. Bias term: passing v=0 returns b(s) exactly.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from decomposed_mcts.src.monotonic_mixer import (
    ComponentValueHead,
    MonotonicMixer,
)


def test_get_weights_sums_to_one():
    torch.manual_seed(0)
    m = MonotonicMixer(feature_dim=8, num_components=4)
    feats = torch.randn(5, 8)
    w, b = m.get_weights(feats)
    assert w.shape == (5, 4)
    assert (w >= 0).all()
    assert torch.allclose(w.sum(dim=-1), torch.ones(5), atol=1e-6)


def test_monotonicity_in_each_component():
    """Increasing v_k while holding others fixed must not decrease Q."""
    torch.manual_seed(1)
    m = MonotonicMixer(feature_dim=4, num_components=3)
    feats = torch.randn(1, 4)
    v_lo = torch.tensor([[0.0, 0.0, 0.0]])
    Q_lo = m(v_lo, feats)
    for k in range(3):
        v_hi = v_lo.clone()
        v_hi[0, k] = 1.0
        Q_hi = m(v_hi, feats)
        assert Q_hi.item() >= Q_lo.item() - 1e-6, (
            f"non-monotone in component {k}: lo={Q_lo.item()}, hi={Q_hi.item()}"
        )


def test_zero_v_returns_bias_only():
    torch.manual_seed(2)
    m = MonotonicMixer(feature_dim=4, num_components=3)
    feats = torch.randn(1, 4)
    v = torch.zeros(1, 3)
    Q = m(v, feats)
    _, b = m.get_weights(feats)
    assert torch.allclose(Q, b, atol=1e-6)


def test_inference_time_override():
    """Passing override_weights = [1, 0, 0, 0] should give Q = v[0] + b."""
    torch.manual_seed(3)
    m = MonotonicMixer(feature_dim=4, num_components=4)
    feats = torch.randn(1, 4)
    v = torch.tensor([[0.3, -0.7, 0.5, 0.1]])
    override = torch.tensor([1.0, 0.0, 0.0, 0.0])
    Q = m(v, feats, override_weights=override)
    _, b = m.get_weights(feats)
    expected = v[0, 0] + b[0]
    assert torch.allclose(Q[0], expected, atol=1e-6), f"Q={Q[0]}, expected={expected}"


def test_inference_override_renormalises():
    """Override weights that don't sum to 1 should be renormalised."""
    torch.manual_seed(4)
    m = MonotonicMixer(feature_dim=4, num_components=3)
    feats = torch.randn(1, 4)
    v = torch.tensor([[1.0, 1.0, 1.0]])
    override = torch.tensor([2.0, 2.0, 2.0])  # uniform after renorm
    Q = m(v, feats, override_weights=override)
    _, b = m.get_weights(feats)
    expected = 1.0 + b[0]
    assert torch.allclose(Q[0], expected, atol=1e-6)


def test_gradients_have_correct_signs():
    """dQ/dv_k = w_k > 0."""
    torch.manual_seed(5)
    m = MonotonicMixer(feature_dim=4, num_components=3)
    feats = torch.randn(1, 4)
    v = torch.zeros(1, 3, requires_grad=True)
    Q = m(v, feats)
    Q.sum().backward()
    assert (v.grad >= 0).all(), f"negative gradient: {v.grad}"
    # Sum of gradients = sum of w_k = 1
    assert abs(v.grad.sum().item() - 1.0) < 1e-6


def test_component_value_head():
    head = ComponentValueHead(feature_dim=8, num_components=4)
    x = torch.randn(3, 8)
    v = head(x)
    assert v.shape == (3, 4)
    assert (v >= -1).all() and (v <= 1).all()
