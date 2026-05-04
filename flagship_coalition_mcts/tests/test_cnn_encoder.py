"""Tests for the CNN encoder for Chinese Checkers.

Verifies:
1. Forward shapes for batch and single-input.
2. Trainable: gradients flow.
3. End-to-end with CDMCTSNetwork (encoder substituted in).
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from flagship_coalition_mcts.src.cnn_encoder import (
    CCCNNEncoder,
    CCEncoderForCMAZ,
    feature_to_tensor,
)
from flagship_coalition_mcts.src.network import CDMCTSNetwork


def test_forward_batch_shape():
    enc = CCCNNEncoder(in_channels=32, channels=16, num_blocks=2, out_dim=64)
    enc.eval()
    x = torch.randn(3, 32, 17, 17)
    h = enc(x)
    assert h.shape == (3, 64)


def test_forward_single_input_unbatched():
    enc = CCCNNEncoder(in_channels=32, channels=16, num_blocks=2, out_dim=64)
    enc.eval()
    x = torch.randn(32, 17, 17)  # no batch dim
    h = enc(x)
    assert h.shape == (1, 64)


def test_gradients_flow():
    enc = CCCNNEncoder(in_channels=8, channels=12, num_blocks=2, out_dim=16)
    x = torch.randn(2, 8, 17, 17, requires_grad=True)
    h = enc(x)
    loss = h.sum()
    loss.backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    for n, p in enc.named_parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all(), f"non-finite grad in {n}"


def test_substituted_into_cdmcts_network():
    """CDMCTSNetwork should accept CCCNNEncoder (just needs .out_dim)."""
    enc = CCCNNEncoder(in_channels=32, channels=16, num_blocks=2, out_dim=32)
    net = CDMCTSNetwork(encoder=enc, action_space_size=1210, max_players=6)
    net.eval()
    x = torch.randn(2, 32, 17, 17)
    pl_logits, theta, A, beta, sv = net(x)
    assert pl_logits.shape == (2, 1210)
    assert theta.shape == (2, 6)
    assert A.shape == (2, 6, 6)
    assert beta.shape == (2,)
    assert sv.shape == (2,)


def test_feature_to_tensor():
    arr = np.random.randn(32, 17, 17).astype(np.float32)
    t = feature_to_tensor(arr)
    assert t.shape == (1, 32, 17, 17)
    assert t.dtype == torch.float32
