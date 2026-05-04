"""Tests for the full wreath-equivariant CC network.

Verifies:
1. Forward output shapes for batch and single-input.
2. Gradients flow.
3. Seat permutation invariance: permuting active seats (and the seat
   features accordingly) leaves the network output unchanged (for the
   pooled invariant outputs: policy logits depend on the *current
   player* identity which is encoded in seat features, but the
   invariant trunk is unchanged).
"""

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
    from equivariant_net.src.wreath_network import (
        WreathCCNetwork,
        cc_seat_features,
    )
    HAVE = True
except ImportError:
    HAVE = False

pytestmark = pytest.mark.skipif(
    not HAVE, reason="WreathCCNetwork requires nexus core/* modules",
)


def test_forward_shapes_batched():
    net = WreathCCNetwork(
        spatial_channels=4, spatial_blocks=2, spatial_out=16,
        seat_hidden=8, seat_blocks=1,
    )
    net.eval()
    state_tensor = torch.randn(2, 32, 17, 17)
    seat_features = torch.randn(2, 6, 8)
    seat_mask = torch.ones(2, 6, dtype=torch.bool)
    seat_mask[:, 3:] = False  # 3 active seats per batch element
    pi, v = net(state_tensor, seat_features, seat_mask)
    assert pi.shape == (2, 1210)
    assert v.shape == (2,)


def test_forward_unbatched():
    net = WreathCCNetwork(
        spatial_channels=4, spatial_blocks=1, spatial_out=8,
        seat_hidden=4, seat_blocks=1,
    )
    net.eval()
    st = torch.randn(32, 17, 17)
    sf = torch.randn(6, 8)
    pi, v = net(st, sf)
    assert pi.shape == (1, 1210)
    assert v.shape == (1,)


def test_gradients_flow():
    net = WreathCCNetwork(
        spatial_channels=2, spatial_blocks=1, spatial_out=4,
        seat_hidden=4, seat_blocks=1,
    )
    state_tensor = torch.randn(1, 32, 17, 17, requires_grad=True)
    seat_features = torch.randn(1, 6, 8, requires_grad=True)
    pi, v = net(state_tensor, seat_features)
    loss = pi.sum() + v.sum()
    loss.backward()
    for n, p in net.named_parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all(), f"non-finite grad in {n}"


def test_seat_permutation_invariance_of_pooled_features():
    """Permuting active seats jointly across seat_features should leave
    the *pooled* invariant features unchanged. Note: the policy logits
    reflect the action space which is tied to specific seats - so the
    test is on the invariant pool, not the final policy."""
    net = WreathCCNetwork(
        spatial_channels=2, spatial_blocks=1, spatial_out=4,
        seat_hidden=4, seat_blocks=1,
    )
    net.eval()
    state_tensor = torch.randn(1, 32, 17, 17)
    seat_features = torch.randn(1, 6, 8)
    seat_mask = torch.ones(1, 6, dtype=torch.bool)

    # Get the seat-pool output directly via the seat net + pool
    h_seat_eq = net.seat_net(seat_features, mask=seat_mask)
    h_inv = net.seat_pool(h_seat_eq, mask=seat_mask)

    # Permute active seats
    perm = torch.randperm(6)
    seat_features_perm = seat_features[:, perm]
    h_seat_eq_perm = net.seat_net(seat_features_perm, mask=seat_mask[:, perm])
    h_inv_perm = net.seat_pool(h_seat_eq_perm, mask=seat_mask[:, perm])

    err = (h_inv - h_inv_perm).abs().max().item()
    assert err < 1e-5, f"seat-pool invariance violated: {err:.2e}"


def test_cc_seat_features_shape_with_real_env():
    from flagship_coalition_mcts.src.games.chinese_checkers import make_cc_env
    env = make_cc_env(num_players=4, seed=0)
    feats = cc_seat_features(env, max_seats=6)
    assert feats.shape == (6, 8)
    # Active seats have nonzero "is_active"
    assert feats[0, 1] == 1.0
    assert feats[3, 1] == 1.0
    # Inactive seats are zeros
    assert (feats[4:] == 0).all()
    # Exactly one current player
    cp_flags = feats[:4, 0]
    assert int(cp_flags.sum()) == 1
