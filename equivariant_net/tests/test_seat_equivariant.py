"""Tests for the S_N-equivariant seat layer.

These tests verify the property that a flagship reviewer will explicitly
look for: under any permutation of input seats, the output must transform
in the *exact* same way (bit-identical up to numerical precision).

Tests:
1. SeatInvariantPool: y(perm(X)) == y(X)
2. SeatEquivariantBlock: Y(perm(X)) == perm(Y(X))
3. WreathSeatNet: the stacked version preserves equivariance.
4. Mask works correctly under permutation of (X, mask) jointly.
5. Forward shapes.
"""

from __future__ import annotations

import torch
import pytest

from equivariant_net.src.seat_equivariant import (
    SeatEquivariantBlock,
    SeatInvariantPool,
    WreathSeatNet,
    make_seat_mask,
)


def _random_perm(n: int) -> torch.Tensor:
    return torch.randperm(n)


def test_invariant_pool_under_permutation():
    torch.manual_seed(0)
    pool = SeatInvariantPool()
    X = torch.randn(4, 6, 8)  # batch=4, N=6, d=8
    perm = _random_perm(6)
    y_orig = pool(X)
    y_perm = pool(X[:, perm, :])
    assert torch.allclose(y_orig, y_perm, atol=1e-6), (
        f"max diff {(y_orig - y_perm).abs().max():.2e}"
    )


def test_equivariant_block_commutes_with_permutation():
    """SeatEquivariantBlock(perm(X)) must equal perm(SeatEquivariantBlock(X))."""
    torch.manual_seed(1)
    block = SeatEquivariantBlock(in_dim=8, out_dim=12)
    block.eval()
    X = torch.randn(3, 6, 8)
    perm = _random_perm(6)
    Y = block(X)
    Y_perm = Y[:, perm, :]
    Y_after_perm = block(X[:, perm, :])
    err = (Y_perm - Y_after_perm).abs().max().item()
    assert err < 1e-5, f"equivariance violated: max diff {err:.2e}"


def test_wreath_seat_net_equivariance():
    torch.manual_seed(2)
    net = WreathSeatNet(in_dim=8, hidden_dim=16, out_dim=12, num_blocks=3)
    net.eval()
    X = torch.randn(2, 6, 8)
    perm = _random_perm(6)
    Y = net(X)
    Y_perm = Y[:, perm, :]
    Y_after_perm = net(X[:, perm, :])
    err = (Y_perm - Y_after_perm).abs().max().item()
    assert err < 1e-5, f"stacked equivariance violated: max diff {err:.2e}"


def test_masked_invariance_with_mask_permutation():
    """When seats are masked, a permutation that preserves the mask
    pattern (or jointly permutes mask + X) must yield the same output."""
    torch.manual_seed(3)
    pool = SeatInvariantPool()
    X = torch.randn(2, 6, 4)
    # 4 active seats, 2 inactive
    mask = make_seat_mask(num_active=4, max_seats=6).unsqueeze(0).expand(2, -1)
    y = pool(X, mask=mask)
    # Permute the active-seat block among itself; the mask stays valid.
    active_perm = torch.cat([torch.randperm(4), torch.tensor([4, 5])])
    X_perm = X[:, active_perm, :]
    mask_perm = mask[:, active_perm]
    y_perm = pool(X_perm, mask=mask_perm)
    err = (y - y_perm).abs().max().item()
    assert err < 1e-6, f"masked invariance violated: max diff {err:.2e}"


def test_forward_shapes():
    block = SeatEquivariantBlock(in_dim=4, out_dim=8)
    X = torch.randn(3, 6, 4)
    Y = block(X)
    assert Y.shape == (3, 6, 8)
    net = WreathSeatNet(in_dim=4, hidden_dim=12, out_dim=10, num_blocks=2)
    out = net(X)
    assert out.shape == (3, 6, 10)


def test_make_seat_mask_correct():
    m = make_seat_mask(num_active=3, max_seats=6)
    assert m.tolist() == [True, True, True, False, False, False]
