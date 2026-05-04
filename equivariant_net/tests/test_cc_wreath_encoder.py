"""Tests for the wreath-equivariant CC encoder.

Verifies:
1. Forward shape: (B, 32, 17, 17) → (B, hidden_dim).
2. Rotation permutation table is well-formed.
3. The encoder's *invariant* output is identical under any C6 rotation
   of the input (this is the killer property the paper claims for the
   equivariant trunk).
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
    from equivariant_net.src.cc_wreath_encoder import (
        CCWreathEncoder,
        _build_cc_rotation_permutation,
        _get_board_axial_coords,
    )
    HAVE_CC = True
except ImportError:
    HAVE_CC = False

pytestmark = pytest.mark.skipif(
    not HAVE_CC, reason="CC wreath encoder requires nexus core/board.py",
)


def test_axial_and_grid_coords_lengths_match():
    axial, grid = _get_board_axial_coords()
    assert len(axial) == len(grid) == 121


def test_rotation_permutation_shape_and_identity_at_zero():
    axial, _ = _get_board_axial_coords()
    P = _build_cc_rotation_permutation(axial)
    assert P.shape == (6, 121)
    # k=0 must be the identity.
    assert (P[0] == np.arange(121)).all()


def test_rotation_six_times_returns_identity():
    """Applying 6 successive 60° rotations (each via P[1]) must be the identity."""
    axial, _ = _get_board_axial_coords()
    P = _build_cc_rotation_permutation(axial)
    # Compose P[1] six times.
    composed = np.arange(121)
    for _ in range(6):
        composed = P[1, composed]
    # On the interior cells (those whose orbit stays on the board),
    # composed must equal arange.
    # The board is symmetric so this should hold exactly.
    assert (composed == np.arange(121)).all()


def test_forward_shape():
    enc = CCWreathEncoder(in_channels=32, c_spatial=8, hidden_dim=32, num_blocks=2)
    enc.eval()
    x = torch.randn(2, 32, 17, 17)
    h = enc(x)
    assert h.shape == (2, 32)


def test_invariant_pooled_output_under_rotation_via_grid():
    """Apply a C6 rotation to the INPUT state tensor (by permuting the
    gx, gy indices via the rotation table) and verify the encoder's
    pooled output is unchanged (the head is fully invariant, by design)."""
    enc = CCWreathEncoder(in_channels=8, c_spatial=4, hidden_dim=8, num_blocks=2)
    enc.eval()

    axial, grid = _get_board_axial_coords()
    P = _build_cc_rotation_permutation(axial)
    # Pre-rotation: input tensor x ∈ (1, 8, 17, 17). Build a tensor whose
    # value at (gx[i], gy[i]) is some function of i; rotation via P[k]
    # moves the value at i to position P[k, i].
    torch.manual_seed(0)
    x = torch.zeros(1, 8, 17, 17)
    # Fill only the 121 valid cells with random data.
    rand_per_cell = torch.randn(1, 8, 121)
    gx = torch.tensor([g[0] for g in grid], dtype=torch.long)
    gy = torch.tensor([g[1] for g in grid], dtype=torch.long)
    for i in range(121):
        x[:, :, gx[i], gy[i]] = rand_per_cell[:, :, i]

    # Build rotated input: value at cell i in original goes to cell P[k, i].
    # Equivalently, value at cell j in rotated = value at cell P^{-1}[k, j]
    # in original.
    k = 1  # 60° rotation
    P_inv = np.argsort(P[k])  # P_inv[j] = i s.t. P[k, i] = j
    x_rot = torch.zeros_like(x)
    for j in range(121):
        x_rot[:, :, gx[j], gy[j]] = x[:, :, gx[P_inv[j]], gy[P_inv[j]]]

    # Forward both
    h_orig = enc(x)
    h_rot = enc(x_rot)
    # The encoder's output is INVARIANT (because we average-pool over
    # rotation channels at the end). So h_orig must equal h_rot
    # bit-identically (within fp tolerance).
    err = (h_orig - h_rot).abs().max().item()
    assert err < 1e-4, f"rotation invariance violated: {err:.2e}"
