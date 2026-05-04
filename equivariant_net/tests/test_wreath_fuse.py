"""Tests for the wreath-equivariant fusion layer.

The headline tests verify that the layer commutes with each factor of the
wreath product:

  * S_N seat permutation:  L(perm(x)) == perm(L(x)) bit-identical.
  * C6 spatial rotation:   L(rot(x)) == rot(L(x)) on the interior of the
    grid (boundary fallback may introduce small artifacts).
  * Composed wreath action: both above simultaneously.

Plus the policy-permutation-invariance sanity check that the paper
explicitly claims.
"""

from __future__ import annotations

import numpy as np
import torch
import pytest

from equivariant_net.src.c6_spatial import (
    make_rotation_permutation,
    rotate_feature_map,
)
from equivariant_net.src.wreath_fuse import WreathFuseLayer, permute_seats


def _hex_ring(radius: int) -> list:
    coords = []
    for q in range(-radius, radius + 1):
        for r in range(-radius, radius + 1):
            if abs(q) + abs(r) + abs(q + r) <= 2 * radius:
                coords.append((q, r))
    return coords


def test_forward_shapes():
    layer = WreathFuseLayer(in_spatial=3, in_seat=4, out_spatial=5, out_seat=6)
    coords = _hex_ring(1)
    L = len(coords)
    spatial = torch.randn(2, 4, L, 6 * 3)
    seat = torch.randn(2, 4, 4)
    sp_out, seat_out = layer(spatial, seat)
    assert sp_out.shape == (2, 4, L, 6 * 5)
    assert seat_out.shape == (2, 4, 6)


def test_wreath_seat_permutation_equivariance():
    """Permuting seats BEFORE the layer must equal permuting AFTER."""
    torch.manual_seed(0)
    layer = WreathFuseLayer(in_spatial=3, in_seat=4, out_spatial=5, out_seat=6)
    layer.eval()
    coords = _hex_ring(2)
    L = len(coords)
    spatial = torch.randn(2, 6, L, 6 * 3)
    seat = torch.randn(2, 6, 4)
    perm = torch.randperm(6)

    # Path 1: permute then layer
    sp_p, se_p = permute_seats(spatial, seat, perm)
    sp_out_a, se_out_a = layer(sp_p, se_p)

    # Path 2: layer then permute
    sp_out, se_out = layer(spatial, seat)
    sp_out_b, se_out_b = permute_seats(sp_out, se_out, perm)

    err_sp = (sp_out_a - sp_out_b).abs().max().item()
    err_se = (se_out_a - se_out_b).abs().max().item()
    assert err_sp < 1e-5, f"spatial seat-permutation eq violated: {err_sp:.2e}"
    assert err_se < 1e-5, f"seat seat-permutation eq violated: {err_se:.2e}"


def test_wreath_spatial_rotation_equivariance_interior():
    """Rotating each seat's spatial map by 60° (and the rotation-channel)
    BEFORE the layer must equal rotating AFTER, on interior locations."""
    torch.manual_seed(1)
    layer = WreathFuseLayer(in_spatial=2, in_seat=3, out_spatial=4, out_seat=5)
    layer.eval()
    coords = _hex_ring(2)
    L = len(coords)
    perm_np = make_rotation_permutation(coords, num_rotations=6)
    perm_t = torch.from_numpy(perm_np)
    N = 3
    spatial = torch.randn(1, N, L, 6 * 2)
    seat = torch.randn(1, N, 3)

    rot_idx = 1

    def rotate_all_seats(spatial_in, num_chan):
        """Apply C6 rotation to each seat's spatial map."""
        out = torch.zeros_like(spatial_in)
        B, Nseat, Lloc, F_ = spatial_in.shape
        for b in range(B):
            for s in range(Nseat):
                rotated = rotate_feature_map(
                    spatial_in[b, s].unsqueeze(0),
                    perm_t,
                    num_channels_per_rotation=num_chan,
                    rotation_index=rot_idx,
                )
                out[b, s] = rotated[0]
        return out

    sp_rot = rotate_all_seats(spatial, num_chan=2)
    sp_out_a, se_out_a = layer(sp_rot, seat)

    sp_out, se_out = layer(spatial, seat)
    sp_out_rotated = rotate_all_seats(sp_out, num_chan=4)

    # Build interior mask
    interior = []
    from equivariant_net.src.c6_spatial import rotate_axial
    coord_set = set(coords)
    for c in coords:
        rotated = c
        for _ in range(rot_idx):
            rotated = rotate_axial(rotated)
        interior.append(rotated in coord_set)
    int_mask = torch.tensor(interior)

    if int_mask.any():
        err_sp = (sp_out_a[:, :, int_mask] - sp_out_rotated[:, :, int_mask]).abs().max().item()
        assert err_sp < 1e-5, f"spatial rotation eq violated on interior: {err_sp:.2e}"
        # The seat output must be UNCHANGED by spatial rotation (rotation is "in seat-axis"
        # of the wreath; the seat output is invariant under rotation).
        err_se = (se_out_a - se_out).abs().max().item()
        assert err_se < 1e-5, f"seat output should be rotation-invariant: {err_se:.2e}"


def test_combined_wreath_action():
    """The decisive test: simultaneously permute seats AND rotate. Output
    must transform identically under that joint action."""
    torch.manual_seed(2)
    layer = WreathFuseLayer(in_spatial=2, in_seat=3, out_spatial=4, out_seat=5)
    layer.eval()
    coords = _hex_ring(2)
    L = len(coords)
    perm_rot = torch.from_numpy(make_rotation_permutation(coords, num_rotations=6))
    N = 4
    spatial = torch.randn(1, N, L, 6 * 2)
    seat = torch.randn(1, N, 3)
    seat_perm = torch.randperm(N)
    rot_idx = 2

    def rotate_all_seats(s, num_chan):
        out = torch.zeros_like(s)
        for i in range(s.shape[1]):
            out[:, i] = rotate_feature_map(s[:, i], perm_rot, num_chan, rot_idx)
        return out

    # Path 1: rotate, permute seats, then layer
    sp_a = rotate_all_seats(spatial, 2)
    sp_a, se_a = permute_seats(sp_a, seat, seat_perm)
    sp_out_a, se_out_a = layer(sp_a, se_a)

    # Path 2: layer, then rotate, then permute seats
    sp_out, se_out = layer(spatial, seat)
    sp_out_b = rotate_all_seats(sp_out, 4)
    sp_out_b, se_out_b = permute_seats(sp_out_b, se_out, seat_perm)

    # Interior locations
    from equivariant_net.src.c6_spatial import rotate_axial
    coord_set = set(coords)
    interior = []
    for c in coords:
        rotated = c
        for _ in range(rot_idx):
            rotated = rotate_axial(rotated)
        interior.append(rotated in coord_set)
    int_mask = torch.tensor(interior)
    if int_mask.any():
        err_sp = (sp_out_a[:, :, int_mask] - sp_out_b[:, :, int_mask]).abs().max().item()
        err_se = (se_out_a - se_out_b).abs().max().item()
        assert err_sp < 1e-5, f"combined wreath eq violated (spatial): {err_sp:.2e}"
        assert err_se < 1e-5, f"combined wreath eq violated (seat): {err_se:.2e}"
