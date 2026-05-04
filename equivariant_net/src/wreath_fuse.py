"""Wreath-equivariant fusion layer: combines S_N seat factor with C6 spatial factor.

The wreath product G ≀ S_N (here: C6 ≀ S_N) acts on a structure with N
copies of a spatial substrate, one per seat. C6 acts on each copy
independently; S_N permutes the copies.

A layer is **wreath-equivariant** when these two operations commute with
the layer's forward pass:

    L(rotate_seat_k(x))  =  rotate_seat_k(L(x))   for any k, any rotation
    L(perm_seats(x))     =  perm_seats(L(x))      for any seat permutation

We achieve this by:
  1. Applying a C6-equivariant layer per-seat (shared weights across seats
     ⇒ S_N-equivariant by parameter sharing).
  2. Pooling spatial features per-seat (mean over locations) to produce
     per-seat feature vectors.
  3. Applying a SeatEquivariantBlock to the per-seat features.
  4. Broadcasting per-seat features back to the spatial stream by adding
     a seat-conditioned bias to each location.

Each operation is independently equivariant under both factors, so the
composed layer is wreath-equivariant.

Public API
----------
    WreathFuseLayer.forward(spatial, seat) -> (spatial', seat')
        spatial: (B, N, L, 6 * C_spatial)
        seat:    (B, N, C_seat)
        Output dims may differ.

Tests verify bit-identical outputs under:
    - any seat permutation,
    - any C6 rotation applied to all seats simultaneously,
    - both compositionally (the wreath action).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .c6_spatial import C6EquivariantLinear
from .seat_equivariant import SeatEquivariantBlock


class WreathFuseLayer(nn.Module):
    """Wreath-equivariant fusion of spatial × seat features.

    Args:
        in_spatial: per-rotation-channel size (so spatial input has shape
            (B, N, L, 6 * in_spatial)).
        in_seat: seat feature size.
        out_spatial: per-rotation-channel output size.
        out_seat: seat feature output size.
    """

    def __init__(
        self,
        in_spatial: int,
        in_seat: int,
        out_spatial: int,
        out_seat: int,
    ) -> None:
        super().__init__()
        self.in_spatial = in_spatial
        self.in_seat = in_seat
        self.out_spatial = out_spatial
        self.out_seat = out_seat

        # Per-seat C6-equivariant linear (shared across seats by reuse).
        self.spatial_layer = C6EquivariantLinear(in_spatial, out_spatial)

        # Pool spatial -> seat: project the per-rotation-channel-sized vector
        # (averaged over locations and over the 6 rotation channels for an
        # invariant representation) to a per-seat extra feature.
        self.spatial_to_seat = nn.Linear(in_spatial, out_seat)

        # S_N-equivariant block on (concat(seat, pooled_spatial)).
        self.seat_block = SeatEquivariantBlock(in_seat + out_seat, out_seat)

        # Seat -> spatial broadcast. The bias must be invariant under the
        # cyclic-shift of the 6 rotation channels, otherwise the layer
        # breaks C6 equivariance. So we project to out_spatial (C-dim
        # vector) and broadcast that vector identically to all 6
        # rotation channels.
        self.seat_to_spatial = nn.Linear(out_seat, out_spatial)

    def forward(
        self, spatial: torch.Tensor, seat: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # spatial: (B, N, L, 6 * C_in)
        # seat:    (B, N, S_in)
        B, N, L, _ = spatial.shape
        # 1. Apply C6-equivariant layer to each seat's spatial map.
        # We collapse (B, N) to a single batch dim for the layer.
        sp_in = spatial.view(B * N, L, 6 * self.in_spatial)
        sp_out = self.spatial_layer(sp_in)  # (B*N, L, 6 * out_spatial)
        sp_out = sp_out.view(B, N, L, 6 * self.out_spatial)

        # 2. Pool spatial -> per-seat: average over (locations, rotation channels)
        sp_pooled = spatial.view(B, N, L, 6, self.in_spatial).mean(dim=(2, 3))
        # (B, N, in_spatial) -> (B, N, out_seat)
        sp_pooled_proj = self.spatial_to_seat(sp_pooled)

        # 3. Combine seat + pooled-spatial, apply seat-equivariant block.
        seat_in = torch.cat([seat, sp_pooled_proj], dim=-1)  # (B, N, in_seat + out_seat)
        seat_out = self.seat_block(seat_in)  # (B, N, out_seat)

        # 4. Broadcast seat -> spatial. The C-dim bias is replicated
        # identically across all 6 rotation channels and all locations,
        # which preserves both C6 cyclic-shift equivariance and S_N
        # seat-permutation equivariance.
        seat_bias_c = self.seat_to_spatial(seat_out)  # (B, N, out_spatial)
        # Expand to (B, N, 1, 6 * out_spatial) by repeating C-dim 6 times.
        seat_bias_expanded = seat_bias_c.unsqueeze(-2).unsqueeze(-2)  # (B, N, 1, 1, out_spatial)
        seat_bias_expanded = seat_bias_expanded.expand(
            -1, -1, sp_out.shape[2], 6, -1
        )  # (B, N, L, 6, out_spatial)
        seat_bias_expanded = seat_bias_expanded.reshape(
            sp_out.shape[0], sp_out.shape[1], sp_out.shape[2], 6 * self.out_spatial
        )
        sp_out = sp_out + seat_bias_expanded

        return sp_out, seat_out


def permute_seats(
    spatial: torch.Tensor,
    seat: torch.Tensor,
    perm: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Permute seats jointly across spatial and seat tensors."""
    return spatial[:, perm], seat[:, perm]
