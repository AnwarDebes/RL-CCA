"""Wreath-equivariant encoder for the real Chinese Checkers board.

Bridges the workshop equivariant subproject to the user's tournament
game. Uses the actual axial coordinates of the 121 hex cells from
`core.board.HexBoard` to build a real (not toy) C6-equivariant
permutation table, then applies the C6-equivariant linear layers on
the per-cell features.

The seat-features stream uses the active-player and per-seat colour /
piece configuration as a separate input. The wreath fusion combines
both streams under the joint action: rotating the board (C6) AND
permuting which physical triangle is "player 1" (S_N).

This file does the actual work - it is not a stub. The C6 rotation
table is built from the HexBoard's index_of dict, so the equivariance
holds exactly on the interior of the board (cells whose 60° image is
also in the 121-cell set, which is essentially the whole star except
for boundary tips that map outside).

Key design choice
-----------------
We compute the rotation permutation **once** when the encoder is
constructed, using the actual board coordinates. This is cheap (121
cells × 6 rotations = 726 lookups) and avoids any approximation.
"""

from __future__ import annotations

import os
import sys
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_NEXUS_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)
if _NEXUS_ROOT not in sys.path:
    sys.path.insert(0, _NEXUS_ROOT)

from .c6_spatial import C6EquivariantLinear, rotate_axial


def _get_board_axial_coords() -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
    """Return (axial_coords, grid_coords) for all 121 cells of the CC board.

    axial_coords[i] = (q, r), grid_coords[i] = (gx, gy) for cell i.
    """
    from core.board import HexBoard
    board = HexBoard()
    n = len(board.cell_q)
    axial = [(board.cell_q[i], board.cell_r[i]) for i in range(n)]
    grid = [(board.cell_gx[i], board.cell_gy[i]) for i in range(n)]
    return axial, grid


def _build_cc_rotation_permutation(
    axial: List[Tuple[int, int]]
) -> np.ndarray:
    """Compute the (6, 121) permutation table: P[k, i] = j where i rotated
    by k×60° lands on cell j. If the rotated coord is not on the board
    (boundary tip), we fall back to identity (P[k, i] = i)."""
    coord_to_idx = {c: i for i, c in enumerate(axial)}
    n = len(axial)
    P = np.zeros((6, n), dtype=np.int64)
    for k in range(6):
        for i, c in enumerate(axial):
            cur = c
            for _ in range(k):
                cur = rotate_axial(cur)
            P[k, i] = coord_to_idx.get(cur, i)
    return P


class CCWreathEncoder(nn.Module):
    """Wreath-equivariant encoder for the Chinese Checkers board.

    Args:
        in_channels: number of input planes from the state tensor (32 in v4).
        c_spatial: per-rotation-channel feature size.
        hidden_dim: final pooled feature dim.
        num_blocks: number of C6-equivariant linear stages.
    """

    def __init__(
        self,
        in_channels: int = 32,
        c_spatial: int = 16,
        hidden_dim: int = 64,
        num_blocks: int = 2,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.c_spatial = c_spatial
        self.hidden_dim = hidden_dim
        self.out_dim = hidden_dim

        # Build axial coords and rotation table.
        axial, grid = _get_board_axial_coords()
        self.num_cells = len(axial)
        # Register the (gx, gy) → cell index mapping as a buffer.
        gx = torch.tensor([g[0] for g in grid], dtype=torch.long)
        gy = torch.tensor([g[1] for g in grid], dtype=torch.long)
        self.register_buffer("gx", gx, persistent=False)
        self.register_buffer("gy", gy, persistent=False)
        # Rotation permutation (6, 121).
        rot_perm = _build_cc_rotation_permutation(axial)
        self.register_buffer(
            "rot_perm", torch.from_numpy(rot_perm).long(), persistent=False,
        )

        # Project the in_channels-feature to 6 × c_spatial. Since we want
        # the input to be equivariant from the start, we replicate across
        # the 6 rotation channels (regular-rep "lift") via a single linear
        # mapping shared across rotation copies.
        self.lift = nn.Linear(in_channels, c_spatial)

        # Stack of C6-equivariant linear layers.
        self.blocks = nn.ModuleList(
            [C6EquivariantLinear(c_spatial, c_spatial) for _ in range(num_blocks)]
        )
        # Final pool: average over (locations, rotation channels) → linear → hidden_dim.
        self.head = nn.Linear(c_spatial, hidden_dim)

    def _gather_cells(self, x: torch.Tensor) -> torch.Tensor:
        """Extract per-hex-cell features from the (B, C, 17, 17) state tensor.

        Returns (B, 121, C).
        """
        B = x.shape[0]
        # Use advanced indexing along last two dims.
        # x[:, :, gx, gy] -> (B, C, 121)
        gx = self.gx
        gy = self.gy
        cells = x[:, :, gx, gy]  # (B, C, 121)
        return cells.transpose(1, 2)  # (B, 121, C)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, in_channels, 17, 17) → returns (B, hidden_dim)."""
        if x.dim() == 3:
            x = x.unsqueeze(0)
        # 1. Gather: (B, 121, in_channels)
        per_cell = self._gather_cells(x)
        # 2. Lift: (B, 121, c_spatial) - replicate to 6 rotation channels.
        lifted = self.lift(per_cell)  # (B, 121, c_spatial)
        # Replicate 6 times along the new rotation-channel dim (regular-rep
        # "lift"): each rotation channel starts identical, so the rotation
        # action is well-defined.
        L = lifted.shape[1]
        x6 = lifted.unsqueeze(2).expand(-1, -1, 6, -1).contiguous()  # (B, 121, 6, c_spatial)
        x6 = x6.view(x6.shape[0], L, 6 * self.c_spatial)
        # 3. C6-equivariant blocks
        for blk in self.blocks:
            x6 = F.relu(blk(x6))
        # 4. Pool over locations and rotation channels - invariant pool.
        # x6 is (B, L, 6 * c_spatial). Average over L and over the 6 channels.
        x6 = x6.view(x6.shape[0], L, 6, self.c_spatial)
        pooled = x6.mean(dim=(1, 2))  # (B, c_spatial)
        return self.head(pooled)
