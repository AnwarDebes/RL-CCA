"""S_N seat-permutation-equivariant layer for the wreath equivariant net.

This is the seat-permutation factor of the wreath product G_N ≀ S_N. It
operates on per-seat feature vectors (seat features = features specific
to each player's role/identity, e.g. their pieces' positions, score,
remaining time). Permuting which physical triangle is "player 1" must
permute the corresponding seat features identically - the policy and
value outputs must transform consistently.

We implement this as a Deep-Sets / Set-Transformer style layer: each
seat feature is processed by a *shared* transformation, then aggregated
across seats by a permutation-invariant pooling (or kept seat-wise for
equivariant outputs). This guarantees S_N-equivariance by construction.

Concretely, given seat features X ∈ ℝ^{N × d}:

    - "Equivariant" output: Y ∈ ℝ^{N × d'} where Y[i] = f(X[i], pool_{j ≠ i} g(X[j]))
      Permuting rows of X permutes rows of Y identically.
    - "Invariant" output: y ∈ ℝ^{d'} where y = pool_i h(X[i])
      Output is unchanged under any seat permutation.

The implementation is pure PyTorch (no escnn dependency) so it runs on
CPU and is easy to test for bit-identical equivariance.

This is one factor of the full wreath; the spatial p6m factor (board
rotation/reflection) is a separate layer that operates on the hex-grid
feature map. The wreath-fuse layer (in wreath_fuse.py) combines them.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class SeatInvariantPool(nn.Module):
    """Mean-pooling across seats for S_N-invariant outputs.

    Input:  X of shape (..., N, d)
    Output: y of shape (..., d)
    """

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        if mask is None:
            return x.mean(dim=-2)
        # mask is (..., N) bool; mean only over True entries
        m = mask.unsqueeze(-1).to(x.dtype)
        s = (x * m).sum(dim=-2)
        n = m.sum(dim=-2).clamp(min=1.0)
        return s / n


class SeatEquivariantBlock(nn.Module):
    """One DeepSets-style equivariant block on per-seat features.

    Update rule:
        Y[i] = sigma( W_self · X[i] + W_pool · pool_{j} X[j] )

    where pool is a seat-symmetric reduction (mean here). This is
    permutation-equivariant by construction: permuting the rows of X
    permutes the rows of Y identically.

    For the active-subset variant (when only some seats are occupied,
    e.g. N=3 in a 6-seat-template game), pass a `mask` of shape (..., N)
    indicating which rows are valid.
    """

    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.W_self = nn.Linear(in_dim, out_dim)
        self.W_pool = nn.Linear(in_dim, out_dim)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        # x: (..., N, d_in)
        if mask is None:
            pooled = x.mean(dim=-2, keepdim=True)
        else:
            m = mask.unsqueeze(-1).to(x.dtype)
            pooled = (x * m).sum(dim=-2, keepdim=True) / m.sum(dim=-2, keepdim=True).clamp(min=1.0)
        out = self.W_self(x) + self.W_pool(pooled.expand_as(x))
        return self.act(out)


class WreathSeatNet(nn.Module):
    """Stack of equivariant blocks producing per-seat features that are
    S_N-equivariant in the seat dimension.
    """

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, num_blocks: int = 2) -> None:
        super().__init__()
        layers = []
        d = in_dim
        for _ in range(num_blocks):
            layers.append(SeatEquivariantBlock(d, hidden_dim))
            d = hidden_dim
        layers.append(SeatEquivariantBlock(d, out_dim))
        self.blocks = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        h = x
        for block in self.blocks:
            h = block(h, mask=mask)
        return h


# ----------------------------------------------------------------------
# Active-seat masking utility for star-hex games where N ∈ {2,3,4,6}.
# The "subgroup gating" of ASEN style is realised by which rows of X are
# masked active; the network adapts automatically because the equivariant
# block's pool is mask-aware.
# ----------------------------------------------------------------------


def make_seat_mask(num_active: int, max_seats: int = 6) -> torch.Tensor:
    """Boolean mask over max_seats slots; first num_active are True.

    For the wreath equivariance to be exact, mask must be over a
    contiguous prefix; if seats are not contiguous in the data layout,
    permute first.
    """
    return torch.arange(max_seats) < num_active
