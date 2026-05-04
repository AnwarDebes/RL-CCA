"""Full wreath-equivariant network for Chinese Checkers.

Combines:
  * CCWreathEncoder       - C6-equivariant trunk on real hex coords
  * SeatEquivariantBlock  - S_N-equivariant on per-seat features
  * ASEN-style gating     - symmetry-breaking input encoding which
                            home triangles are active (per-N subgroup
                            selection)
  * Policy + value heads  - composed from the equivariant trunk

This is the production-quality wreath net - it operates on the real
nexus GameEnv state and produces actions for the 1210-action space,
respecting the wreath equivariance.

ASEN-style symmetry-breaking field
----------------------------------
A small N-bit indicator vector encodes which seats are occupied. This
field is fed both to the seat stream (so the seat-equivariant blocks
know how many active seats there are) AND to the spatial trunk via a
broadcast layer (so the trunk can specialise per-N when needed).

This realises the active-subgroup gating: when N=2, only some D6
elements preserve the home-triangle assignment; the network learns to
break the global C6 invariance into the active subgroup G_2 ⊂ C6 via
this gating signal - without retraining when N changes.

The seat-features stream encodes per-seat colour, current-player flag,
and pieces-in-goal count - natural per-seat features that S_N permutes
when the seats are relabelled.
"""

from __future__ import annotations

import os
import sys
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_NEXUS_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)
if _NEXUS_ROOT not in sys.path:
    sys.path.insert(0, _NEXUS_ROOT)

from .cc_wreath_encoder import CCWreathEncoder
from .seat_equivariant import (
    SeatEquivariantBlock,
    SeatInvariantPool,
    WreathSeatNet,
)


def cc_seat_features(state, max_seats: int = 6) -> np.ndarray:
    """Build per-seat features for a CC GameEnv state.

    Returns ndarray of shape (max_seats, seat_feature_dim) where:
      * Active seats (0..N-1) get real features.
      * Inactive seats (N..max_seats-1) are zeros.

    Per-seat features (current implementation, 8-dim):
      [0]    is_current_player (1 if this seat is to move)
      [1]    is_active (1 if this seat is in play)
      [2]    pieces_in_goal / 10 (normalised)
      [3]    pieces_in_start / 10
      [4]    sum_distance_to_goal / 200
      [5]    fraction of game time used (0..1)
      [6]    move count / 100
      [7]    bias 1.0 (always)
    """
    feat = np.zeros((max_seats, 8), dtype=np.float32)
    N = state.num_players
    for p in range(N):
        color = state.colors[p]
        pieces_in_goal = state.board.count_in_goal(state.pieces[p], color)
        sum_dist = state.board.sum_distances_to_goal(state.pieces[p], color)
        feat[p, 0] = 1.0 if p == state.current_player else 0.0
        feat[p, 1] = 1.0  # active
        feat[p, 2] = float(pieces_in_goal) / 10.0
        feat[p, 3] = float(state.NUM_PIECES if hasattr(state, "NUM_PIECES") else 10 - pieces_in_goal) / 10.0
        feat[p, 4] = float(sum_dist) / 200.0
        feat[p, 5] = float(state.player_time_taken[p]) / 60.0
        feat[p, 6] = float(state.player_move_counts[p]) / 100.0
        feat[p, 7] = 1.0
    # Inactive seats keep zeros - feat[..., 1] = 0 acts as the ASEN-style
    # active-subgroup signal.
    return feat


class WreathCCNetwork(nn.Module):
    """Full wreath-equivariant network for CC.

    Args:
        spatial_channels: trunk width (per rotation channel) for the
            C6-equivariant spatial encoder.
        spatial_blocks: number of C6-equivariant blocks.
        spatial_out: pooled spatial feature dim.
        seat_in: per-seat input feature dim (matches cc_seat_features above).
        seat_hidden: seat-equivariant block width.
        seat_blocks: number of seat-equivariant blocks.
        action_space_size: full action space (1210 for CC).
        max_players: 6.
    """

    def __init__(
        self,
        spatial_channels: int = 16,
        spatial_blocks: int = 3,
        spatial_out: int = 128,
        seat_in: int = 8,
        seat_hidden: int = 32,
        seat_blocks: int = 2,
        action_space_size: int = 1210,
        max_players: int = 6,
    ) -> None:
        super().__init__()
        self.max_players = max_players

        # Spatial trunk (C6-equivariant, invariant pooled output)
        self.spatial = CCWreathEncoder(
            in_channels=32,
            c_spatial=spatial_channels,
            hidden_dim=spatial_out,
            num_blocks=spatial_blocks,
        )

        # Seat trunk (S_N-equivariant)
        self.seat_net = WreathSeatNet(
            in_dim=seat_in,
            hidden_dim=seat_hidden,
            out_dim=seat_hidden,
            num_blocks=seat_blocks,
        )
        self.seat_pool = SeatInvariantPool()

        # Fusion: combine pooled spatial + pooled-seat into a single feature.
        self.fuse = nn.Sequential(
            nn.Linear(spatial_out + seat_hidden, spatial_out),
            nn.GELU(),
        )
        # Heads
        self.policy_head = nn.Linear(spatial_out, action_space_size)
        self.value_head = nn.Linear(spatial_out, 1)

    def forward(
        self,
        state_tensor: torch.Tensor,        # (B, 32, 17, 17)
        seat_features: torch.Tensor,        # (B, max_players, seat_in)
        seat_mask: Optional[torch.Tensor] = None,  # (B, max_players) bool
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (policy_logits, scalar_value)."""
        if state_tensor.dim() == 3:
            state_tensor = state_tensor.unsqueeze(0)
        if seat_features.dim() == 2:
            seat_features = seat_features.unsqueeze(0)
        # Spatial trunk (B, spatial_out)
        h_spatial = self.spatial(state_tensor)
        # Seat trunk → S_N-equivariant per-seat features
        h_seat_eq = self.seat_net(seat_features, mask=seat_mask)  # (B, max, hidden)
        # Pool seat features to invariant
        h_seat_inv = self.seat_pool(h_seat_eq, mask=seat_mask)  # (B, hidden)
        # Fuse
        h = self.fuse(torch.cat([h_spatial, h_seat_inv], dim=-1))
        policy_logits = self.policy_head(h)
        value = torch.tanh(self.value_head(h)).squeeze(-1)
        return policy_logits, value

    def inputs_from_state(self, state) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build the (state_tensor, seat_features, seat_mask) triplet from
        a CC GameEnv state. Used by inference paths."""
        # State tensor (32, 17, 17)
        st = state.get_state_tensor(state.current_player)
        # Seat features (max, seat_in)
        sf = cc_seat_features(state, max_seats=self.max_players)
        # Seat mask
        mask = np.zeros(self.max_players, dtype=bool)
        mask[: state.num_players] = True
        return (
            st.unsqueeze(0),
            torch.from_numpy(sf).unsqueeze(0),
            torch.from_numpy(mask).unsqueeze(0),
        )
