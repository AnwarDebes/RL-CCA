"""NexusNetV3 - bigger backbone (SE), per-player value vector, aux heads.

Forward(state, legal_mask) returns a dict with keys:
    policy:     [B, 1210]   softmax over legal moves
    logits:     [B, 1210]   raw masked logits (illegal = -inf)
    value:      [B]         current-player slot of value_vec (back-compat name)
    value_vec:  [B, 6]      per-player value (tanh, [-1, 1])
    opp_logits: [B, 1210]   opponent-next-move prediction (raw logits)
    plies:      [B]         plies-remaining regression

For MCTS / inference, callers can ignore aux outputs - `value` is the scalar
they need. For training, the loss uses all four heads.

To keep `value` correctly populated, callers MUST pass `current_seat` (a
[B] long tensor in [0, 5]) when they want per-batch slot extraction. If
None, slot 0 is used (fine for single-state inference where the state is
always rotated/canonicalized to the current player's perspective - which
NEXUS already does via `state_encoder`).
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn

from config import Config
from core.board import HexBoard
from network.restnet_v3 import ResTNetBackboneV3
from network.heads_v3 import (
    PolicyHeadV3, ValueVectorHead, OppPolicyHead, PliesHead,
)


MAX_PLAYERS = 6


class NexusNetV3(nn.Module):
    """v3 model: backbone (SE+Trans) + 4 heads."""

    def __init__(self, board: Optional[HexBoard] = None):
        super().__init__()
        if board is None:
            board = HexBoard()
        self._board = board
        positions = board.get_valid_grid_positions()

        self.backbone = ResTNetBackboneV3(positions)
        H = Config.HIDDEN_DIM_V3

        self.policy_head = PolicyHeadV3(H)
        self.value_head = ValueVectorHead(H, max_players=MAX_PLAYERS)
        self.opp_policy_head = OppPolicyHead(H)
        self.plies_head = PliesHead(H)

    def forward(
        self,
        state: torch.Tensor,
        legal_mask: torch.Tensor,
        current_seat: Optional[torch.Tensor] = None,
        opp_action: Optional[torch.Tensor] = None,    # ignored - for v2 compat
        opp_hidden: Optional[torch.Tensor] = None,    # ignored - for v2 compat
    ) -> Dict[str, torch.Tensor]:
        repr_ = self.backbone(state)                  # [B, H]

        policy = self.policy_head(repr_, legal_mask)
        logits = self.policy_head.forward_logits(repr_, legal_mask)

        value_vec = self.value_head(repr_)            # [B, MAX_PLAYERS]
        if current_seat is None:
            seat_idx = torch.zeros(value_vec.size(0), dtype=torch.long,
                                   device=value_vec.device)
        else:
            seat_idx = current_seat.long().clamp_(0, MAX_PLAYERS - 1)
        value = value_vec.gather(1, seat_idx.unsqueeze(1)).squeeze(1)  # [B]

        opp_logits = self.opp_policy_head(repr_)      # [B, 1210]
        plies = self.plies_head(repr_)                # [B]

        return {
            "policy": policy,
            "logits": logits,
            "value": value,
            "value_vec": value_vec,
            "opp_logits": opp_logits,
            "plies": plies,
        }

    def get_representation(self, state: torch.Tensor) -> torch.Tensor:
        return self.backbone(state)

    def aggregate_value(self, value: torch.Tensor) -> torch.Tensor:
        if value.dim() == 2 and value.size(-1) == 1:
            return value.squeeze(-1)
        return value

    @staticmethod
    def load(path: str, device: str = "cuda") -> "NexusNetV3":
        dev = torch.device(device if torch.cuda.is_available() else "cpu")
        model = NexusNetV3()
        state_dict = torch.load(path, map_location=dev, weights_only=True)
        model.load_state_dict(state_dict)
        model.to(dev)
        return model

    def save(self, path: str):
        torch.save(self.state_dict(), path)
