"""NexusNetV4 - bigger NBT backbone + 6 heads.

Forward(state, legal_mask, current_seat) returns dict:
    policy:        [B, 1210]
    logits:        [B, 1210]
    value:         [B]              current-player slot of value_vec (back-compat)
    value_vec:     [B, 6]
    opp_logits:    [B, 1210]
    plies:         [B]
    score_margin:  [B, 6]
    pin_final:     [B, NUM_PIECES, K]
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn

from config import Config
from core.board import HexBoard
from network.restnet_v4 import ResTNetBackboneV4
from network.heads_v4 import (
    PolicyHeadV4, ValueVectorHeadV4, OppPolicyHeadV4, PliesHeadV4,
    ScoreMarginHeadV4, PinFinalHeadV4,
)


MAX_PLAYERS = Config.MAX_PLAYERS


class NexusNetV4(nn.Module):
    def __init__(self, board: Optional[HexBoard] = None):
        super().__init__()
        if board is None:
            board = HexBoard()
        self._board = board
        positions = board.get_valid_grid_positions()

        self.backbone = ResTNetBackboneV4(positions)
        H = Config.HIDDEN_DIM_V4

        self.policy_head = PolicyHeadV4(H)
        self.value_head = ValueVectorHeadV4(H)
        self.opp_policy_head = OppPolicyHeadV4(H)
        self.plies_head = PliesHeadV4(H)
        self.score_margin_head = ScoreMarginHeadV4(H)
        self.pin_final_head = PinFinalHeadV4(H)

    def forward(
        self,
        state: torch.Tensor,
        legal_mask: torch.Tensor,
        current_seat: Optional[torch.Tensor] = None,
        opp_action: Optional[torch.Tensor] = None,    # ignored (compat)
        opp_hidden: Optional[torch.Tensor] = None,    # ignored (compat)
    ) -> Dict[str, torch.Tensor]:
        repr_ = self.backbone(state)

        policy = self.policy_head(repr_, legal_mask)
        logits = self.policy_head.forward_logits(repr_, legal_mask)

        value_vec = self.value_head(repr_)
        if current_seat is None:
            seat_idx = torch.zeros(value_vec.size(0), dtype=torch.long,
                                   device=value_vec.device)
        else:
            seat_idx = current_seat.long().clamp(0, MAX_PLAYERS - 1)
        value = value_vec.gather(1, seat_idx.unsqueeze(1)).squeeze(1)

        return {
            "policy": policy,
            "logits": logits,
            "value": value,
            "value_vec": value_vec,
            "opp_logits": self.opp_policy_head(repr_),
            "plies": self.plies_head(repr_),
            "score_margin": self.score_margin_head(repr_),
            "pin_final": self.pin_final_head(repr_),
        }

    def get_representation(self, state: torch.Tensor) -> torch.Tensor:
        return self.backbone(state)

    def aggregate_value(self, value: torch.Tensor) -> torch.Tensor:
        if value.dim() == 2 and value.size(-1) == 1:
            return value.squeeze(-1)
        return value

    @staticmethod
    def load(path: str, device: str = "cuda") -> "NexusNetV4":
        dev = torch.device(device if torch.cuda.is_available() else "cpu")
        model = NexusNetV4()
        sd = torch.load(path, map_location=dev, weights_only=True)
        model.load_state_dict(sd)
        model.to(dev)
        return model

    def save(self, path: str):
        torch.save(self.state_dict(), path)
