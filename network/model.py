"""NexusNet: ResTNet backbone + policy head + single-scalar value head.

Removed in this rewrite (vs. the 2-player version):
- OpponentGRU (was a training/inference mismatch - inference passed None anyway)
- AuxHead y, z (DiscoRL - not used)
- 4-component value head (replaced with single scalar)

Forward:
  state [B, 22, 17, 17] -> backbone [B, 128] -> {policy [B, 1210], value [B]}
"""

from typing import Dict, Optional

import torch
import torch.nn as nn

from config import Config
from core.board import HexBoard
from network.restnet import ResTNetBackbone
from network.policy_head import PolicyHead
from network.value_head import ValueHead


class NexusNet(nn.Module):
    """NEXUS v3 - N-player aware (via the encoder), single-scalar value head."""

    def __init__(self, board: Optional[HexBoard] = None):
        super().__init__()
        if board is None:
            board = HexBoard()
        self._board = board

        valid_positions = board.get_valid_grid_positions()

        self.backbone = ResTNetBackbone(valid_positions)
        self.policy_head = PolicyHead()
        self.value_head = ValueHead()

    def forward(
        self,
        state: torch.Tensor,
        legal_mask: torch.Tensor,
        opp_action: Optional[torch.Tensor] = None,   # ignored - kept for API compat
        opp_hidden: Optional[torch.Tensor] = None,   # ignored - kept for API compat
    ) -> Dict[str, torch.Tensor]:
        """Full forward pass.

        Args:
            state:      [B, 22, 17, 17] float tensor.
            legal_mask: [B, 1210] bool tensor.

        Returns dict with keys:
            policy: [B, 1210] softmax distribution masked to legal moves.
            logits: [B, 1210] raw masked logits (illegal = -inf).
            value:  [B] scalar in [-1, 1].
        """
        board_repr = self.backbone(state)              # [B, 128]
        policy = self.policy_head(board_repr, legal_mask)
        logits = self.policy_head.forward_logits(board_repr, legal_mask)
        value = self.value_head(board_repr)            # [B]
        return {
            "policy": policy,
            "logits": logits,
            "value": value,
        }

    def get_representation(self, state: torch.Tensor) -> torch.Tensor:
        """Get backbone representation only (e.g., for consistency loss)."""
        return self.backbone(state)

    def aggregate_value(self, value: torch.Tensor) -> torch.Tensor:
        """Single-scalar value - pass-through. Kept for back-compat with MCTS."""
        if value.dim() == 2:
            return value.squeeze(-1)
        return value

    @staticmethod
    def load(path: str, device: str = "cuda") -> "NexusNet":
        dev = torch.device(device if torch.cuda.is_available() else "cpu")
        model = NexusNet()
        state_dict = torch.load(path, map_location=dev, weights_only=True)
        model.load_state_dict(state_dict)
        model.to(dev)
        return model

    def save(self, path: str):
        torch.save(self.state_dict(), path)
