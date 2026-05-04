"""Policy head: fused representation -> 1210 action logits with legal masking."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import Config


class PolicyHead(nn.Module):
    def __init__(self, input_dim: int = Config.HIDDEN_DIM):
        super().__init__()
        self.fc = nn.Linear(input_dim, Config.ACTION_SPACE)

    def forward(
        self, x: torch.Tensor, legal_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            x: [B, input_dim] fused representation.
            legal_mask: [B, 1210] boolean, True = legal.

        Returns:
            policy: [B, 1210] probability distribution (after masking + softmax).
        """
        logits = self.fc(x)
        # Mask illegal to -inf BEFORE softmax
        logits = logits.masked_fill(~legal_mask, float('-inf'))
        policy = F.softmax(logits, dim=-1)
        return policy

    def forward_logits(
        self, x: torch.Tensor, legal_mask: torch.Tensor
    ) -> torch.Tensor:
        """Return masked logits (before softmax) for loss computation."""
        logits = self.fc(x)
        logits = logits.masked_fill(~legal_mask, float('-inf'))
        return logits
