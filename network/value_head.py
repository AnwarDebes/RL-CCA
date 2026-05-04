"""Value head: single scalar prediction in [-1, 1].

Regresses to the normalized teacher final_score. Replaces the previous
4-component decomposition which had ill-formed targets (terminal-state
quantities applied to all trajectory entries; see plan v2 § Strategic
Decisions).
"""

import torch
import torch.nn as nn

from config import Config


class ValueHead(nn.Module):
    def __init__(self, input_dim: int = Config.HIDDEN_DIM):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, input_dim] -> [B] with tanh in [-1, 1]."""
        h = torch.relu(self.fc1(x))
        return torch.tanh(self.fc2(h)).squeeze(-1)
