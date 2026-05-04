"""Auxiliary y and z heads for DiscoRL experiment."""

import torch
import torch.nn as nn

from config import Config


class AuxHead(nn.Module):
    """Simple FC -> GELU -> FC head for y or z predictions."""

    def __init__(self, input_dim: int = Config.HIDDEN_DIM, output_dim: int = Config.AUX_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.GELU(),
            nn.Linear(output_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
