"""v3 heads: policy + per-player value vector + KataGo-style aux heads.

Heads:
- policy: same shape as v2 (1210), masked softmax.
- value_vec: [B, MAX_PLAYERS=6] regression on per-seat normalized final_score
  in [-1, 1]. The "aggregated value" used by MCTS is the current player's slot
  (extracted via a per-batch index).
- opp_policy: [B, 1210] predict the next opponent's move (KataGo aux, w=0.15).
- plies: [B] regression on plies-remaining (game-length pred).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import Config


class PolicyHeadV3(nn.Module):
    def __init__(self, hidden: int):
        super().__init__()
        self.fc = nn.Linear(hidden, Config.ACTION_SPACE)

    def forward(self, x: torch.Tensor, legal_mask: torch.Tensor) -> torch.Tensor:
        logits = self.fc(x).masked_fill(~legal_mask, float("-inf"))
        return F.softmax(logits, dim=-1)

    def forward_logits(self, x: torch.Tensor, legal_mask: torch.Tensor) -> torch.Tensor:
        return self.fc(x).masked_fill(~legal_mask, float("-inf"))


class ValueVectorHead(nn.Module):
    """Per-player value head - outputs a vector of length MAX_PLAYERS.

    For N<MAX_PLAYERS, only the first N entries are meaningful. The loss uses
    a length-N mask so unused slots get zero gradient.
    """

    def __init__(self, hidden: int, max_players: int = 6):
        super().__init__()
        self.max_players = max_players
        self.fc1 = nn.Linear(hidden, 256)
        self.fc2 = nn.Linear(256, max_players)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.fc1(x))
        return torch.tanh(self.fc2(h))                 # [B, MAX_PLAYERS]


class OppPolicyHead(nn.Module):
    """Predict the next opponent's chosen action (after the current player moves)."""

    def __init__(self, hidden: int):
        super().__init__()
        self.fc = nn.Linear(hidden, Config.ACTION_SPACE)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)                              # raw logits, masked at loss time


class PliesHead(nn.Module):
    """Regress plies-remaining-until-terminal (normalized by /200)."""

    def __init__(self, hidden: int):
        super().__init__()
        self.fc1 = nn.Linear(hidden, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.fc1(x))
        return self.fc2(h).squeeze(-1)                 # [B]
