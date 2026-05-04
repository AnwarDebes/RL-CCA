"""v4 heads: policy + per-player value vector + KataGo-style aux heads.

Heads:
- policy:        [B, 1210]
- value_vec:     [B, 6]
- opp_logits:    [B, 1210]   predict next opponent's move
- plies:         [B]         regression on plies-remaining/200
- score_margin:  [B, 6]      regression on per-seat (score - mean(others)) / 1300
- pin_final:     [B, NUM_PIECES, K]   per-pin distance-bucket prediction at game end
                              (own pieces only - KataGo ownership analog)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import Config


MAX_PLAYERS = Config.MAX_PLAYERS


class PolicyHeadV4(nn.Module):
    def __init__(self, hidden: int):
        super().__init__()
        self.fc = nn.Linear(hidden, Config.ACTION_SPACE)

    def forward(self, x: torch.Tensor, legal_mask: torch.Tensor) -> torch.Tensor:
        logits = self.fc(x).masked_fill(~legal_mask, float("-inf"))
        return F.softmax(logits, dim=-1)

    def forward_logits(self, x: torch.Tensor, legal_mask: torch.Tensor) -> torch.Tensor:
        return self.fc(x).masked_fill(~legal_mask, float("-inf"))


class ValueVectorHeadV4(nn.Module):
    def __init__(self, hidden: int, max_players: int = MAX_PLAYERS):
        super().__init__()
        self.fc1 = nn.Linear(hidden, 256)
        self.fc2 = nn.Linear(256, max_players)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.fc1(x))
        return torch.tanh(self.fc2(h))


class OppPolicyHeadV4(nn.Module):
    def __init__(self, hidden: int):
        super().__init__()
        self.fc = nn.Linear(hidden, Config.ACTION_SPACE)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


class PliesHeadV4(nn.Module):
    def __init__(self, hidden: int):
        super().__init__()
        self.fc1 = nn.Linear(hidden, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.fc1(x))
        return self.fc2(h).squeeze(-1)


class ScoreMarginHeadV4(nn.Module):
    """Per-seat score margin = (final_score - mean_other_final_scores) / 1300.

    Output range roughly [-1, 1] via tanh. Padded to MAX_PLAYERS.
    """

    def __init__(self, hidden: int, max_players: int = MAX_PLAYERS):
        super().__init__()
        self.fc1 = nn.Linear(hidden, 128)
        self.fc2 = nn.Linear(128, max_players)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.fc1(x))
        return torch.tanh(self.fc2(h))


class PinFinalHeadV4(nn.Module):
    """Predict per-pin final distance bucket (KataGo ownership-map analog for racing games).

    Output: [B, NUM_PIECES, K_BUCKETS] logits - softmax over distance buckets per pin.
    Buckets coarsely group remaining distance-to-goal at game end:
        0 = at goal (dist=0)
        1 = within 4 cells
        2 = 5-8 cells
        3 = 9-15 cells
        4 = >=16 cells
    """

    def __init__(self, hidden: int, num_pieces: int = Config.NUM_PIECES,
                 num_buckets: int = Config.PIN_FINAL_BUCKETS_V4):
        super().__init__()
        self.num_pieces = num_pieces
        self.num_buckets = num_buckets
        self.fc1 = nn.Linear(hidden, 256)
        self.fc2 = nn.Linear(256, num_pieces * num_buckets)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.fc1(x))
        out = self.fc2(h)
        return out.view(-1, self.num_pieces, self.num_buckets)
