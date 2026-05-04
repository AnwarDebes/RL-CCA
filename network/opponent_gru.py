"""Opponent GRU: encodes opponent action history into a latent style embedding.

Hidden state persists across moves within a game, resets between games.
"""

import torch
import torch.nn as nn

from config import Config


class OpponentGRU(nn.Module):
    """Learned opponent modeling via GRU over opponent action sequence."""

    def __init__(
        self,
        n_actions: int = Config.ACTION_SPACE,
        embed_dim: int = Config.OPP_GRU_EMBED,
        hidden_size: int = Config.OPP_GRU_HIDDEN,
    ):
        super().__init__()
        # +1 for "no action yet" token
        self.action_embed = nn.Embedding(n_actions + 1, embed_dim)
        self.gru = nn.GRU(embed_dim, hidden_size, batch_first=True)
        self.hidden_size = hidden_size
        self.no_action_token = n_actions  # index for "no action yet"

    def forward(
        self, opponent_action: torch.Tensor, hidden: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            opponent_action: [B] int - last opponent action (or no_action_token).
            hidden: [1, B, hidden_size] - GRU hidden state.

        Returns:
            (opp_embedding [B, hidden_size], new_hidden [1, B, hidden_size])
        """
        embedded = self.action_embed(opponent_action).unsqueeze(1)  # [B, 1, embed_dim]
        output, new_hidden = self.gru(embedded, hidden)  # output: [B, 1, hidden_size]
        opp_embedding = output.squeeze(1)  # [B, hidden_size]
        return opp_embedding, new_hidden

    def init_hidden(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Initialize hidden state to zeros."""
        return torch.zeros(1, batch_size, self.hidden_size, device=device)
