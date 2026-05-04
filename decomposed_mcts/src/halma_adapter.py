"""CMAZ-on-Halma adapter: lets CMAZ run on the small-Halma testbed.

Halma is the 3-player coordination/race game. Like kingmaker, the
natural per-component score decomposition is the rank one-hot:
  c0 = won  (rank 1)
  c1 = mid  (rank 2)
  c2 = last (rank 3)

This adapter mirrors `kingmaker_adapter.py` for the Halma board (5×5
grid, 3 players × 3 pieces).
"""

from __future__ import annotations

import os
import sys
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

_NEXUS_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)
if _NEXUS_ROOT not in sys.path:
    sys.path.insert(0, _NEXUS_ROOT)

from flagship_coalition_mcts.src.games.halma_small import (
    HalmaSmallGame,
    HalmaState,
    NUM_ACTIONS as HALMA_NUM_ACTIONS,
    NUM_PLAYERS as HALMA_NUM_PLAYERS,
    final_ranks as halma_final_ranks,
    state_to_features as halma_state_to_features,
)
from .cmaz_mcts import CMAZNetworkOutput
from .network import CMAZEncoder, CMAZNetwork


def halma_score_components(state, player: int) -> np.ndarray:
    """One-hot rank decomposition for the Halma player.

    For non-terminal states we fall back to a position-based heuristic
    that is consistent with the terminal final_ranks tie-breaking.
    """
    out = np.zeros(3, dtype=np.float32)
    if HalmaSmallGame.is_terminal(state):
        ranks = halma_final_ranks(state)
        rank = ranks[player]
        out[rank - 1] = 1.0
        return out
    # Non-terminal heuristic: rank by goal-progress (pieces in goal).
    progress = []
    for p in range(HALMA_NUM_PLAYERS):
        from flagship_coalition_mcts.src.games.halma_small import _progress
        progress.append((-_progress(state, p), p))
    progress.sort()
    rank = [pp for _, pp in progress].index(player) + 1
    out[rank - 1] = 1.0
    return out


class HalmaCMAZEvaluator:
    """CMAZ evaluator on the Halma testbed."""

    def __init__(
        self, network: CMAZNetwork, override_weights: Optional[np.ndarray] = None,
    ) -> None:
        self.network = network
        self.override_weights = override_weights

    def terminal_components(self, state) -> np.ndarray:
        cp = state.next_player
        return halma_score_components(state, cp)

    @torch.no_grad()
    def evaluate_cmaz(self, state) -> CMAZNetworkOutput:
        feats = halma_state_to_features(state)
        x = torch.from_numpy(feats).float().unsqueeze(0)
        h = self.network.encoder(x)
        policy_logits = self.network.policy_proj(h)
        v_components = self.network.component_head(h)[0]
        prior = F.softmax(policy_logits[0], dim=-1).cpu().numpy().astype(np.float64)
        return CMAZNetworkOutput(
            prior_policy=prior,
            component_values=v_components.cpu().numpy().astype(np.float64),
            encoder_features=h[0].cpu().numpy().astype(np.float64),
        )

    def mixer_apply(self, v: np.ndarray, features: np.ndarray) -> float:
        v_t = torch.from_numpy(v).float().unsqueeze(0)
        f_t = torch.from_numpy(features).float().unsqueeze(0)
        with torch.no_grad():
            if self.override_weights is not None:
                ow = torch.from_numpy(self.override_weights).float()
                Q = self.network.mixer(v_t, f_t, override_weights=ow)
            else:
                Q = self.network.mixer(v_t, f_t)
        return float(Q.item())


def build_cmaz_halma_network(
    hidden_dim: int = 64,
    num_components: int = 3,
) -> CMAZNetwork:
    # halma_state_to_features dim = 25*4 + 3 + 3 + 1 = 107
    encoder = CMAZEncoder(input_dim=107, hidden_dim=hidden_dim, num_layers=2)
    return CMAZNetwork(
        encoder=encoder,
        action_space_size=HALMA_NUM_ACTIONS,
        num_components=num_components,
    )
