"""CMAZ-on-kingmaker adapter: lets CMAZ run on the synthetic testbed.

The kingmaker game has 3 outcome components per player:
  c0 = won  (binary 1 if rank 1, else 0)
  c1 = mid  (1 if rank 2, else 0)
  c2 = last (1 if rank 3, else 0)

These form a one-hot decomposition of rank, and the CMAZ mixer learns
to combine them. For inference-time override, we can demonstrate:
  * weight on c0 = "win at any cost"
  * weight on c2 negatively = "avoid last place"
  * uniform = "treat all places equally" (degenerates to expected rank)

Also provides a simple feature encoder matching the kingmaker state.
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

from flagship_coalition_mcts.src.games.kingmaker import (
    KingmakerGame, KingmakerState, NUM_ACTIONS, NUM_PLAYERS, final_ranks,
)
from .cmaz_mcts import CMAZNetworkOutput
from .network import CMAZEncoder, CMAZNetwork


def kingmaker_features_for_cmaz(state: KingmakerState) -> np.ndarray:
    """12-dim feature vector matching flagship's kingmaker_features."""
    feats = np.zeros(12, dtype=np.float32)
    feats[0:3] = np.array(state.positions) / 3.0
    for p in state.finish_order:
        feats[3 + p] = 1.0
    feats[6 + state.next_player] = 1.0
    feats[9] = state.move_count / 6.0
    feats[10] = float(len(state.finish_order)) / 3.0
    feats[11] = 1.0
    return feats


def kingmaker_score_components(state: KingmakerState, player: int) -> np.ndarray:
    """At a terminal state, return one-hot rank decomposition for `player`.

    For non-terminal states (used during MCTS rollouts that bottom out
    early), we approximate by `is leading` / `is mid` / `is last` based
    on current position.
    """
    N = NUM_PLAYERS
    out = np.zeros(3, dtype=np.float32)
    if KingmakerGame.is_terminal(state):
        ranks = final_ranks(state)
        rank = ranks[player]
        out[rank - 1] = 1.0
    else:
        # Heuristic: rank by current position
        positions = list(state.positions)
        sorted_players = sorted(range(N), key=lambda p: (-positions[p], p))
        rank = sorted_players.index(player) + 1
        out[rank - 1] = 1.0
    return out


class KingmakerCMAZEvaluator:
    """CMAZ evaluator on the kingmaker game."""

    def __init__(
        self, network: CMAZNetwork, override_weights: Optional[np.ndarray] = None,
    ) -> None:
        self.network = network
        self.override_weights = override_weights

    def terminal_components(self, state) -> np.ndarray:
        cp = state.next_player
        return kingmaker_score_components(state, cp)

    @torch.no_grad()
    def evaluate_cmaz(self, state) -> CMAZNetworkOutput:
        feats = kingmaker_features_for_cmaz(state)
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


def build_cmaz_kingmaker_network(
    feature_dim: int = 12,
    hidden_dim: int = 24,
    num_components: int = 3,
) -> CMAZNetwork:
    encoder = CMAZEncoder(input_dim=feature_dim, hidden_dim=hidden_dim, num_layers=2)
    return CMAZNetwork(
        encoder=encoder,
        action_space_size=NUM_ACTIONS,
        num_components=num_components,
    )
