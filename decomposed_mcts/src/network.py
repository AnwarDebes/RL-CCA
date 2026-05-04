"""CMAZ network: encoder + policy head + K-component value head + monotonic mixer.

This is the unified network for the CMAZ workshop subproject. Differs
from the flagship CD-MCTS network in:

  * The "value" head is K-component (one scalar per score component),
    not a Plackett-Luce rank distribution.
  * No coalition head.
  * The monotonic mixer collapses the K-vector to a scalar at MCTS time.
  * The mixer can be overridden at inference for the killer property
    (user-tunable utility).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .cmaz_mcts import CMAZNetworkOutput
from .monotonic_mixer import ComponentValueHead, MonotonicMixer


class CMAZEncoder(nn.Module):
    """MLP encoder identical in spirit to the flagship's MLPEncoder."""

    def __init__(self, input_dim: int, hidden_dim: int = 64, num_layers: int = 2) -> None:
        super().__init__()
        layers = []
        d = input_dim
        for _ in range(num_layers):
            layers.append(nn.Linear(d, hidden_dim))
            layers.append(nn.GELU())
            d = hidden_dim
        self.net = nn.Sequential(*layers)
        self.out_dim = hidden_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CMAZNetwork(nn.Module):
    """Encoder + policy head + per-component value head + mixer.

    Args:
        encoder: nn.Module with attribute `out_dim`.
        action_space_size: int.
        num_components: K, the number of score components.
    """

    def __init__(
        self,
        encoder: nn.Module,
        action_space_size: int,
        num_components: int,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        d = encoder.out_dim
        self.policy_proj = nn.Linear(d, action_space_size)
        self.component_head = ComponentValueHead(d, num_components=num_components)
        self.mixer = MonotonicMixer(feature_dim=d, num_components=num_components)
        self.num_components = num_components

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        policy_logits = self.policy_proj(h)
        v_components = self.component_head(h)  # (..., K)
        # mixer scalar (with learned weights)
        Q_scalar = self.mixer(v_components, h)
        return policy_logits, v_components, Q_scalar


@dataclass
class CMAZEvaluator:
    network: CMAZNetwork
    state_to_features: Callable[[Any], np.ndarray]
    current_player_fn: Callable[[Any], int]
    num_players_fn: Callable[[Any], int]
    terminal_components_fn: Callable[[Any], np.ndarray]
    override_weights: Optional[np.ndarray] = None  # for inference re-weighting

    def terminal_components(self, state: Any) -> np.ndarray:
        return self.terminal_components_fn(state)

    @torch.no_grad()
    def evaluate_cmaz(self, state: Any) -> CMAZNetworkOutput:
        feats = self.state_to_features(state)
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
        """Run the monotonic mixer on numpy inputs (with optional override)."""
        v_t = torch.from_numpy(v).float().unsqueeze(0)
        f_t = torch.from_numpy(features).float().unsqueeze(0)
        with torch.no_grad():
            if self.override_weights is not None:
                ow = torch.from_numpy(self.override_weights).float()
                Q = self.network.mixer(v_t, f_t, override_weights=ow)
            else:
                Q = self.network.mixer(v_t, f_t)
        return float(Q.item())


def cmaz_loss(
    network: CMAZNetwork,
    features: torch.Tensor,         # (B, feature_dim)
    target_policy: torch.Tensor,    # (B, action_space_size)
    legal_mask: torch.Tensor,       # (B, action_space_size) bool
    target_components: torch.Tensor, # (B, K)
    target_total_utility: Optional[torch.Tensor] = None,  # (B,) scalar; if given,
        # supervises the mixer's scalar Q output. Without this signal the mixer
        # parameters receive NO gradient (since the per-component MSE bypasses
        # the mixer), so passing target_total_utility is strongly recommended
        # in production training. Defaults to the mean of target_components if
        # not provided - a soft self-bootstrap target.
    weights: Optional[dict] = None,
) -> tuple[torch.Tensor, dict]:
    """Joint loss: policy + per-component value MSE + mixer-Q regression.

    Three supervisory signals:
      * policy_loss: cross-entropy on masked-softmax against MCTS-derived
        visit-count distribution (AlphaZero standard).
      * components_loss: MSE on per-component values against terminal
        score-component decomposition.
      * value_loss: MSE on mixer's scalar Q against `target_total_utility`
        (the empirical normalised player utility). This is what trains the
        mixer's hypernetwork.

    Defaults: weights = {policy=1.0, components=1.0, value=0.5}.
    """
    if weights is None:
        weights = dict(policy=1.0, components=1.0, value=0.5)

    policy_logits, v_components, q_scalar = network(features)

    # Policy: masked cross-entropy
    masked_logits = policy_logits.masked_fill(~legal_mask, -1e9)
    log_probs = F.log_softmax(masked_logits, dim=-1)
    policy_loss = -(target_policy * log_probs).sum(dim=-1).mean()

    # Per-component MSE
    components_loss = F.mse_loss(v_components, target_components)

    # Value-mixer MSE: ensures the mixer's hypernetwork actually receives
    # gradient. Without this signal the mixer is dead weight.
    if target_total_utility is None:
        # Soft self-target: mean across components, detached so we don't
        # double-count the per-component signal.
        target_total_utility = target_components.mean(dim=-1).detach()
    value_loss = F.mse_loss(q_scalar, target_total_utility.float())

    total = (
        weights.get("policy", 1.0) * policy_loss
        + weights.get("components", 1.0) * components_loss
        + weights.get("value", 0.5) * value_loss
    )
    return total, dict(
        total=total.item(),
        policy=policy_loss.item(),
        components=components_loss.item(),
        value=value_loss.item(),
    )
