"""CD-MCTS network: shared encoder + four heads (policy, PL, coalition, value).

This is the unified neural net that the MCTS calls. It exposes the
duck-typed `evaluate(state)` interface expected by mcts.py.

Heads
-----
1. **Policy head** - softmax over the full action space, restricted to
   legal actions at MCTS time. Trained on visit-count distributions.

2. **Plackett-Luce rank head** - outputs strength logits θ ∈ ℝ^N. Trained
   on the empirical final ranking (one observation per game outcome) via
   PL log-likelihood.

3. **Coalition head** - outputs pairwise alignment logits A ∈ ℝ^{N×N} and
   a positive concentration β. Trained on an empirical "co-finished-
   ahead" indicator: for each rollout, compute for each player p the
   subset of opponents who finished ahead - that's the observed
   coalition. NLL of that observed subset under the head's posterior.

4. **Scalar value head** - outputs a single scalar V(s) per current
   player (kept for ablations: A0/A1 use this instead of PL marginal-
   derived value). Trained against the normalised utility of the current
   player's final rank.

Encoder
-------
A small MLP for the kingmaker game (state size ~12 features). For real
games (Chinese Checkers, Halma) this would be a CNN or graph net; the
interface is unchanged.

This file is CPU-runnable and depends only on torch + numpy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .coalition_head import CoalitionHead, coalition_marginal_alignment
from .mcts import NetworkOutput
from .plackett_luce import PlackettLuceHead, placement_marginals_exact


class MLPEncoder(nn.Module):
    """Generic MLP encoder used by the kingmaker testbed."""

    def __init__(self, input_dim: int, hidden_dim: int = 64, num_layers: int = 2) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        d = input_dim
        for _ in range(num_layers):
            layers.append(nn.Linear(d, hidden_dim))
            layers.append(nn.GELU())
            d = hidden_dim
        self.net = nn.Sequential(*layers)
        self.out_dim = hidden_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CDMCTSNetwork(nn.Module):
    """Full CD-MCTS network module.

    Parameters
    ----------
    encoder : nn.Module with attribute `out_dim`
    action_space_size : int (full action space, MCTS restricts to legal)
    max_players : int

    Forward returns:
        policy_logits, theta, A_alignment, beta, scalar_value
    """

    def __init__(
        self,
        encoder: nn.Module,
        action_space_size: int,
        max_players: int = 6,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.max_players = max_players
        d = encoder.out_dim
        self.policy_proj = nn.Linear(d, action_space_size)
        self.pl_head = PlackettLuceHead(d, max_players=max_players)
        self.coalition_head = CoalitionHead(d, max_players=max_players)
        self.value_proj = nn.Linear(d, 1)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        policy_logits = self.policy_proj(h)
        theta = self.pl_head(h)
        A, beta = self.coalition_head(h)
        scalar_v = torch.tanh(self.value_proj(h)).squeeze(-1)
        return policy_logits, theta, A, beta, scalar_v


# ----------------------------------------------------------------------
# Adapter: wraps a CDMCTSNetwork + a state-encoder function and exposes
# `evaluate(state)` returning a NetworkOutput as expected by mcts.py.
# ----------------------------------------------------------------------


@dataclass
class CDMCTSEvaluator:
    """Glue between an arbitrary game state and the torch-side network.

    Args:
        network: CDMCTSNetwork
        state_to_features: callable(state) -> 1D float ndarray of fixed length.
        action_space_size: int (full action space; mcts restricts later).
        current_player_fn: callable(state) -> int.
        num_players_fn: callable(state) -> int.
    """

    network: CDMCTSNetwork
    state_to_features: Callable[[Any], np.ndarray]
    action_space_size: int
    current_player_fn: Callable[[Any], int]
    num_players_fn: Callable[[Any], int]

    @torch.no_grad()
    def evaluate(self, state: Any) -> NetworkOutput:
        feats = self.state_to_features(state)
        x = torch.from_numpy(feats).float().unsqueeze(0)
        policy_logits, theta, A, beta, _ = self.network(x)
        # Drop batch dim
        policy_logits = policy_logits[0]
        theta = theta[0]
        A = A[0]
        beta = beta[0]
        N = self.num_players_fn(state)
        # Restrict theta to active players for marginal computation.
        # Use exact enumeration since N <= 6 in our games.
        M = placement_marginals_exact(theta[:N], num_players=N)
        # If N < max_players, pad M to (max_players, max_players) with zeros.
        N_max = self.network.max_players
        M_full = np.zeros((N_max, N_max), dtype=np.float64)
        M_full[:N, :N] = M.cpu().numpy().astype(np.float64)
        # Coalition alignment from current player.
        cp = self.current_player_fn(state)
        coal_align = coalition_marginal_alignment(A, beta, player=cp, num_players=N)
        coal_full = np.zeros(N_max, dtype=np.float64)
        coal_full[:N] = coal_align.cpu().numpy().astype(np.float64)
        # Policy: softmax over the full action space
        prior = F.softmax(policy_logits, dim=-1).cpu().numpy().astype(np.float64)
        return NetworkOutput(
            prior_policy=prior,
            placement_marginals=M_full,
            coalition_alignment=coal_full,
        )


# ----------------------------------------------------------------------
# Loss functions for joint training
# ----------------------------------------------------------------------


def cdmcts_loss(
    network: CDMCTSNetwork,
    features: torch.Tensor,        # (batch, feature_dim)
    target_policy: torch.Tensor,    # (batch, action_space_size)
    legal_mask: torch.Tensor,       # (batch, action_space_size) bool
    observed_ranking: torch.Tensor, # (batch, max_players) long, with -1 padding
    num_players_per: torch.Tensor,  # (batch,) long
    observed_coalition_index: torch.Tensor,  # (batch,) long index into the
    #   subset enumeration for the current_player; -1 if not used
    current_player_per: torch.Tensor,  # (batch,) long
    target_scalar_value: torch.Tensor,  # (batch,) float in [-1, 1]
    weights: Optional[dict] = None,
) -> tuple[torch.Tensor, dict]:
    """Joint loss: policy + PL log-likelihood + coalition NLL + scalar value.

    The coalition NLL is evaluated per-state by enumerating subsets for
    the current player and indexing into them. Caller supplies the
    integer index of the observed coalition.

    Returns (total_loss, components_dict).
    """
    if weights is None:
        weights = dict(policy=1.0, pl=1.0, coalition=0.5, value=0.5)

    # Forward pass
    policy_logits, theta, A_batch, beta_batch, scalar_v = network(features)

    # Policy: cross-entropy with masked softmax
    masked_logits = policy_logits.masked_fill(~legal_mask, -1e9)
    log_probs = F.log_softmax(masked_logits, dim=-1)
    policy_loss = -(target_policy * log_probs).sum(dim=-1).mean()

    # Plackett-Luce log-likelihood on observed ranking
    from .plackett_luce import log_likelihood as pl_ll
    pl_log_lik = pl_ll(theta, observed_ranking, num_players=num_players_per)
    pl_loss = -pl_log_lik.mean()

    # Scalar value
    value_loss = F.mse_loss(scalar_v, target_scalar_value)

    # Coalition NLL - compute per-batch-element with the subset enumeration.
    # We sum per-element coalition log-prob then negate. The enumeration is
    # done in pure python (small N), so no autograd issue.
    from .coalition_head import coalition_log_probs
    coalition_terms = []
    valid_coal = (observed_coalition_index >= 0)
    for i in range(features.shape[0]):
        if not valid_coal[i].item():
            continue
        N_i = int(num_players_per[i].item())
        cp_i = int(current_player_per[i].item())
        idx_i = int(observed_coalition_index[i].item())
        _, log_probs_i = coalition_log_probs(A_batch[i], beta_batch[i], cp_i, N_i)
        coalition_terms.append(-log_probs_i[idx_i])
    if coalition_terms:
        coalition_loss = torch.stack(coalition_terms).mean()
    else:
        coalition_loss = torch.tensor(0.0, device=features.device)

    total = (
        weights["policy"] * policy_loss
        + weights["pl"] * pl_loss
        + weights["coalition"] * coalition_loss
        + weights["value"] * value_loss
    )
    return total, dict(
        total=total.item(),
        policy=policy_loss.item(),
        pl=pl_loss.item(),
        coalition=coalition_loss.item(),
        value=value_loss.item(),
    )
