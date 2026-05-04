"""Plackett-Luce rank-distribution head for N-player value estimation.

Given an N-player game state s, the head outputs a vector of player "strengths"
theta in R^N. The induced distribution over the symmetric group S_N of final
rank-orderings is

    P(sigma | theta) = prod_{k=1}^{N} exp(theta_{sigma(k)}) /
                       sum_{j=k}^{N}  exp(theta_{sigma(j)})

where sigma(k) is the player who finishes in position k (1 = winner).

This is the Plackett-Luce model (Plackett 1975, Luce 1959). Why we use it:

  * O(N) parameters per state for a distribution over N! orderings.
  * Closed-form sampling: sequential softmax sampling, removing each chosen
    player from the pool.
  * Differentiable log-likelihood for an observed ranking.
  * Closed-form winner marginal P(player p finishes 1st) = softmax(theta)_p.
  * Top-k placement marginals computable in O(N * 2^N) via DP, or by
    enumeration for N <= 6 (which covers the Chinese Checkers tournament).
  * Provably consistent with random-utility theory: theta_p = log E[u_p] under
    a Gumbel utility model.

The downstream MCTS uses placement marginals to compute the per-component
value used in coalition-aware action selection.
"""

from __future__ import annotations

import itertools
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class PlackettLuceHead(nn.Module):
    """Maps an encoder feature vector to player-strength logits theta in R^N.

    The head is intentionally minimal - a single Linear over the encoder
    output. This keeps the inductive bias (PL factorization) the actual
    contribution rather than head capacity.

    Args:
        feature_dim: dimension of the encoder feature vector.
        max_players: maximum number of players supported (head emits a
            fixed-size vector; unused slots are masked at inference using
            ``num_players``).

    Forward returns an unnormalised theta in R^{max_players}. Callers should
    pass a per-state ``num_players`` to mask out unused player slots when
    computing log-likelihoods or marginals.
    """

    def __init__(self, feature_dim: int, max_players: int = 6) -> None:
        super().__init__()
        self.feature_dim = feature_dim
        self.max_players = max_players
        self.proj = nn.Linear(feature_dim, max_players)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.proj(features)


# -----------------------------------------------------------------------------
# Functional core: sampling, log-likelihood, marginals.
#
# All functions accept theta of shape (..., N) where the leading dims may be
# batch dims. Where ``num_players`` is omitted, all N entries are treated as
# active. Where given, theta[..., num_players:] is masked with -inf so it
# contributes zero probability.
# -----------------------------------------------------------------------------


def _mask_inactive(theta: torch.Tensor, num_players: Optional[torch.Tensor]) -> torch.Tensor:
    """Replace logits beyond num_players with -inf (broadcast-safe)."""
    if num_players is None:
        return theta
    N = theta.size(-1)
    arange = torch.arange(N, device=theta.device)
    # num_players has shape (...,) - append a singleton for broadcasting.
    active = arange < num_players.unsqueeze(-1)
    return torch.where(active, theta, torch.full_like(theta, float("-inf")))


def sample_ranking(
    theta: torch.Tensor,
    num_players: Optional[torch.Tensor] = None,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Sample a ranking sigma ~ PL(theta).

    Returns a LongTensor of shape (..., N) where entry [..., k] is the index
    of the player finishing in position k+1 (0-indexed). For inactive slots
    (k >= num_players) the value is -1.

    Implementation: iterated Gumbel-max softmax - equivalent to the
    Plackett-Luce sequential factorization but vectorised.
    """
    theta = _mask_inactive(theta, num_players)
    *batch, N = theta.shape

    # Gumbel-perturbed scores; sorting these gives a PL sample exactly.
    # (This is a folklore identity; cf. Vitelli et al. 2018.)
    if generator is not None:
        u = torch.empty_like(theta).uniform_(generator=generator)
    else:
        u = torch.empty_like(theta).uniform_()
    # Avoid log(0) - clamp away from 0 by a tiny epsilon.
    u = u.clamp(min=1e-12)
    g = -torch.log(-torch.log(u))
    perturbed = theta + g
    # Sort descending: positions 1..N in order.
    order = torch.argsort(perturbed, dim=-1, descending=True)
    # Mask out inactive positions.
    if num_players is not None:
        N_dev = torch.tensor(N, device=theta.device)
        idx = torch.arange(N, device=theta.device)
        active_pos = idx < num_players.unsqueeze(-1)
        order = torch.where(active_pos, order, torch.full_like(order, -1))
    return order


def log_likelihood(
    theta: torch.Tensor,
    ranking: torch.Tensor,
    num_players: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """log P(ranking | theta) under the Plackett-Luce model.

    Args:
        theta: (..., N) player-strength logits.
        ranking: (..., N) LongTensor where ranking[..., k] is the player in
            position k. Inactive positions (k >= num_players) must be -1.
        num_players: optional (...,) LongTensor. If given, only the first
            ``num_players`` positions contribute.

    Returns:
        (...,) log-likelihood.
    """
    theta = _mask_inactive(theta, num_players)
    *batch, N = theta.shape

    # For each k, log P(sigma(k) | sigma(1..k-1)) =
    #     theta_{sigma(k)} - logsumexp(theta_{sigma(k..N)})
    # where logsumexp is taken over the players not yet placed.
    #
    # Vectorise: gather theta in ranking order, then compute
    # cumulative reverse-logsumexp.

    # Replace -1 in ranking with 0 for safe gather; we'll mask them out below.
    safe_rank = ranking.clamp(min=0)
    # gather along last dim: theta_in_order[..., k] = theta[..., ranking[..., k]]
    theta_in_order = torch.gather(theta, dim=-1, index=safe_rank)

    # Mask inactive positions in theta_in_order to -inf BEFORE reverse-logsumexp,
    # otherwise the safe_rank=0 phantoms contaminate the denominator.
    idx = torch.arange(N, device=theta.device)
    if num_players is not None:
        active_pos = idx < num_players.unsqueeze(-1)
    else:
        active_pos = ranking >= 0
    neg_inf = torch.full_like(theta_in_order, float("-inf"))
    theta_in_order_masked = torch.where(active_pos, theta_in_order, neg_inf)

    # reverse cumulative logsumexp: r[k] = logsumexp(theta_in_order_masked[k:])
    flipped = torch.flip(theta_in_order_masked, dims=(-1,))
    rev_logsumexp = torch.flip(torch.logcumsumexp(flipped, dim=-1), dims=(-1,))

    # per-position log-prob: theta_in_order - rev_logsumexp on active positions.
    # Use zeros at inactive positions (so they sum to 0 contribution).
    per_pos = torch.where(
        active_pos,
        theta_in_order_masked - rev_logsumexp,
        torch.zeros_like(theta_in_order_masked),
    )

    return per_pos.sum(dim=-1)


def winner_marginal(theta: torch.Tensor, num_players: Optional[torch.Tensor] = None) -> torch.Tensor:
    """P(player p finishes in position 1) under PL(theta) - closed form softmax."""
    theta = _mask_inactive(theta, num_players)
    return F.softmax(theta, dim=-1)


def placement_marginals_exact(
    theta: torch.Tensor,
    num_players: int,
) -> torch.Tensor:
    """Exact placement marginals P(player p finishes in position k) by enumeration.

    Computes the full N x N marginal matrix M where M[p, k] = P(p in position k+1).
    Cost is O(N!), tractable for N <= 6 which covers our Chinese Checkers
    tournament. For larger N use ``placement_marginals_sampling``.

    Theta must be a 1-D tensor of length N (no batch dimension).
    """
    if theta.dim() != 1:
        raise ValueError("placement_marginals_exact expects 1-D theta")
    if num_players > 8:
        raise ValueError(f"Exact enumeration impractical for N={num_players}")
    N = num_players
    theta = theta[:N]
    # Compute log P(sigma) for every permutation, normalise, then accumulate.
    M = torch.zeros(N, N, device=theta.device, dtype=theta.dtype)
    log_probs = []
    perms = list(itertools.permutations(range(N)))
    for sigma in perms:
        # P(sigma) under PL: prod over k of softmax over remaining.
        lp = 0.0
        remaining = list(range(N))
        for k, p in enumerate(sigma):
            log_denom = torch.logsumexp(theta[remaining], dim=0)
            lp = lp + theta[p] - log_denom
            remaining.remove(p)
        log_probs.append(lp)
    log_probs_t = torch.stack(log_probs)
    # Numerically: PL probabilities sum to 1 already by construction; this
    # normalisation is only a safety net against floating-point drift.
    probs = torch.exp(log_probs_t - torch.logsumexp(log_probs_t, dim=0))
    for sigma, prob in zip(perms, probs):
        for k, p in enumerate(sigma):
            M[p, k] = M[p, k] + prob
    return M


def placement_marginals_sampling(
    theta: torch.Tensor,
    num_players: int,
    num_samples: int = 4096,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Monte-Carlo estimate of placement marginals - used when N > 6."""
    if theta.dim() != 1:
        raise ValueError("placement_marginals_sampling expects 1-D theta")
    N = num_players
    theta = theta[:N]
    # Repeat theta into (num_samples, N) and sample.
    theta_rep = theta.unsqueeze(0).expand(num_samples, -1)
    rankings = sample_ranking(theta_rep, num_players=None, generator=generator)
    M = torch.zeros(N, N, device=theta.device, dtype=theta.dtype)
    for k in range(N):
        # rankings[:, k] is the player in position k+1 across samples
        for p in range(N):
            M[p, k] = (rankings[:, k] == p).float().mean()
    return M


def expected_ranks(theta: torch.Tensor, num_players: int) -> torch.Tensor:
    """E[rank(player p)] under PL(theta), 1-indexed.

    Uses exact enumeration for N <= 6, sampling otherwise. Used as a scalar
    summary of the rank distribution for sanity-checks and ablations.
    """
    if num_players <= 6:
        M = placement_marginals_exact(theta, num_players)
    else:
        M = placement_marginals_sampling(theta, num_players)
    positions = torch.arange(1, num_players + 1, dtype=theta.dtype, device=theta.device)
    return (M * positions).sum(dim=-1)
