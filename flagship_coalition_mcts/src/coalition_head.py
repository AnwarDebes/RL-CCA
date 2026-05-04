"""Coalition-belief head.

At each state s, this head outputs a posterior over which subsets of
opponents are *aligned against* the current player. This is the second
novel pillar of CD-MCTS (the first being the Plackett-Luce rank value head).

Why this head exists
--------------------
In N-player non-zero-sum games (Chinese Checkers, Halma, multi-player Go),
"kingmaker" and "coalition" dynamics are pervasive: a losing player can
choose who wins, multiple opponents can implicitly cooperate to suppress a
leader, etc. Standard MCTS variants assume one of three rigid opponent
models - independent (maxn), all-vs-me (paranoid), or one-best-reply
(BRS) - all of which are provably suboptimal in the general N-player case
(Sturtevant & Korf 2000, Schadd 2011).

We instead *learn* a state-conditional belief over coalition structures,
trained via self-play, and use it to weight the no-regret action selector.

Tractable factorization
-----------------------
The number of coalitions on N-1 opponents is 2^{N-1} (subsets), and the
number of partitions is the Bell number B_{N-1}. For N=6 these are 32 and
52 respectively - small enough to enumerate at inference, but only with a
factorization that keeps the parameter count manageable.

We factorize via *pairwise alignment*. The head outputs:

  * an N x N symmetric matrix A of pairwise alignment logits (zero
    diagonal), passed through sigmoid -> pairwise alignment probabilities;
  * a scalar concentration parameter beta >= 0.

The implied posterior over a coalition C ⊆ {opponents of player p} is

    P(C | s, p) ∝ exp(beta * sum_{q in C} A[p, q]
                      - beta * sum_{q ∉ C, q ≠ p} A[p, q])

This is a state-conditional Ising-like model on opponent alignment with
field beta * A[p,:]. It

  * collapses to uniform when beta -> 0 (no coalition information);
  * collapses to a single hard coalition when beta -> infty;
  * has O(N^2) parameters (matches Plackett-Luce);
  * is differentiable and trainable end-to-end via a self-play target
    that compares predicted P(C) against an empirical "co-finished-ahead"
    indicator from rollout outcomes.

The factorization is admittedly weaker than a full Bell-number-many
distribution, but it captures the dominant pairwise structure, and the
ablation in our paper compares against full-categorical and truncated
versions to show the trade-off.
"""

from __future__ import annotations

import itertools
import math
from typing import Iterable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class CoalitionHead(nn.Module):
    """Outputs (pairwise_alignment_logits, concentration) given encoder features.

    Args:
        feature_dim: encoder feature size.
        max_players: head emits an N x N matrix sized for max_players.

    Forward returns:
        align_logits: (..., N, N) symmetric, zero-diagonal logits.
        beta: (...,) non-negative concentration scalar.
    """

    def __init__(self, feature_dim: int, max_players: int = 6) -> None:
        super().__init__()
        self.feature_dim = feature_dim
        self.max_players = max_players
        # Predict the upper-triangular entries of A; we symmetrise.
        self.num_pairs = max_players * (max_players - 1) // 2
        self.pair_proj = nn.Linear(feature_dim, self.num_pairs)
        # Predict log-beta, then exp to ensure positivity.
        self.beta_proj = nn.Linear(feature_dim, 1)

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        N = self.max_players
        pair_logits = self.pair_proj(features)  # (..., num_pairs)
        # Build symmetric (..., N, N) with zero diagonal.
        batch_shape = pair_logits.shape[:-1]
        A = torch.zeros(*batch_shape, N, N, dtype=features.dtype, device=features.device)
        idx_i, idx_j = torch.triu_indices(N, N, offset=1)
        A[..., idx_i, idx_j] = pair_logits
        A[..., idx_j, idx_i] = pair_logits  # symmetric
        log_beta = self.beta_proj(features).squeeze(-1)
        beta = F.softplus(log_beta)  # non-negative
        return A, beta


# -----------------------------------------------------------------------------
# Functional core: coalition log-probabilities, entropy, sampling.
# All functions assume a SINGLE state (no batch) for clarity. Vectorisation
# wraps them at training time.
# -----------------------------------------------------------------------------


def _enumerate_coalitions(opponents: Iterable[int]) -> list[tuple[int, ...]]:
    """All subsets of `opponents`, returned as sorted tuples."""
    opponents = sorted(opponents)
    out = []
    for r in range(len(opponents) + 1):
        for s in itertools.combinations(opponents, r):
            out.append(s)
    return out


def coalition_log_unnormalised(
    A: torch.Tensor,
    beta: torch.Tensor,
    coalition: tuple[int, ...],
    player: int,
    num_players: int,
) -> torch.Tensor:
    """log f(C) where P(C) ∝ f(C) under the Ising-style coalition model.

        log f(C) = beta * (sum_{q in C} A[player, q]
                            - sum_{q in opp \\ C} A[player, q])

    where opp = opponents-of-player. This pulls A toward +1 for pairs
    that should be in C and toward -1 for pairs that should not.
    """
    opp = [q for q in range(num_players) if q != player]
    in_c = list(coalition)
    out_c = [q for q in opp if q not in coalition]
    align_in = A[player, in_c].sum() if in_c else torch.tensor(0.0, device=A.device)
    align_out = A[player, out_c].sum() if out_c else torch.tensor(0.0, device=A.device)
    return beta * (align_in - align_out)


def coalition_log_probs(
    A: torch.Tensor,
    beta: torch.Tensor,
    player: int,
    num_players: int,
) -> tuple[list[tuple[int, ...]], torch.Tensor]:
    """Enumerate all 2^{N-1} coalitions of opponents-of-player and return
    their normalised log-probabilities under the Ising model.

    Returns:
        coalitions: list of opponent-subset tuples (sorted, possibly empty).
        log_probs: (2^{N-1},) tensor of log-probabilities, summing to 1
            after exp.
    """
    opp = [q for q in range(num_players) if q != player]
    coals = _enumerate_coalitions(opp)
    logf = torch.stack(
        [coalition_log_unnormalised(A, beta, c, player, num_players) for c in coals]
    )
    log_z = torch.logsumexp(logf, dim=0)
    return coals, logf - log_z


def coalition_marginal_alignment(
    A: torch.Tensor,
    beta: torch.Tensor,
    player: int,
    num_players: int,
) -> torch.Tensor:
    """For each opponent q, compute P(q in coalition against player).

    Returns a vector of length num_players where entry [player] is
    arbitrary (set to 0).
    """
    coals, log_probs = coalition_log_probs(A, beta, player, num_players)
    probs = torch.exp(log_probs)
    out = torch.zeros(num_players, device=A.device, dtype=A.dtype)
    for c, p in zip(coals, probs):
        for q in c:
            out[q] = out[q] + p
    return out


def coalition_entropy(
    A: torch.Tensor,
    beta: torch.Tensor,
    player: int,
    num_players: int,
) -> torch.Tensor:
    """H(coalition posterior) - used as an auxiliary loss to prevent
    degeneracy (head collapsing to uniform or to a single fixed coalition).
    """
    _, log_probs = coalition_log_probs(A, beta, player, num_players)
    probs = torch.exp(log_probs)
    return -(probs * log_probs).sum()


def negative_log_likelihood_coalition(
    A: torch.Tensor,
    beta: torch.Tensor,
    observed_coalition: tuple[int, ...],
    player: int,
    num_players: int,
) -> torch.Tensor:
    """-log P(observed_coalition | s, player). The training target.

    The "observed coalition" is built from rollout outcomes: at the end of
    a self-play game, the set of opponents who finished ahead of `player`
    (or analogously, opponents whose actions correlated with hurting
    `player`'s outcome - this is the operationalisation we ablate in the
    paper).
    """
    coals, log_probs = coalition_log_probs(A, beta, player, num_players)
    obs = tuple(sorted(observed_coalition))
    idx = coals.index(obs)
    return -log_probs[idx]
