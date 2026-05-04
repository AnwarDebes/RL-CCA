"""MCTS utilities: Gumbel sampling, Q normalization, sequential halving."""

import math
from typing import List, Tuple

import numpy as np
import torch


def sample_gumbel(shape, eps=1e-20) -> np.ndarray:
    """Sample from Gumbel(0, 1) distribution."""
    u = np.random.uniform(0.0, 1.0, shape)
    return -np.log(-np.log(u + eps) + eps)


def gumbel_top_k(
    log_priors: np.ndarray,
    legal_actions: List[int],
    m: int,
) -> List[int]:
    """Gumbel-Top-k: select m actions using Gumbel noise + log(prior).

    Args:
        log_priors: log(pi(a)) for each action in the full action space.
        legal_actions: list of legal action indices.
        m: number of candidates to select.

    Returns:
        List of m selected action indices.
    """
    m = min(m, len(legal_actions))
    if m == len(legal_actions):
        return list(legal_actions)

    # Sample Gumbel noise for each legal action
    gumbel_noise = sample_gumbel(len(legal_actions))
    scores = np.array([log_priors[a] for a in legal_actions]) + gumbel_noise

    # Top-m by score
    top_indices = np.argsort(scores)[-m:]
    return [legal_actions[i] for i in top_indices]


def normalize_q_values(q_values: dict, actions: List[int]) -> dict:
    """Normalize Q values to [0, 1] range using min/max across given actions.

    Args:
        q_values: {action: q_value}
        actions: list of actions to consider.

    Returns:
        {action: normalized_q} in [0, 1].
    """
    if not actions:
        return {}

    vals = [q_values.get(a, 0.0) for a in actions]
    q_min = min(vals)
    q_max = max(vals)
    spread = q_max - q_min

    if spread < 1e-8:
        return {a: 0.5 for a in actions}

    return {a: (q_values.get(a, 0.0) - q_min) / spread for a in actions}


def completed_q_score(
    gumbel_noise: float,
    log_prior: float,
    q_normalized: float,
    c_scale: float = 2.5,
) -> float:
    """Compute completed Q-score for sequential halving.

    score(a) = g(a) + log(pi(a)) + sigma(Q_normalized(a))
    where sigma is a monotonic transform (here: c_scale * q_normalized).
    """
    return gumbel_noise + log_prior + c_scale * q_normalized


def sequential_halving_rounds(m: int) -> int:
    """Number of rounds for sequential halving: ceil(log2(m))."""
    if m <= 1:
        return 1
    return math.ceil(math.log2(m))


def compute_improved_policy(
    priors: np.ndarray,
    q_values: dict,
    legal_actions: List[int],
    c_scale: float = 2.5,
) -> np.ndarray:
    """Compute MCTS-improved policy target.

    pi_improved(a) proportional to pi(a) * exp(sigma(Q(a)))
    Normalized to sum to 1. Guaranteed to be at least as good as pi.

    Per Danihelka et al.: Q-values are normalized only over VISITED actions.
    Unvisited legal actions keep their prior probability (no Q bonus).

    Args:
        priors: full prior array of shape (ACTION_SPACE,).
        q_values: {action: q_value} for visited actions.
        legal_actions: list of legal action indices.
        c_scale: temperature for Q normalization.

    Returns:
        Improved policy array of shape (ACTION_SPACE,).
    """
    visited = list(q_values.keys())
    q_norm = normalize_q_values(q_values, visited)
    improved = np.zeros_like(priors)

    for a in legal_actions:
        if a in q_norm:
            # Visited action: boost prior by exp(c_scale * normalized_Q)
            improved[a] = priors[a] * math.exp(c_scale * q_norm[a])
        else:
            # Unvisited action: keep prior only (no Q bonus)
            improved[a] = priors[a]

    # Normalize to sum to 1
    total = improved.sum()
    if total > 1e-8:
        improved /= total
    else:
        # Fallback: uniform over legal
        for a in legal_actions:
            improved[a] = 1.0 / len(legal_actions)

    return improved
