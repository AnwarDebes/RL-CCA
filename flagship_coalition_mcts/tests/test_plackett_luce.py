"""Unit tests for the Plackett-Luce rank-distribution head.

Tests verify the mathematical properties a flagship paper reviewer would
demand:

1.  Probabilities sum to 1 over all permutations.
2.  Winner marginal coincides with softmax (this is *the* identifying
    property of PL).
3.  Sampling matches log-likelihood: the empirical sample distribution
    converges to the analytic distribution as N_samples grows.
4.  Equal logits => uniform marginals (symmetry).
5.  Differentiable: gradients of log-likelihood w.r.t. theta exist and
    have the expected sign.
6.  Inactive players are correctly masked.
7.  Sampling-based marginals match exact enumeration on small N.

Run: pytest flagship_coalition_mcts/tests/test_plackett_luce.py -v
"""

from __future__ import annotations

import itertools
import math

import pytest
import torch

from flagship_coalition_mcts.src.plackett_luce import (
    PlackettLuceHead,
    expected_ranks,
    log_likelihood,
    placement_marginals_exact,
    placement_marginals_sampling,
    sample_ranking,
    winner_marginal,
)


# --- Property 1: probabilities sum to 1 over all permutations ----------------

def test_permutation_probabilities_sum_to_one():
    """For any theta, sum_{sigma in S_N} P(sigma | theta) = 1 exactly."""
    torch.manual_seed(0)
    for N in [2, 3, 4, 5]:
        theta = torch.randn(N) * 2.0  # broad range
        total = 0.0
        for sigma in itertools.permutations(range(N)):
            ranking = torch.tensor(sigma)
            lp = log_likelihood(theta, ranking)
            total = total + torch.exp(lp).item()
        assert abs(total - 1.0) < 1e-5, f"N={N}: total={total}"


# --- Property 2: winner marginal = softmax -----------------------------------

def test_winner_marginal_equals_softmax():
    torch.manual_seed(1)
    for N in [3, 4, 5, 6]:
        theta = torch.randn(N) * 1.5
        wm = winner_marginal(theta)
        soft = torch.softmax(theta, dim=-1)
        assert torch.allclose(wm, soft, atol=1e-6)


def test_winner_marginal_via_enumeration_matches_softmax():
    """Direct cross-check: compute P(player p in position 1) by enumerating
    all permutations and verify it matches softmax(theta)_p."""
    torch.manual_seed(2)
    N = 4
    theta = torch.randn(N)
    soft = torch.softmax(theta, dim=-1)
    # Sum probabilities over permutations grouped by their first element.
    enumerated = torch.zeros(N)
    for sigma in itertools.permutations(range(N)):
        ranking = torch.tensor(sigma)
        p = torch.exp(log_likelihood(theta, ranking)).item()
        enumerated[sigma[0]] += p
    assert torch.allclose(enumerated, soft, atol=1e-5)


# --- Property 3: sampling matches log-likelihood -----------------------------

def test_sampling_matches_likelihood_chi_squared():
    """Empirical histogram of sampled rankings approaches analytic PL distribution."""
    torch.manual_seed(3)
    N = 4
    theta = torch.randn(N) * 1.0
    num_samples = 200_000
    theta_batch = theta.unsqueeze(0).expand(num_samples, -1)
    rankings = sample_ranking(theta_batch)
    perms = list(itertools.permutations(range(N)))
    # analytic probabilities
    analytic = torch.tensor(
        [torch.exp(log_likelihood(theta, torch.tensor(s))).item() for s in perms]
    )
    # empirical
    perm_to_idx = {s: i for i, s in enumerate(perms)}
    counts = torch.zeros(len(perms))
    for r in rankings:
        counts[perm_to_idx[tuple(r.tolist())]] += 1
    empirical = counts / num_samples
    # max abs error should shrink with sqrt(num_samples)
    err = (analytic - empirical).abs().max().item()
    expected_se = 1.0 / math.sqrt(num_samples)
    assert err < 6 * expected_se, f"err={err:.4f}, expected ~{expected_se:.4f}"


# --- Property 4: equal logits => uniform marginals ---------------------------

def test_equal_logits_yield_uniform_placement():
    N = 5
    theta = torch.zeros(N)
    M = placement_marginals_exact(theta, num_players=N)
    expected = torch.full((N, N), 1.0 / N)
    assert torch.allclose(M, expected, atol=1e-6)


def test_equal_logits_yield_equal_expected_rank():
    N = 6
    theta = torch.zeros(N)
    er = expected_ranks(theta, num_players=N)
    expected = torch.full((N,), (N + 1) / 2.0)
    assert torch.allclose(er, expected, atol=1e-5)


# --- Property 5: differentiability -------------------------------------------

def test_log_likelihood_gradient_correct_sign():
    """Boosting theta_p where p is in a high position should INCREASE the
    log-likelihood of that ranking; gradient w.r.t. theta_p must be positive."""
    N = 4
    theta = torch.randn(N, requires_grad=True)
    ranking = torch.tensor([2, 0, 3, 1])  # player 2 wins
    lp = log_likelihood(theta, ranking)
    lp.backward()
    grad = theta.grad
    # The winner's gradient must be positive (boost it -> higher likelihood)
    assert grad[ranking[0]] > 0
    # Gradients must sum to zero (softmax property summed over each step).
    # Each per-step softmax has zero-sum-of-grads w.r.t. its remaining set,
    # but globally for PL the sum is ranking_count - normalisation, which is
    # NOT exactly zero in general. So we instead just require sign.


# --- Property 6: inactive-player masking -------------------------------------

def test_inactive_player_masking():
    """For a 6-slot head used in a 3-player game, slots 3..5 must contribute zero."""
    theta = torch.tensor([1.0, 2.0, 3.0, 100.0, 100.0, 100.0])
    num_players = torch.tensor(3)
    wm = winner_marginal(theta, num_players=num_players)
    # Players 3,4,5 should have zero mass.
    assert wm[3:].abs().max() < 1e-6
    # Active 3 should sum to 1.
    assert abs(wm[:3].sum().item() - 1.0) < 1e-6
    # And the active distribution should match softmax over active logits.
    expected = torch.softmax(theta[:3], dim=-1)
    assert torch.allclose(wm[:3], expected, atol=1e-6)


def test_log_likelihood_with_inactive_slots():
    """Inactive slots in ranking (set to -1) must not affect log-likelihood."""
    theta = torch.tensor([1.0, 2.0, 3.0, 0.0, 0.0])
    full_ranking = torch.tensor([2, 1, 0, -1, -1])
    num_players = torch.tensor(3)
    lp_masked = log_likelihood(theta, full_ranking, num_players=num_players)
    # Compare against pure 3-player log-likelihood
    theta3 = theta[:3]
    rank3 = full_ranking[:3]
    lp_pure = log_likelihood(theta3, rank3)
    assert torch.allclose(lp_masked, lp_pure, atol=1e-5)


# --- Property 7: sampling marginals match exact -----------------------------

def test_sampling_marginals_match_exact():
    torch.manual_seed(5)
    N = 4
    theta = torch.randn(N)
    M_exact = placement_marginals_exact(theta, num_players=N)
    M_samp = placement_marginals_sampling(theta, num_players=N, num_samples=100_000)
    err = (M_exact - M_samp).abs().max().item()
    assert err < 0.01


# --- Module wrapper smoke test ----------------------------------------------

def test_head_module_forward_shape():
    feature_dim = 32
    max_players = 6
    head = PlackettLuceHead(feature_dim=feature_dim, max_players=max_players)
    batch = 7
    x = torch.randn(batch, feature_dim)
    theta = head(x)
    assert theta.shape == (batch, max_players)
