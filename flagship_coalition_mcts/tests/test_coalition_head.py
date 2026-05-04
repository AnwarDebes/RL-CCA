"""Unit tests for the coalition-belief head.

Tests verify:

1. Coalition probabilities sum to 1.
2. beta -> 0 collapses to uniform over coalitions.
3. beta -> infty collapses to argmax coalition (the one that maximises
   alignment with high A entries).
4. Symmetry: pairwise matrix is symmetric and zero-diagonal.
5. Marginal alignments are in [0, 1].
6. Entropy is non-negative and maximised at uniform.
7. Differentiability: NLL of an observed coalition is differentiable
   w.r.t. A and beta.
8. Module-forward shape sanity.
9. NLL gradient sign: increasing the alignment of an opponent that IS in
   the observed coalition decreases NLL.
"""

from __future__ import annotations

import math

import pytest
import torch

from flagship_coalition_mcts.src.coalition_head import (
    CoalitionHead,
    coalition_entropy,
    coalition_log_probs,
    coalition_marginal_alignment,
    negative_log_likelihood_coalition,
)


def _random_A(N: int, scale: float = 1.0) -> torch.Tensor:
    """Symmetric, zero-diagonal random matrix."""
    R = torch.randn(N, N) * scale
    A = (R + R.t()) / 2.0
    A.fill_diagonal_(0.0)
    return A


# --- Property 1: probabilities sum to 1 -------------------------------------

def test_coalition_probabilities_sum_to_one():
    torch.manual_seed(0)
    for N in [2, 3, 4, 5, 6]:
        A = _random_A(N, scale=1.5)
        beta = torch.tensor(0.7)
        for player in range(N):
            _, log_probs = coalition_log_probs(A, beta, player=player, num_players=N)
            total = torch.exp(log_probs).sum().item()
            assert abs(total - 1.0) < 1e-5, f"N={N}, player={player}: total={total}"


# --- Property 2: beta -> 0 yields uniform -----------------------------------

def test_beta_zero_yields_uniform():
    torch.manual_seed(1)
    N = 4
    A = _random_A(N, scale=10.0)  # large logits, but beta will damp them
    beta = torch.tensor(0.0)
    coals, log_probs = coalition_log_probs(A, beta, player=0, num_players=N)
    expected_log_uniform = -math.log(len(coals))
    err = (log_probs - expected_log_uniform).abs().max().item()
    assert err < 1e-5


# --- Property 3: beta -> infinity collapses to argmax -----------------------

def test_beta_large_collapses_to_argmax():
    torch.manual_seed(2)
    N = 4
    A = _random_A(N, scale=1.0)
    beta = torch.tensor(200.0)
    coals, log_probs = coalition_log_probs(A, beta, player=0, num_players=N)
    probs = torch.exp(log_probs)
    # The argmax coalition for player 0 is C = {q : A[0,q] > 0}.
    desired = tuple(sorted([q for q in range(1, N) if A[0, q].item() > 0]))
    argmax_idx = probs.argmax().item()
    assert coals[argmax_idx] == desired, (
        f"got {coals[argmax_idx]}, expected {desired} from A[0]={A[0]}"
    )
    assert probs[argmax_idx].item() > 0.999


# --- Property 4: symmetry of forward output ---------------------------------

def test_module_forward_symmetric_and_zero_diag():
    torch.manual_seed(3)
    head = CoalitionHead(feature_dim=16, max_players=5)
    x = torch.randn(3, 16)
    A, beta = head(x)
    assert A.shape == (3, 5, 5)
    # Symmetric
    assert torch.allclose(A, A.transpose(-1, -2), atol=1e-6)
    # Zero diagonal
    diag = torch.diagonal(A, dim1=-2, dim2=-1)
    assert torch.allclose(diag, torch.zeros_like(diag), atol=1e-6)
    # Beta non-negative
    assert (beta >= 0).all()


# --- Property 5: marginals are valid probabilities --------------------------

def test_marginal_alignments_in_unit_interval():
    torch.manual_seed(4)
    N = 5
    A = _random_A(N, scale=1.5)
    beta = torch.tensor(0.8)
    for player in range(N):
        m = coalition_marginal_alignment(A, beta, player=player, num_players=N)
        assert (m >= -1e-6).all() and (m <= 1.0 + 1e-6).all()
        # Self-marginal is 0 by definition
        assert abs(m[player].item()) < 1e-6


# --- Property 6: entropy non-negative, max at uniform -----------------------

def test_entropy_non_negative_and_max_at_uniform():
    torch.manual_seed(5)
    N = 4
    A = _random_A(N, scale=2.0)
    beta_vals = [0.0, 0.5, 2.0, 10.0]
    entropies = [
        coalition_entropy(A, torch.tensor(b), player=0, num_players=N).item()
        for b in beta_vals
    ]
    assert all(e >= -1e-6 for e in entropies)
    # Uniform (beta=0) achieves max entropy = log(2^{N-1}).
    max_entropy = math.log(2 ** (N - 1))
    assert abs(entropies[0] - max_entropy) < 1e-5
    # Entropy strictly decreases as beta grows (for non-degenerate A).
    for prev, nxt in zip(entropies, entropies[1:]):
        assert nxt < prev + 1e-5


# --- Property 7 + 9: differentiability and gradient sign --------------------

def test_nll_gradient_decreases_when_aligning_observed():
    """If we increase A[player, q] for an opponent q that IS in the observed
    coalition, the NLL should decrease (gradient w.r.t. A[player, q] negative)."""
    torch.manual_seed(6)
    N = 4
    A = _random_A(N, scale=0.5).clone().detach().requires_grad_(True)
    beta = torch.tensor(1.0, requires_grad=True)
    observed = (1, 2)  # opponents 1 and 2 finished ahead, opponent 3 didn't
    nll = negative_log_likelihood_coalition(A, beta, observed, player=0, num_players=N)
    nll.backward()
    # Symmetric A's gradient has both [0, q] and [q, 0] entries.
    grad_aligned = A.grad[0, 1].item()  # opponent 1 is in observed coalition
    grad_unaligned = A.grad[0, 3].item()  # opponent 3 is NOT
    # Increasing A[0, 1] should DECREASE NLL: gradient is negative.
    assert grad_aligned < 0, f"expected negative grad on aligned, got {grad_aligned}"
    # Increasing A[0, 3] should INCREASE NLL: gradient is positive.
    assert grad_unaligned > 0, f"expected positive grad on unaligned, got {grad_unaligned}"


# --- Property 8: module forward smoke ---------------------------------------

def test_module_forward_no_batch():
    head = CoalitionHead(feature_dim=8, max_players=4)
    x = torch.randn(8)  # no batch dim
    A, beta = head(x)
    assert A.shape == (4, 4)
    assert beta.shape == ()
