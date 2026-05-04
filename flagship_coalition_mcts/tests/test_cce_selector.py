"""Unit tests for the EXP-IX CCE-regret selector.

Tests verify:

1. Distributions are valid probability vectors.
2. Initial Hedge distribution = prior (no information yet).
3. Single-arm bandit: regret converges to O(sqrt(T)) on a synthetic 4-arm
   stochastic bandit problem.
4. PUCT-like recovery: with high prior on one action and coalition_weight=0,
   that action is selected disproportionately early on.
5. Coalition_score actually shifts the distribution.
6. Visit counts increase monotonically.
7. policy_at_root with T=0 returns the prior; with very low temperature
   returns a near-one-hot at most-visited.
8. Equilibrium of a 2x2 matrix game under self-play between two
   selectors converges toward the analytic Nash mixed strategy.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from flagship_coalition_mcts.src.cce_selector import (
    SelectorState,
    hedge_distribution,
    ix_distribution,
    policy_at_root,
    select_action,
    update_regrets,
)


# --- Property 1: valid distributions ----------------------------------------

def test_hedge_and_ix_distributions_valid():
    rng = np.random.default_rng(0)
    K = 5
    state = SelectorState(num_actions=K, prior=rng.dirichlet(np.ones(K)))
    state.cumulative_regret = rng.normal(size=K) * 2.0
    state.visits = rng.integers(0, 10, size=K)
    pi_h = hedge_distribution(state)
    pi_i = ix_distribution(state)
    for pi in [pi_h, pi_i]:
        assert pi.shape == (K,)
        assert np.all(pi >= 0)
        assert abs(pi.sum() - 1.0) < 1e-10


# --- Property 2: zero regret + zero visits => Hedge = prior -----------------

def test_zero_visits_yields_prior():
    K = 4
    prior = np.array([0.5, 0.2, 0.2, 0.1])
    state = SelectorState(num_actions=K, prior=prior)
    pi = hedge_distribution(state)
    # Hedge with zero regret = uniform; blended with prior at alpha=exp(0)=1
    # so output = prior exactly.
    assert np.allclose(pi, prior, atol=1e-10)


# --- Property 3: regret O(sqrt(T)) on stochastic bandit ---------------------

def test_regret_sqrt_t_on_stochastic_bandit():
    """4-arm bandit; one arm has higher mean. Cumulative regret of a
    Hedge selector should grow sublinearly (O(sqrt(T))).

    We measure pseudo-regret = T * mu_max - sum_t mu_{a_t}."""
    rng = np.random.default_rng(42)
    K = 4
    means = np.array([0.2, 0.5, 0.3, 0.8])  # arm 3 is best
    mu_max = means.max()
    T = 4000
    state = SelectorState(num_actions=K, eta=1.0, gamma=0.05)
    cumulative_pseudo_regret = 0.0
    for t in range(T):
        a = select_action(state, rng=rng)
        # observe full Q-vector from "perfect oracle" (means + small noise).
        # Use noisy mean to update regrets.
        q = means + rng.normal(scale=0.1, size=K)
        update_regrets(state, a, q)
        cumulative_pseudo_regret += mu_max - means[a]
    # O(sqrt(T)) bound: regret/sqrt(T) should be a small constant.
    rate = cumulative_pseudo_regret / math.sqrt(T)
    # Constant depends on noise level + IX gamma; demanding < 5.0 is loose.
    assert rate < 5.0, f"rate {rate:.3f} too high"
    # Best arm should be most visited
    assert state.visits.argmax() == 3, f"best arm not most visited: {state.visits}"


# --- Property 4: prior dominates early selection ----------------------------

def test_high_prior_dominates_early_selection():
    rng = np.random.default_rng(1)
    K = 4
    prior = np.array([0.85, 0.05, 0.05, 0.05])
    state = SelectorState(num_actions=K, prior=prior)
    counts = np.zeros(K)
    for _ in range(100):
        # No regret update - fresh state each "trial". Mimic small-T regime.
        state.cumulative_regret = np.zeros(K)
        state.visits = np.zeros(K, dtype=np.int64)
        a = select_action(state, rng=rng)
        counts[a] += 1
    # Action 0 should be picked > 70 / 100 times under the prior+IX mixing.
    assert counts[0] > 70


# --- Property 5: coalition score shifts the distribution --------------------

def test_coalition_score_shifts_distribution():
    rng = np.random.default_rng(2)
    K = 3
    state = SelectorState(num_actions=K, coalition_weight=2.0)
    coal_score = np.array([0.0, 5.0, 0.0])  # action 1 is "bad" under coalition
    counts = np.zeros(K)
    for _ in range(2000):
        a = select_action(state, coalition_score=coal_score, rng=rng)
        counts[a] += 1
    # Action 1 should be heavily down-weighted.
    assert counts[1] < counts[0] / 5
    assert counts[1] < counts[2] / 5


# --- Property 6: visits monotone --------------------------------------------

def test_visits_monotone_after_updates():
    K = 3
    state = SelectorState(num_actions=K)
    update_regrets(state, action=1, q_values=np.array([0.0, 1.0, 0.0]))
    update_regrets(state, action=1, q_values=np.array([0.0, 1.0, 0.0]))
    update_regrets(state, action=0, q_values=np.array([1.0, 0.0, 0.0]))
    assert state.visits.tolist() == [1, 2, 0]


# --- Property 7: policy_at_root edge cases ----------------------------------

def test_policy_at_root_with_no_visits_returns_prior():
    K = 3
    prior = np.array([0.6, 0.2, 0.2])
    state = SelectorState(num_actions=K, prior=prior)
    pi = policy_at_root(state, temperature=1.0)
    assert np.allclose(pi, prior)


def test_policy_at_root_low_temperature_collapses_to_argmax():
    K = 4
    state = SelectorState(num_actions=K)
    state.visits = np.array([5, 50, 10, 3])
    pi = policy_at_root(state, temperature=0.01)
    assert pi.argmax() == 1
    assert pi[1] > 0.999


# --- Property 8: matrix-game self-play converges to Nash ---------------------

def test_two_selectors_self_play_matching_pennies_converges_to_nash():
    """Matching Pennies matrix game.
    P1 plays {Heads, Tails}, P2 plays {Heads, Tails}.
    Payoff to P1: H,H=+1; H,T=-1; T,H=-1; T,T=+1 (zero-sum).
    Nash: both players play (0.5, 0.5).
    Two EXP-IX selectors against each other should converge to that."""
    rng = np.random.default_rng(7)
    payoff_p1 = np.array([[1.0, -1.0], [-1.0, 1.0]])
    s1 = SelectorState(num_actions=2, eta=1.0, gamma=0.05)
    s2 = SelectorState(num_actions=2, eta=1.0, gamma=0.05)
    T = 5000
    for _ in range(T):
        a1 = select_action(s1, rng=rng)
        a2 = select_action(s2, rng=rng)
        # P1 sees Q-values for each row given P2's mixed strategy estimate.
        # Use the OPPONENT's IX distribution as an estimate.
        q1 = payoff_p1 @ ix_distribution(s2)
        q2 = -payoff_p1.T @ ix_distribution(s1)
        update_regrets(s1, a1, q1)
        update_regrets(s2, a2, q2)
    # Empirical visit frequencies should be near 0.5.
    f1 = s1.visits / s1.visits.sum()
    f2 = s2.visits / s2.visits.sum()
    assert abs(f1[0] - 0.5) < 0.07, f"P1 not at Nash: {f1}"
    assert abs(f2[0] - 0.5) < 0.07, f"P2 not at Nash: {f2}"
