"""EXP-IX-style no-regret action selector for CD-MCTS.

This is the third novel pillar of CD-MCTS. It replaces PUCT with a Hedge /
EXP-IX-style sampling rule whose theoretical anchor is a coarse correlated
equilibrium (CCE) of the per-state action meta-game.

Why we use EXP-IX
-----------------
Classical AlphaZero/PUCT (Silver 2017) has no equilibrium-theoretic anchor
in N-player non-zero-sum settings. UCB-style criteria converge to the
single-player optimal arm, which is a meaningful target only in 2-player
zero-sum (Nash) or single-agent (best response).

For N-player non-zero-sum games, the appropriate convergence target is a
coarse correlated equilibrium (CCE). Sokota et al. (and a line of CFR/MCTS
work) show that **no-regret learners** at each tree node converge to a
CCE of the induced extensive-form game. EXP-IX (Neu 2015) is a no-regret
algorithm with O(sqrt(T)) regret in the bandit setting and is well-suited
to the partial-observability of MCTS where only the sampled action's
outcome is known.

Adaptation for CD-MCTS
----------------------
We additionally condition the sampling distribution on the coalition
belief. Under coalition C against the current player, the effective
"opponent strength" felt by action a is

    eff_strength(a; C) = sum_{q in C} placement_marginal(q, top-1 | s_a)

i.e., the probability that some opponent in the coalition wins from the
state reached by action a. The selector mixes this into the per-action
score with weight controlled by `coalition_weight`.

The bound: for the standard EXP-IX (no coalition mixing), regret is
O(sqrt(T log K)) per node, which by the Sokota-style argument implies
empirical CCE-gap -> 0 at rate O(1/sqrt(T)). The coalition-weighted
variant retains this rate when the coalition belief is exogenous (e.g.,
fixed during a search), which is the case in our implementation since
the belief is computed once per node from the encoder.

Public API
----------
    SelectorState: per-node mutable state (cumulative regrets, visits).
    select_action(): returns an action to expand at this visit.
    update_regrets(): updates regrets given a single sampled rollout's
        per-action Q estimate.
    policy_at_root(): the visit-count-weighted policy used as the
        AlphaZero training target.

Unit-test obligations: regret-bound sanity (regret per round is
O(sqrt(log K))) on a synthetic stochastic-bandit problem, recovery of
PUCT-like behaviour in a degenerate regime, equilibrium of a known
matrix game.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class SelectorState:
    """Per-node bookkeeping for EXP-IX selection.

    Attributes:
        num_actions: number of legal children at this node.
        cumulative_regret: ndarray (num_actions,) of cumulative regret.
        visits: ndarray (num_actions,) of visit counts.
        prior: ndarray (num_actions,) of prior policy from network.
            Used as the initial sampling distribution before any visits;
            also mixed into the IX exploration.
        eta: learning rate for Hedge updates. Set to sqrt(log K / T)
            where T is the budget; we use a running eta = sqrt(log K /
            max(1, total_visits)) to be anytime.
        gamma: IX exploration parameter, in (0, 0.5].
        coalition_weight: how much the coalition prior shifts the
            sampling distribution. 0 disables coalition awareness.
    """

    num_actions: int
    cumulative_regret: np.ndarray = field(default=None)
    visits: np.ndarray = field(default=None)
    prior: np.ndarray = field(default=None)
    eta: float = 0.5
    gamma: float = 0.05
    coalition_weight: float = 0.5

    def __post_init__(self) -> None:
        K = self.num_actions
        if self.cumulative_regret is None:
            self.cumulative_regret = np.zeros(K, dtype=np.float64)
        if self.visits is None:
            self.visits = np.zeros(K, dtype=np.int64)
        if self.prior is None:
            self.prior = np.full(K, 1.0 / K, dtype=np.float64)
        assert self.cumulative_regret.shape == (K,)
        assert self.visits.shape == (K,)
        assert self.prior.shape == (K,)
        assert 0.0 < self.gamma <= 0.5
        assert self.eta > 0.0


def _stable_softmax(x: np.ndarray) -> np.ndarray:
    z = x - x.max()
    e = np.exp(z)
    return e / e.sum()


def hedge_distribution(state: SelectorState) -> np.ndarray:
    """Hedge sampling distribution: pi(a) ∝ exp(eta * R_a) blended with the prior.

    We use the convention pi = (1 - alpha) * Hedge(eta * R) + alpha * prior
    with alpha = exp(-total_visits / num_actions) so that early in the
    search the prior dominates and Hedge takes over as visits accumulate.

    This is one common way to recover PUCT-like behaviour in the small-T
    regime while preserving asymptotic Hedge guarantees.
    """
    K = state.num_actions
    T = int(state.visits.sum())
    # No information yet -> trust prior exactly. This is also the
    # AlphaZero convention.
    if T == 0:
        return state.prior.copy()
    # anytime learning rate
    eta_eff = state.eta * math.sqrt(math.log(K + 1) / T)
    hedge = _stable_softmax(eta_eff * state.cumulative_regret)
    # As T grows, Hedge takes over from the prior.
    alpha = math.exp(-T / max(1, K))
    return (1.0 - alpha) * hedge + alpha * state.prior


def ix_distribution(state: SelectorState) -> np.ndarray:
    """The IX-mixed sampling distribution: gamma-uniform mixed with Hedge.

    pi_IX(a) = (1 - gamma) * pi_Hedge(a) + gamma / K
    """
    K = state.num_actions
    pi = hedge_distribution(state)
    return (1.0 - state.gamma) * pi + state.gamma / K


def select_action(
    state: SelectorState,
    coalition_score: Optional[np.ndarray] = None,
    rng: Optional[np.random.Generator] = None,
) -> int:
    """Sample an action from the IX-mixed Hedge distribution.

    Args:
        state: SelectorState
        coalition_score: optional ndarray (num_actions,) of per-action
            scores from the coalition-belief head. Subtracted from the
            sampling logits with weight `state.coalition_weight`. Higher
            coalition_score = action looks worse under inferred coalitions.
        rng: optional numpy RNG.

    Returns:
        Sampled action index.
    """
    if rng is None:
        rng = np.random.default_rng()
    pi = ix_distribution(state)
    if coalition_score is not None and state.coalition_weight > 0.0:
        # Soft penalty by coalition score: re-shape pi by element-wise
        # multiplication with exp(-w * coalition_score), then renormalise.
        penalty = np.exp(-state.coalition_weight * coalition_score)
        pi = pi * penalty
        pi /= pi.sum()
    return int(rng.choice(len(pi), p=pi))


def update_regrets(state: SelectorState, action: int, q_values: np.ndarray) -> None:
    """Update cumulative regret given the per-action Q-estimates from this rollout.

    Standard external-regret update: r_a += Q_a - Q_{action_played}.
    We use the all-action variant (assumes the simulator can give Q_a for
    every legal action - which we do via the value head at the expanded
    leaf, similar to how AlphaZero uses V at the leaf to bootstrap all
    siblings' priors).
    """
    if q_values.shape != (state.num_actions,):
        raise ValueError(f"q_values shape {q_values.shape} != ({state.num_actions},)")
    state.cumulative_regret += q_values - q_values[action]
    state.visits[action] += 1


def policy_at_root(state: SelectorState, temperature: float = 1.0) -> np.ndarray:
    """Visit-count-weighted policy for AlphaZero-style training targets.

    Same as standard AlphaZero. Temperature tau controls greediness:
    pi_train(a) ∝ visits(a)^{1/tau}.
    """
    if temperature <= 0:
        out = np.zeros_like(state.visits, dtype=np.float64)
        out[state.visits.argmax()] = 1.0
        return out
    counts = state.visits.astype(np.float64)
    if counts.sum() == 0:
        return state.prior.copy()
    powered = counts ** (1.0 / temperature)
    return powered / powered.sum()
