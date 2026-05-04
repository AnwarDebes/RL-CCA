"""Tests for the CMAZ MCTS module.

Uses a tiny stub game (single decision, K=2 components) so the test runs
in well under a second.

Verifies:
1. run_mcts_cmaz returns a valid root + policy.
2. Visit counts sum to num_simulations.
3. Vector backup is consistent: child_value_sum / visits matches the
   leaf component_value for terminal children in a one-step game.
4. Inference-time mixer override changes the chosen action.
5. Mixer monotonicity drives action preference: when component k is
   highly weighted and an action has high v_k, that action is preferred.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
import pytest

from decomposed_mcts.src.cmaz_mcts import (
    CMAZNetworkOutput,
    run_mcts_cmaz,
)


# --- Tiny stub game ---------------------------------------------------------


@dataclass(frozen=True)
class StubState:
    depth: int
    last_action: int


class StubGame:
    @staticmethod
    def num_players(state: StubState) -> int:
        return 1  # single-player; CMAZ is per-component, not per-player

    @staticmethod
    def current_player(state: StubState) -> int:
        return 0

    @staticmethod
    def legal_actions(state: StubState) -> List[int]:
        return [0, 1] if state.depth == 0 else []

    @staticmethod
    def is_terminal(state: StubState) -> bool:
        return state.depth >= 1

    @staticmethod
    def step(state: StubState, action: int):
        return StubState(depth=1, last_action=action), 0


class StubNetwork:
    """Returns deterministic outputs.

    For terminal states, action 0 yields component_values [1.0, 0.0],
    action 1 yields [0.0, 1.0].
    """

    feature_dim = 4

    def evaluate_cmaz(self, state: StubState) -> CMAZNetworkOutput:
        return CMAZNetworkOutput(
            prior_policy=np.array([0.5, 0.5]),
            component_values=np.array([0.5, 0.5]),
            encoder_features=np.zeros(self.feature_dim, dtype=np.float64),
        )

    def terminal_components(self, state: StubState) -> np.ndarray:
        if state.last_action == 0:
            return np.array([1.0, 0.0])
        else:
            return np.array([0.0, 1.0])


# --- Mixer factories --------------------------------------------------------


def fixed_weight_mixer(weights: np.ndarray):
    """Return a mixer-apply callable that uses fixed weights and zero bias."""
    def apply(v: np.ndarray, _features: np.ndarray) -> float:
        w = weights / weights.sum()
        return float((w * v).sum())
    return apply


# --- Tests ------------------------------------------------------------------


def test_run_mcts_cmaz_smoke():
    mixer = fixed_weight_mixer(np.array([0.5, 0.5]))
    root, pi = run_mcts_cmaz(
        state=StubState(depth=0, last_action=-1),
        network=StubNetwork(),
        game=StubGame(),
        mixer_apply=mixer,
        num_simulations=20,
    )
    assert pi.shape == (2,)
    assert abs(pi.sum() - 1.0) < 1e-9


def test_visits_sum_to_simulations():
    mixer = fixed_weight_mixer(np.array([0.5, 0.5]))
    root, _ = run_mcts_cmaz(
        state=StubState(depth=0, last_action=-1),
        network=StubNetwork(),
        game=StubGame(),
        mixer_apply=mixer,
        num_simulations=37,
    )
    assert root.child_visits.sum() == 37


def test_vector_backup_matches_terminal_components():
    mixer = fixed_weight_mixer(np.array([0.5, 0.5]))
    root, _ = run_mcts_cmaz(
        state=StubState(depth=0, last_action=-1),
        network=StubNetwork(),
        game=StubGame(),
        mixer_apply=mixer,
        num_simulations=400,
    )
    avg0 = root.child_value_sum[0] / root.child_visits[0]
    avg1 = root.child_value_sum[1] / root.child_visits[1]
    assert np.allclose(avg0, [1.0, 0.0], atol=1e-10)
    assert np.allclose(avg1, [0.0, 1.0], atol=1e-10)


def test_mixer_weighting_drives_action_preference():
    """When the mixer puts all weight on component 0, action 0 (which
    yields component value 1.0 in dim 0) should be preferred."""
    weighted_to_0 = fixed_weight_mixer(np.array([1.0, 0.0]))
    root_a, pi_a = run_mcts_cmaz(
        state=StubState(depth=0, last_action=-1),
        network=StubNetwork(),
        game=StubGame(),
        mixer_apply=weighted_to_0,
        num_simulations=100,
    )
    weighted_to_1 = fixed_weight_mixer(np.array([0.0, 1.0]))
    root_b, pi_b = run_mcts_cmaz(
        state=StubState(depth=0, last_action=-1),
        network=StubNetwork(),
        game=StubGame(),
        mixer_apply=weighted_to_1,
        num_simulations=100,
    )
    # With mixer favouring component 0, action 0 should dominate.
    assert pi_a[0] > pi_a[1], f"weighted_to_0 -> pi={pi_a}"
    # With mixer favouring component 1, action 1 should dominate.
    assert pi_b[1] > pi_b[0], f"weighted_to_1 -> pi={pi_b}"


def test_inference_time_override_changes_decision():
    """Same network, but two different mixers produce different policies.
    This is the core CMAZ killer property."""
    weighted_to_0 = fixed_weight_mixer(np.array([1.0, 0.0]))
    weighted_to_1 = fixed_weight_mixer(np.array([0.0, 1.0]))
    _, pi_a = run_mcts_cmaz(
        state=StubState(depth=0, last_action=-1),
        network=StubNetwork(), game=StubGame(),
        mixer_apply=weighted_to_0,
        num_simulations=80,
    )
    _, pi_b = run_mcts_cmaz(
        state=StubState(depth=0, last_action=-1),
        network=StubNetwork(), game=StubGame(),
        mixer_apply=weighted_to_1,
        num_simulations=80,
    )
    # The two policies must differ - the same network adapts to a
    # different utility at inference time.
    assert not np.allclose(pi_a, pi_b, atol=0.05), (
        "inference-time override did not change the policy"
    )
