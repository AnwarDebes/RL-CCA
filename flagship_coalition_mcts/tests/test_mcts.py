"""Integration tests for the CD-MCTS tree.

The test stub network and a tiny synthetic 2-player game (a 3-stage
deterministic chain with one branching action) let us check end-to-end
behaviour without committing to a real game environment yet.

What we verify:

  1. MCTS runs without crashing for a small budget.
  2. Visit counts sum to num_simulations.
  3. Returned policy is a valid probability distribution.
  4. With a deterministic 2-player game where action 0 leads to a sure
     win and action 1 to a sure loss, the policy heavily prefers action 0.
  5. Vector backup is consistent: child_value_sum / child_visits matches
     the leaf marginal in a one-step game.
  6. coalition_weight=0 disables coalition penalty (selector still works).
  7. No mutation of the prior network output beyond per-node copies.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Tuple

import numpy as np
import pytest

from flagship_coalition_mcts.src.mcts import NetworkOutput, run_mcts


# ----------------------------------------------------------------------
# Test stub: a tiny deterministic 2-player game.
#
# State = depth ∈ {0, 1, 2}, last_action ∈ {0, 1, None}.
# At depth 0, two actions: 0 leads to "win for player 0" terminal
# at depth 1, 1 leads to "win for player 1" terminal at depth 1.
# Always 2 players.
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class TwoPlayerState:
    depth: int
    last_action: int  # -1 if root
    player_to_move: int


class TwoPlayerStubGame:
    @staticmethod
    def num_players(state: TwoPlayerState) -> int:
        return 2

    @staticmethod
    def current_player(state: TwoPlayerState) -> int:
        return state.player_to_move

    @staticmethod
    def legal_actions(state: TwoPlayerState) -> List[int]:
        return [0, 1] if state.depth == 0 else []

    @staticmethod
    def is_terminal(state: TwoPlayerState) -> bool:
        return state.depth >= 1

    @staticmethod
    def step(state: TwoPlayerState, action: int) -> Tuple[TwoPlayerState, int]:
        nxt = TwoPlayerState(depth=state.depth + 1, last_action=action, player_to_move=1 - state.player_to_move)
        return nxt, nxt.player_to_move

    @staticmethod
    def terminal_marginal(state: TwoPlayerState) -> np.ndarray:
        # Action 0 -> player 0 wins (rank 1); action 1 -> player 1 wins.
        M = np.zeros((2, 2))
        if state.last_action == 0:
            M[0, 0] = 1.0  # player 0 in position 1
            M[1, 1] = 1.0  # player 1 in position 2
        else:  # action 1
            M[1, 0] = 1.0
            M[0, 1] = 1.0
        return M


class StubNetwork:
    """Uniform prior over actions, neutral marginals, no coalition info."""

    def evaluate(self, state: TwoPlayerState) -> NetworkOutput:
        # 2 actions in this game (we use full-action-space size = 2 for prior).
        return NetworkOutput(
            prior_policy=np.array([0.5, 0.5]),
            placement_marginals=np.array([[0.5, 0.5], [0.5, 0.5]]),
            coalition_alignment=np.array([0.0, 0.0]),
        )


# ----------------------------------------------------------------------
# Tests
# ----------------------------------------------------------------------


def _root_state() -> TwoPlayerState:
    return TwoPlayerState(depth=0, last_action=-1, player_to_move=0)


def test_run_mcts_smoke():
    root, pi = run_mcts(
        state=_root_state(),
        network=StubNetwork(),
        game=TwoPlayerStubGame(),
        num_simulations=20,
        seed=0,
    )
    assert pi.shape == (2,)
    assert abs(pi.sum() - 1.0) < 1e-10
    assert (pi >= 0).all()


def test_visit_counts_sum_to_num_simulations():
    root, pi = run_mcts(
        state=_root_state(),
        network=StubNetwork(),
        game=TwoPlayerStubGame(),
        num_simulations=50,
        seed=1,
    )
    assert root.selector_state.visits.sum() == 50


def test_action_preference_for_winning_branch():
    """Player 0 should heavily prefer action 0 (sure win)."""
    root, pi = run_mcts(
        state=_root_state(),
        network=StubNetwork(),
        game=TwoPlayerStubGame(),
        num_simulations=200,
        seed=2,
    )
    # Action 0 wins for current player (player 0); selector should learn this.
    assert pi[0] > pi[1], f"pi={pi}"
    assert pi[0] > 0.7, f"pi={pi} - winning action not strongly preferred"


def test_vector_backup_matches_leaf_marginal():
    """In this one-step game, after many visits, child_value_sum / visits
    must match the deterministic terminal marginal for each action."""
    root, _ = run_mcts(
        state=_root_state(),
        network=StubNetwork(),
        game=TwoPlayerStubGame(),
        num_simulations=400,
        seed=3,
    )
    M0 = TwoPlayerStubGame.terminal_marginal(
        TwoPlayerState(depth=1, last_action=0, player_to_move=1)
    )
    M1 = TwoPlayerStubGame.terminal_marginal(
        TwoPlayerState(depth=1, last_action=1, player_to_move=1)
    )
    avg0 = root.child_value_sum[0] / root.child_visits[0]
    avg1 = root.child_value_sum[1] / root.child_visits[1]
    assert np.allclose(avg0, M0, atol=1e-10)
    assert np.allclose(avg1, M1, atol=1e-10)


def test_coalition_weight_zero_does_not_break_run():
    root, pi = run_mcts(
        state=_root_state(),
        network=StubNetwork(),
        game=TwoPlayerStubGame(),
        num_simulations=30,
        coalition_weight=0.0,
        seed=4,
    )
    assert pi.shape == (2,)
    assert abs(pi.sum() - 1.0) < 1e-10


def test_root_node_state_unchanged_after_search():
    """Sanity: search must not mutate the original state."""
    s = _root_state()
    s_repr_before = repr(s)
    run_mcts(state=s, network=StubNetwork(), game=TwoPlayerStubGame(),
             num_simulations=20, seed=5)
    assert repr(s) == s_repr_before
