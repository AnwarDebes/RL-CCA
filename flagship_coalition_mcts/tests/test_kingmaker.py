"""Tests for the synthetic kingmaker game.

Property tests:

1. Initial state legal-actions has the right cardinality.
2. step() correctly advances positions, finish_order, move_count, next_player.
3. Terminal at move_count == TOTAL_MOVES.
4. final_ranks is a permutation of {1,2,3}.
5. terminal_marginal is one-hot per row and sums to 1 per column.
6. solve_optimal_value caches and returns consistent values.
7. solve_coalition_optimal returns ≥ sum of solo-utilities for the
   coalition (cooperation can only help).
8. **Game-design check:** the (1, 2)-coalition can achieve a strictly
   better combined utility than maxn play. This is the property that
   makes the game a useful kingmaker testbed.
"""

from __future__ import annotations

import numpy as np
import pytest

from flagship_coalition_mcts.src.games.kingmaker import (
    ACTION_SPRINT,
    ACTION_TRIP_P0,
    ACTION_TRIP_P1,
    ACTION_TRIP_P2,
    KingmakerGame,
    KingmakerState,
    NUM_PLAYERS,
    TOTAL_MOVES,
    final_ranks,
    is_terminal,
    legal_actions,
    solve_coalition_optimal,
    solve_optimal_value,
    step,
    terminal_marginal,
)


def test_initial_state_has_three_legal_actions():
    s = KingmakerState.initial()
    actions = legal_actions(s)
    assert ACTION_SPRINT in actions
    assert ACTION_TRIP_P1 in actions
    assert ACTION_TRIP_P2 in actions
    assert ACTION_TRIP_P0 not in actions  # cannot trip self
    assert len(actions) == 3


def test_step_sprint_advances():
    s = KingmakerState.initial()
    assert s.positions == (1, 0, 0)
    s2, np_next = step(s, ACTION_SPRINT)
    assert s2.positions == (2, 0, 0)
    assert s2.finish_order == ()
    assert s2.move_count == 1
    assert s2.next_player == 1


def test_step_trip_floors_at_zero():
    s = KingmakerState(
        positions=(0, 1, 1), finish_order=(), move_count=0, next_player=1
    )
    s2, _ = step(s, ACTION_TRIP_P0)
    assert s2.positions == (0, 1, 1)  # already at 0


def test_terminal_after_total_moves():
    s = KingmakerState.initial()
    for _ in range(TOTAL_MOVES):
        if is_terminal(s):
            break
        a = legal_actions(s)[0]
        s, _ = step(s, a)
    assert is_terminal(s)


def test_final_ranks_is_permutation():
    s = KingmakerState.initial()
    while not is_terminal(s):
        a = legal_actions(s)[0]
        s, _ = step(s, a)
    ranks = final_ranks(s)
    assert sorted(ranks) == [1, 2, 3]


def test_terminal_marginal_one_hot():
    s = KingmakerState.initial()
    while not is_terminal(s):
        a = legal_actions(s)[0]
        s, _ = step(s, a)
    M = terminal_marginal(s)
    # Each row sums to 1
    assert np.allclose(M.sum(axis=1), 1.0)
    # Each column sums to 1
    assert np.allclose(M.sum(axis=0), 1.0)
    # Each entry is 0 or 1
    assert np.all((M == 0) | (M == 1))


def test_solver_returns_value_in_unit_interval():
    s = KingmakerState.initial()
    u0, _ = solve_optimal_value(s, perspective_player=0)
    u1, _ = solve_optimal_value(s, perspective_player=1)
    u2, _ = solve_optimal_value(s, perspective_player=2)
    for u in [u0, u1, u2]:
        assert 0.0 <= u <= 1.0


def test_coalition_solver_dominates_solo():
    """The (1, 2) coalition should achieve combined utility >= sum of
    the solo (maxn) utilities of players 1 and 2."""
    s = KingmakerState.initial()
    u1_solo, _ = solve_optimal_value(s, perspective_player=1)
    u2_solo, _ = solve_optimal_value(s, perspective_player=2)
    u_coalition, _ = solve_coalition_optimal(s, coalition=(1, 2))
    assert u_coalition >= u1_solo + u2_solo - 1e-9, (
        f"coalition {u_coalition:.3f} < sum of solos {u1_solo + u2_solo:.3f}"
    )


def test_kingmaker_game_design_property():
    """The crucial design property: the (1, 2) coalition's combined
    utility STRICTLY exceeds the maxn-play sum.

    If this fails, the game is not a useful kingmaker testbed and we
    must fix the parameters."""
    s = KingmakerState.initial()
    u1_solo, _ = solve_optimal_value(s, perspective_player=1)
    u2_solo, _ = solve_optimal_value(s, perspective_player=2)
    u_coal, _ = solve_coalition_optimal(s, coalition=(1, 2))
    margin = u_coal - (u1_solo + u2_solo)
    print(f"\n[kingmaker design check] solo (1)={u1_solo:.3f}, solo (2)={u2_solo:.3f}, "
          f"coalition combined={u_coal:.3f}, margin={margin:.3f}")
    assert margin > 0.0, (
        f"coalition has no advantage over maxn - game design broken: "
        f"u1_solo={u1_solo}, u2_solo={u2_solo}, u_coal={u_coal}"
    )


def test_game_adapter_is_consistent():
    """Class adapter should match free functions exactly."""
    s = KingmakerState.initial()
    g = KingmakerGame()
    assert g.num_players(s) == NUM_PLAYERS
    assert g.current_player(s) == 0
    assert g.legal_actions(s) == legal_actions(s)
    assert not g.is_terminal(s)
    s2, np_next = g.step(s, ACTION_SPRINT)
    assert s2.positions == (2, 0, 0)
