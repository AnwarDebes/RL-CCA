"""Tests for the small Halma testbed.

Verifies the game implementation is correct: no piece teleporting,
goal detection works, terminal_marginal is well-formed, action space
encoding round-trips.
"""

from __future__ import annotations

import numpy as np
import pytest

from flagship_coalition_mcts.src.games.halma_small import (
    GRID,
    HalmaSmallGame,
    HalmaState,
    NUM_ACTIONS,
    NUM_CELLS,
    NUM_PLAYERS,
    PIECES_PER_PLAYER,
    _action_decode,
    _action_idx,
    _cell_idx,
    final_ranks,
    is_terminal,
    legal_actions,
    state_to_features,
    step,
    terminal_marginal,
)


def test_initial_state_pieces_in_homes():
    s = HalmaState.initial()
    for p in range(NUM_PLAYERS):
        assert len(s.pieces[p]) == PIECES_PER_PLAYER
    # No piece overlap
    all_pieces = sum(s.pieces, ())
    assert len(all_pieces) == len(set(all_pieces))


def test_legal_actions_nonempty_at_start():
    s = HalmaState.initial()
    legal = legal_actions(s)
    assert len(legal) > 0
    for a in legal:
        src, dst = _action_decode(a)
        assert 0 <= src < NUM_CELLS
        assert 0 <= dst < NUM_CELLS


def test_action_idx_round_trip():
    for src in range(NUM_CELLS):
        for dst in range(NUM_CELLS):
            a = _action_idx(src, dst)
            s, d = _action_decode(a)
            assert (s, d) == (src, dst)
            assert 0 <= a < NUM_ACTIONS


def test_step_moves_only_one_piece():
    s = HalmaState.initial()
    legal = legal_actions(s)
    a = legal[0]
    src, dst = _action_decode(a)
    s2, _ = step(s, a)
    # Source no longer in pieces
    assert src not in s2.pieces[0]
    # Destination is in pieces
    assert dst in s2.pieces[0]
    # Total piece count preserved
    assert len(sum(s2.pieces, ())) == len(sum(s.pieces, ()))


def test_terminal_after_max_moves():
    s = HalmaState.initial()
    while not is_terminal(s):
        legal = legal_actions(s)
        if not legal:
            break
        s, _ = step(s, legal[0])
    assert is_terminal(s)


def test_final_ranks_permutation():
    s = HalmaState.initial()
    while not is_terminal(s):
        legal = legal_actions(s)
        if not legal:
            break
        s, _ = step(s, legal[0])
    ranks = final_ranks(s)
    assert sorted(ranks) == list(range(1, NUM_PLAYERS + 1))


def test_terminal_marginal_one_hot():
    s = HalmaState.initial()
    while not is_terminal(s):
        legal = legal_actions(s)
        if not legal:
            break
        s, _ = step(s, legal[0])
    M = terminal_marginal(s)
    assert M.shape == (NUM_PLAYERS, NUM_PLAYERS)
    assert np.allclose(M.sum(axis=1), 1.0)
    assert np.allclose(M.sum(axis=0), 1.0)
    assert np.all((M == 0) | (M == 1))


def test_state_to_features_shape():
    s = HalmaState.initial()
    f = state_to_features(s)
    expected_dim = NUM_CELLS * (NUM_PLAYERS + 1) + NUM_PLAYERS + NUM_PLAYERS + 1
    assert f.shape == (expected_dim,)
    # Cells with pieces should set the corresponding player flag.
    for p in range(NUM_PLAYERS):
        for cell in s.pieces[p]:
            assert f[cell * (NUM_PLAYERS + 1) + p] == 1.0


def test_class_adapter_consistency():
    s = HalmaState.initial()
    g = HalmaSmallGame()
    assert g.num_players(s) == NUM_PLAYERS
    assert g.current_player(s) == 0
    assert g.legal_actions(s) == legal_actions(s)
    legal = legal_actions(s)
    s2_a, _ = step(s, legal[0])
    s2_b, _ = g.step(s, legal[0])
    assert s2_a == s2_b
