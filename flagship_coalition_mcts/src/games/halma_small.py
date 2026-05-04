"""Small Halma - third testbed for the ablation ladder.

Halma is the natural N-player generalisation of Chinese Checkers played
on a square (rather than star-hex) board. Coalition incentives are
**stronger** than in CC because all players' paths cross the centre,
making it the harder testbed where the coalition pillar should yield
larger gains.

Design: 5×5 grid, 3 players, each with 3 pieces.
  - P0 home: top-left 3 cells of the diagonal
  - P1 home: top-right 3 cells
  - P2 home: bottom-centre 3 cells
  - Goal for each player: their opposite home

Movement (per turn): a player may
  - "step": move ONE piece to an empty adjacent cell (8-directional king's
    moves), OR
  - "hop": move ONE piece by jumping over an immediately-adjacent
    occupied cell to land on an empty cell on the far side (single hop;
    chained hopping is disallowed for tractability).

Game ends when a player has all 3 pieces in their goal (winner), or
after MAX_MOVES (50 here) - at which point ranks are by progress.

Why small?
----------
The point is to have a third testbed in the ablation ladder. The full
Halma board (16×16 with 19 pieces) is unnecessary and would dwarf
training time. 5×5 with 3 players × 3 pieces gives a tree depth of
~30-50 plies - small enough to train multiple ablation variants in
parallel on CPU.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

GRID = 5
PIECES_PER_PLAYER = 3
NUM_PLAYERS = 3
MAX_MOVES = 50

# Home zones: (player) -> list of (row, col)
HOME_ZONES = [
    [(0, 0), (0, 1), (1, 0)],            # P0: top-left corner
    [(0, 4), (0, 3), (1, 4)],            # P1: top-right corner
    [(4, 1), (4, 2), (4, 3)],            # P2: bottom-centre
]
# Goal = opposite-corner home.
GOAL_ZONES = [
    [(4, 4), (4, 3), (3, 4)],            # P0 -> bottom-right
    [(4, 0), (4, 1), (3, 0)],            # P1 -> bottom-left
    [(0, 1), (0, 2), (0, 3)],            # P2 -> top-centre
]
# Cells: 25 total, indexed 0..24 = row * 5 + col.
NUM_CELLS = GRID * GRID

# Action space: source_cell × dest_cell. With 25 cells, there are
# 25*25 = 625 candidate (src, dst) pairs; only legal moves are valid.
NUM_ACTIONS = NUM_CELLS * NUM_CELLS

DIRECTIONS_8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1),
                (1, -1), (1, 0), (1, 1)]


def _cell_idx(r: int, c: int) -> int:
    return r * GRID + c


def _cell_rc(idx: int) -> Tuple[int, int]:
    return idx // GRID, idx % GRID


def _action_idx(src: int, dst: int) -> int:
    return src * NUM_CELLS + dst


def _action_decode(a: int) -> Tuple[int, int]:
    return a // NUM_CELLS, a % NUM_CELLS


@dataclass(frozen=True)
class HalmaState:
    """Immutable Halma state.

    pieces: tuple of frozensets of cell-indices, one per player.
    move_count: total moves played.
    next_player: current player (skips finished players).
    finish_order: tuple of player indices who have all pieces in goal,
        in the order they finished.
    """

    pieces: Tuple[Tuple[int, ...], Tuple[int, ...], Tuple[int, ...]]
    move_count: int
    next_player: int
    finish_order: Tuple[int, ...]

    @staticmethod
    def initial() -> "HalmaState":
        return HalmaState(
            pieces=tuple(
                tuple(sorted(_cell_idx(r, c) for r, c in HOME_ZONES[p]))
                for p in range(NUM_PLAYERS)
            ),
            move_count=0,
            next_player=0,
            finish_order=(),
        )


def occupied(state: HalmaState) -> set:
    return set(state.pieces[0] + state.pieces[1] + state.pieces[2])


def is_terminal(state: HalmaState) -> bool:
    return (
        state.move_count >= MAX_MOVES
        or len(state.finish_order) >= NUM_PLAYERS
    )


def current_player(state: HalmaState) -> int:
    return state.next_player


def num_players(state: HalmaState) -> int:
    return NUM_PLAYERS


def _legal_moves_for_piece(state: HalmaState, src: int) -> List[int]:
    occ = occupied(state)
    occ.discard(src)
    sr, sc = _cell_rc(src)
    out = []
    for dr, dc in DIRECTIONS_8:
        # Step
        nr, nc = sr + dr, sc + dc
        if 0 <= nr < GRID and 0 <= nc < GRID:
            ndst = _cell_idx(nr, nc)
            if ndst not in occ:
                out.append(ndst)
        # Hop: jump over an adjacent piece to two-away cell
        nr2, nc2 = sr + 2 * dr, sc + 2 * dc
        if 0 <= nr2 < GRID and 0 <= nc2 < GRID:
            mid = _cell_idx(sr + dr, sc + dc)
            ndst2 = _cell_idx(nr2, nc2)
            if mid in occ and ndst2 not in occ:
                out.append(ndst2)
    return out


def legal_actions(state: HalmaState) -> List[int]:
    if is_terminal(state):
        return []
    p = state.next_player
    out = []
    for src in state.pieces[p]:
        for dst in _legal_moves_for_piece(state, src):
            out.append(_action_idx(src, dst))
    return out


def step(state: HalmaState, action: int) -> Tuple[HalmaState, int]:
    src, dst = _action_decode(action)
    p = state.next_player
    new_pieces = list(state.pieces)
    cur = list(new_pieces[p])
    cur.remove(src)
    cur.append(dst)
    cur.sort()
    new_pieces[p] = tuple(cur)

    # Did p finish?
    new_finish = list(state.finish_order)
    goal_set = set(_cell_idx(r, c) for r, c in GOAL_ZONES[p])
    if p not in new_finish and goal_set.issubset(set(new_pieces[p])):
        new_finish.append(p)

    new_move_count = state.move_count + 1
    np_next = p
    for _ in range(NUM_PLAYERS):
        np_next = (np_next + 1) % NUM_PLAYERS
        if np_next not in new_finish:
            break

    nxt = HalmaState(
        pieces=tuple(new_pieces),
        move_count=new_move_count,
        next_player=np_next,
        finish_order=tuple(new_finish),
    )
    return nxt, nxt.next_player


def _progress(state: HalmaState, player: int) -> int:
    """Number of pieces in player's goal."""
    goal_set = set(_cell_idx(r, c) for r, c in GOAL_ZONES[player])
    return sum(1 for c in state.pieces[player] if c in goal_set)


def final_ranks(state: HalmaState) -> Tuple[int, int, int]:
    rank = [0] * NUM_PLAYERS
    next_rank = 1
    for p in state.finish_order:
        rank[p] = next_rank
        next_rank += 1
    unfinished = [p for p in range(NUM_PLAYERS) if rank[p] == 0]
    unfinished.sort(key=lambda p: (-_progress(state, p), p))
    for p in unfinished:
        rank[p] = next_rank
        next_rank += 1
    return tuple(rank)


def terminal_marginal(state: HalmaState) -> np.ndarray:
    if not is_terminal(state):
        raise ValueError("terminal_marginal called on non-terminal state")
    ranks = final_ranks(state)
    M = np.zeros((NUM_PLAYERS, NUM_PLAYERS), dtype=np.float64)
    for p, r in enumerate(ranks):
        M[p, r - 1] = 1.0
    return M


class HalmaSmallGame:
    @staticmethod
    def num_players(state):
        return num_players(state)

    @staticmethod
    def current_player(state):
        return current_player(state)

    @staticmethod
    def legal_actions(state):
        return legal_actions(state)

    @staticmethod
    def is_terminal(state):
        return is_terminal(state)

    @staticmethod
    def step(state, action):
        return step(state, action)

    @staticmethod
    def terminal_marginal(state):
        return terminal_marginal(state)

    @staticmethod
    def initial():
        return HalmaState.initial()


def state_to_features(state: HalmaState) -> np.ndarray:
    """Flat feature vector for an MLP encoder.

    NUM_CELLS × (NUM_PLAYERS + 1) = 25 × 4 = 100 occupancy features
    + N current-player one-hot (3) + finish-status bits (3) + move-count
    normalised (1) = 107 dims.
    """
    feats = np.zeros(NUM_CELLS * (NUM_PLAYERS + 1) + NUM_PLAYERS + NUM_PLAYERS + 1, dtype=np.float32)
    occ = [set(state.pieces[p]) for p in range(NUM_PLAYERS)]
    for c in range(NUM_CELLS):
        for p in range(NUM_PLAYERS):
            if c in occ[p]:
                feats[c * (NUM_PLAYERS + 1) + p] = 1.0
                break
        else:
            feats[c * (NUM_PLAYERS + 1) + NUM_PLAYERS] = 1.0  # empty
    # Current player one-hot
    base = NUM_CELLS * (NUM_PLAYERS + 1)
    feats[base + state.next_player] = 1.0
    # Finish status
    base += NUM_PLAYERS
    for p in state.finish_order:
        feats[base + p] = 1.0
    # Move count
    base += NUM_PLAYERS
    feats[base] = state.move_count / MAX_MOVES
    return feats
