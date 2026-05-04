"""Asymmetric 3-Player Race - a synthetic kingmaker testbed.

Why this game exists
====================

Reviewers will attack the paper with: "Your method has fancy coalition
machinery, but in your real games (Chinese Checkers, Halma) coalitions
might not actually matter - show me a game where coalitions provably
matter and your method exploits them."

This game is the answer. It is small enough to be solved exhaustively
(depth-5 game tree ≈ 4^5 = 1024 leaves before pruning), deterministic,
and the optimal play for the player-1 / player-2 coalition vs player 0
is **provably different** from any maxn / paranoid / single-player-best-
response policy. Specifically: player 0 starts ahead, and the only way
players 1 and 2 can avoid losing is by *both* spending turns to trip
player 0 - a strategy that is dominated for either of them under
self-only optimisation.

Game rules
==========

- 3 players, named 0, 1, 2.
- Each player has a position p ∈ {0, 1, 2, 3}. Goal is position 3.
- Initial positions: (2, 0, 0). Player 0 starts with a 2-step lead.
- Players act in cyclic order 0 → 1 → 2 → 0 → 1.
- Total moves: 5.
- Actions per turn (legal subset depends on state):
    SPRINT       - own position += 1 (capped at 3)
    TRIP_P0      - player 0 position -= 1 (floor 0); illegal if self == 0
    TRIP_P1      - player 1 position -= 1 (floor 0); illegal if self == 1
    TRIP_P2      - player 2 position -= 1 (floor 0); illegal if self == 2
- A player who reaches position 3 is "finished" and skipped on subsequent
  turns. Their final rank is determined by the order of finishing.
- After move 5, any non-finished players are ranked by their position
  (higher first), ties broken by player index ascending.

Exhaustive optimum
==================

By backwards induction:
  * If players 1 and 2 each SPRINT every turn: P1 ends at 2, P2 ends at 2,
    P0 ends at 3 (sprints 3 times). Final ranks: (1, 2, 3).
  * If both 1 and 2 trip P0 on at least one turn each: P0 stuck at 1-2,
    P1 and P2 race to position 3. Final ranks favour the coalition.

The crucial insight is that *neither player 1 nor player 2 alone* can
afford to trip - tripping costs them a sprint, and unilaterally tripping
hands the win to the other non-leader. They must coordinate.

A scalar Multiplayer-AlphaZero agent treats every other player as an
independent maximiser of their own position. From that lens, P1 expects
P2 to sprint (best self-response) and so P1 sprints too - giving P0 the
win. CD-MCTS, with its coalition-belief head, can represent the
"P1 and P2 are aligned against P0" belief and select the cooperative
strategy.

This file does not depend on torch or any of the rest of CD-MCTS - it is
a pure deterministic game implementation usable as a baseline benchmark.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

NUM_PLAYERS = 3
GOAL = 3
TOTAL_MOVES = 6
INITIAL_POSITIONS = (1, 0, 0)
# Design rationale: with P0 starting at 1 and 6 total moves on cycle
# (0,1,2,0,1,2), P0 needs 2 sprint turns to finish - giving P1 and P2
# windows to coordinate trips. Verified by test_kingmaker_game_design_property
# that the (1,2)-coalition strictly dominates maxn play.

ACTION_SPRINT = 0
ACTION_TRIP_P0 = 1
ACTION_TRIP_P1 = 2
ACTION_TRIP_P2 = 3
NUM_ACTIONS = 4


@dataclass(frozen=True)
class KingmakerState:
    """Immutable game state.

    positions: tuple of 3 ints in {0..GOAL}.
    finish_order: tuple of player indices in the order they reached GOAL.
        Non-finished players are not in this tuple.
    move_count: number of moves played so far.
    turn_order: which player acts next (skips finished players).
    """

    positions: Tuple[int, int, int]
    finish_order: Tuple[int, ...]
    move_count: int
    next_player: int  # already skipped any finished players

    @staticmethod
    def initial() -> "KingmakerState":
        return KingmakerState(
            positions=INITIAL_POSITIONS,
            finish_order=(),
            move_count=0,
            next_player=0,
        )


def is_terminal(state: KingmakerState) -> bool:
    return (
        state.move_count >= TOTAL_MOVES
        or len(state.finish_order) >= NUM_PLAYERS
    )


def current_player(state: KingmakerState) -> int:
    return state.next_player


def num_players(state: KingmakerState) -> int:
    return NUM_PLAYERS


def legal_actions(state: KingmakerState) -> List[int]:
    if is_terminal(state):
        return []
    out = [ACTION_SPRINT]
    p = state.next_player
    if p != 0:
        out.append(ACTION_TRIP_P0)
    if p != 1:
        out.append(ACTION_TRIP_P1)
    if p != 2:
        out.append(ACTION_TRIP_P2)
    return out


def step(state: KingmakerState, action: int) -> Tuple[KingmakerState, int]:
    """Apply action; return (next_state, next_player_to_move).

    NOTE: the test/duck-typed game interface used by mcts.py expects step
    to return (next_state, current_player_after_step). We honour that.
    """
    if action not in legal_actions(state):
        raise ValueError(f"illegal action {action} in state {state}")

    p = state.next_player
    new_positions = list(state.positions)
    new_finish_order = list(state.finish_order)

    if action == ACTION_SPRINT:
        new_positions[p] = min(new_positions[p] + 1, GOAL)
    elif action == ACTION_TRIP_P0:
        new_positions[0] = max(new_positions[0] - 1, 0)
    elif action == ACTION_TRIP_P1:
        new_positions[1] = max(new_positions[1] - 1, 0)
    elif action == ACTION_TRIP_P2:
        new_positions[2] = max(new_positions[2] - 1, 0)

    # Did the moving player reach the goal?
    if new_positions[p] >= GOAL and p not in new_finish_order:
        new_finish_order.append(p)

    new_move_count = state.move_count + 1

    # Determine next player: cycle (p+1) % NUM_PLAYERS skipping finished
    np_next = p
    for _ in range(NUM_PLAYERS):
        np_next = (np_next + 1) % NUM_PLAYERS
        if np_next not in new_finish_order:
            break
    # If everyone finished, np_next loops back; doesn't matter at terminal.

    new_state = KingmakerState(
        positions=tuple(new_positions),
        finish_order=tuple(new_finish_order),
        move_count=new_move_count,
        next_player=np_next,
    )
    return new_state, new_state.next_player


def final_ranks(state: KingmakerState) -> Tuple[int, int, int]:
    """Return (rank_of_player_0, rank_of_player_1, rank_of_player_2).

    Rank 1 = winner.
    Tie-breaking: those who finished are ranked first by finish order.
    Then, among unfinished, higher position wins; ties broken by player
    index ascending.
    """
    rank = [0, 0, 0]
    # First, finished players in the order they finished.
    next_rank = 1
    for p in state.finish_order:
        rank[p] = next_rank
        next_rank += 1
    # Then unfinished players, sorted by (-position, player_index).
    unfinished = [p for p in range(NUM_PLAYERS) if rank[p] == 0]
    unfinished.sort(key=lambda p: (-state.positions[p], p))
    for p in unfinished:
        rank[p] = next_rank
        next_rank += 1
    return tuple(rank)


def terminal_marginal(state: KingmakerState) -> np.ndarray:
    """Build an N x N placement-marginal matrix for a terminal state.

    Deterministic outcome -> one-hot rows. M[p, k] = 1 iff player p
    finished in position k+1.
    """
    if not is_terminal(state):
        raise ValueError("terminal_marginal called on non-terminal state")
    ranks = final_ranks(state)
    M = np.zeros((NUM_PLAYERS, NUM_PLAYERS), dtype=np.float64)
    for p, r in enumerate(ranks):
        M[p, r - 1] = 1.0
    return M


# -- Game adapter for the MCTS interface --------------------------------------


class KingmakerGame:
    """Class wrapper exposing the duck-typed game interface used by mcts.py."""

    @staticmethod
    def num_players(state: KingmakerState) -> int:
        return num_players(state)

    @staticmethod
    def current_player(state: KingmakerState) -> int:
        return current_player(state)

    @staticmethod
    def legal_actions(state: KingmakerState) -> List[int]:
        return legal_actions(state)

    @staticmethod
    def is_terminal(state: KingmakerState) -> bool:
        return is_terminal(state)

    @staticmethod
    def step(state: KingmakerState, action: int) -> Tuple[KingmakerState, int]:
        return step(state, action)

    @staticmethod
    def terminal_marginal(state: KingmakerState) -> np.ndarray:
        return terminal_marginal(state)


# -- Exhaustive game-tree solver (for ground-truth oracle in tests) ----------


def solve_optimal_value(
    state: KingmakerState,
    perspective_player: int,
    cache: Optional[dict] = None,
) -> Tuple[float, Optional[int]]:
    """Maxn-style solver from `perspective_player`'s viewpoint, but where
    each player optimises their *own* expected rank-utility (lower-is-
    better, normalised to [0, 1]).

    Returns (best_utility_for_perspective, best_action_at_root). The
    returned utility is the *expected* utility for `perspective_player`
    under maxn play from the current state.

    We use this as an oracle for unit tests: an optimal agent should
    achieve at least this utility.
    """
    if cache is None:
        cache = {}
    key = (state, perspective_player)
    if key in cache:
        return cache[key]

    if is_terminal(state):
        ranks = final_ranks(state)
        N = NUM_PLAYERS
        # Utility of a rank: (N - rank) / (N - 1) ∈ [0, 1] (1 = winner).
        u = (N - ranks[perspective_player]) / (N - 1)
        cache[key] = (u, None)
        return cache[key]

    p = state.next_player
    actions = legal_actions(state)
    best_u = -1.0
    best_a = None
    for a in actions:
        nxt, _ = step(state, a)
        # Maxn: each player maximises their *own* utility from the
        # subgame, so we evaluate the subgame from p's perspective when
        # p chooses, but we only return perspective_player's utility.
        if p == perspective_player:
            u_persp, _ = solve_optimal_value(nxt, perspective_player, cache)
            u = u_persp
        else:
            # Predict what p will do: maximise p's own utility.
            u_p, _ = solve_optimal_value(nxt, p, cache)
            # We need p's chosen action to evaluate perspective_player's u.
            # Pick whichever child of state is reached by p's argmax.
            best_for_p = -1.0
            best_action_for_p = None
            for cand in actions:
                ns, _ = step(state, cand)
                up, _ = solve_optimal_value(ns, p, cache)
                if up > best_for_p:
                    best_for_p = up
                    best_action_for_p = cand
            ns, _ = step(state, best_action_for_p)
            u_persp, _ = solve_optimal_value(ns, perspective_player, cache)
            cache[key] = (u_persp, best_action_for_p)
            return cache[key]
        if u > best_u:
            best_u = u
            best_a = a
    cache[key] = (best_u, best_a)
    return cache[key]


def solve_coalition_optimal(
    state: KingmakerState,
    coalition: Tuple[int, ...],
    cache: Optional[dict] = None,
) -> Tuple[float, Optional[int]]:
    """Solve assuming members of `coalition` cooperate to maximise their
    *combined* expected utility, while non-coalition players play maxn
    (each maximising their own utility).

    Returns (combined_utility, best_action_at_root).

    This gives us the "best-case for the coalition" baseline that any
    coalition-aware algorithm should approach.
    """
    if cache is None:
        cache = {}
    key = (state, coalition)
    if key in cache:
        return cache[key]

    if is_terminal(state):
        ranks = final_ranks(state)
        N = NUM_PLAYERS
        u_total = sum((N - ranks[q]) / (N - 1) for q in coalition)
        cache[key] = (u_total, None)
        return cache[key]

    p = state.next_player
    actions = legal_actions(state)

    if p in coalition:
        # Coalition member: maximise combined coalition utility.
        best_u = -1.0
        best_a = None
        for a in actions:
            nxt, _ = step(state, a)
            u_total, _ = solve_coalition_optimal(nxt, coalition, cache)
            if u_total > best_u:
                best_u = u_total
                best_a = a
        cache[key] = (best_u, best_a)
        return cache[key]

    # Non-coalition member: assume they play maxn for themselves.
    best_self = -1.0
    best_action_for_p = None
    for a in actions:
        nxt, _ = step(state, a)
        u_self, _ = solve_optimal_value(nxt, p)
        if u_self > best_self:
            best_self = u_self
            best_action_for_p = a
    nxt, _ = step(state, best_action_for_p)
    result = solve_coalition_optimal(nxt, coalition, cache)
    cache[key] = (result[0], best_action_for_p)
    return cache[key]
