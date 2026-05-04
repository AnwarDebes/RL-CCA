"""Head-to-head agent evaluator with bootstrap-CI Elo.

Production tool for comparing any two CD-MCTS-compatible agents on
Chinese Checkers (or any of the three testbed games). Reports:

  * raw win/place/show counts per seat
  * Elo gap with 95% bootstrap confidence interval
  * statistical significance (p-value via permutation test)

Used to populate the paper's results tables. The "agent" interface is
duck-typed: any callable that maps state -> action_idx (over legal
actions) suffices. CD-MCTS, CMAZ, NN-CCE, scalar-PUCT all conform.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Tuple

import numpy as np


Agent = Callable[[Any], int]   # state -> action index over LEGAL actions


@dataclass
class HeadToHeadResult:
    num_games: int
    seat_assignments: List[Tuple[int, ...]]
    rank_per_game: List[Tuple[int, ...]]
    win_counts_per_agent: List[int]
    name_a: str
    name_b: str

    def expected_score_a(self) -> float:
        """Expected score for A (1 = win, 0.5 = tie, 0 = loss). Generalised
        to N-player by using rank-based score: 1 / rank with N players →
        score_p = (N - rank_p) / (N - 1).
        """
        if self.num_games == 0:
            return 0.5
        total = 0.0
        for assign, ranks in zip(self.seat_assignments, self.rank_per_game):
            N = len(assign)
            scores_a = [
                (N - ranks[p]) / max(1, N - 1)
                for p, who in enumerate(assign) if who == 0
            ]
            scores_b = [
                (N - ranks[p]) / max(1, N - 1)
                for p, who in enumerate(assign) if who == 1
            ]
            # If A has multiple seats, average; same for B.
            if scores_a and scores_b:
                total += np.mean(scores_a)
        return total / self.num_games

    def elo_gap(self) -> float:
        """Elo gap of A relative to B: positive = A is stronger.

        Uses standard Elo conversion: gap_elo = -400 * log10(1/p - 1)
        where p is A's expected score.
        """
        p = max(1e-6, min(1 - 1e-6, self.expected_score_a()))
        return -400.0 * math.log10(1.0 / p - 1.0)

    def bootstrap_elo_ci(self, n_resamples: int = 1000, seed: int = 0) -> Tuple[float, float]:
        """95% CI on the Elo gap via paired bootstrap resampling."""
        rng = np.random.default_rng(seed)
        if self.num_games < 2:
            return (-9999.0, 9999.0)
        gaps = []
        for _ in range(n_resamples):
            idx = rng.integers(0, self.num_games, self.num_games)
            assigns = [self.seat_assignments[i] for i in idx]
            ranks = [self.rank_per_game[i] for i in idx]
            sub = HeadToHeadResult(
                num_games=self.num_games,
                seat_assignments=assigns,
                rank_per_game=ranks,
                win_counts_per_agent=self.win_counts_per_agent,
                name_a=self.name_a, name_b=self.name_b,
            )
            gaps.append(sub.elo_gap())
        gaps.sort()
        lo = gaps[int(0.025 * n_resamples)]
        hi = gaps[int(0.975 * n_resamples)]
        return float(lo), float(hi)

    def summary(self) -> str:
        lo, hi = self.bootstrap_elo_ci()
        return (
            f"{self.name_a} vs {self.name_b}: "
            f"games={self.num_games}, "
            f"E[score_a]={self.expected_score_a():.3f}, "
            f"Elo gap = {self.elo_gap():+.0f} "
            f"(95% CI [{lo:+.0f}, {hi:+.0f}])"
        )


def _seat_pattern_for_n(num_players: int, num_a_seats: int) -> List[int]:
    """Returns a fixed 'who plays which seat' pattern.

    For a 2-player game, returns [0, 1] (A in seat 0, B in seat 1).
    For 3-player games with num_a_seats=2, returns [0, 0, 1] (A coalition).
    """
    pat = [0] * num_a_seats + [1] * (num_players - num_a_seats)
    return pat


def play_game_with_seats(
    game,
    initial_state_fn,
    seats: List[int],   # length num_players, 0 or 1 indicating which agent
    agent_a: Agent,
    agent_b: Agent,
    rng: np.random.Generator,
) -> Tuple[int, ...]:
    """Play one game with the given seat assignment; return ranks tuple."""
    state = initial_state_fn()
    while not game.is_terminal(state):
        legal = game.legal_actions(state)
        if not legal:
            break
        cp = game.current_player(state)
        who = seats[cp]
        agent = agent_a if who == 0 else agent_b
        action_idx = agent(state)
        action = legal[action_idx]
        state, _ = game.step(state, action)
    if not game.is_terminal(state):
        return tuple([1] * len(seats))  # tie all if not terminal
    M = game.terminal_marginal(state)
    N = M.shape[0]
    return tuple(int(M[p].argmax()) + 1 for p in range(N))


def head_to_head(
    game,
    initial_state_fn,
    num_players: int,
    agent_a: Agent,
    agent_b: Agent,
    name_a: str = "A",
    name_b: str = "B",
    num_games: int = 30,
    num_a_seats: int = 1,
    seed: int = 0,
) -> HeadToHeadResult:
    """Play num_games. Each game: A occupies num_a_seats seats, B the rest.

    Seat permutation is randomised per game to control for first-player
    advantage.
    """
    rng = np.random.default_rng(seed)
    pattern = _seat_pattern_for_n(num_players, num_a_seats)
    seat_assignments = []
    rank_per_game = []
    for g in range(num_games):
        # Random permutation of the pattern
        perm = list(range(num_players))
        rng.shuffle(perm)
        assigned = [pattern[i] for i in perm]
        ranks = play_game_with_seats(
            game, initial_state_fn, assigned, agent_a, agent_b, rng,
        )
        seat_assignments.append(tuple(assigned))
        rank_per_game.append(ranks)
    win_a = sum(1 for assigned, ranks in zip(seat_assignments, rank_per_game)
                if any(r == 1 for who, r in zip(assigned, ranks) if who == 0))
    win_b = sum(1 for assigned, ranks in zip(seat_assignments, rank_per_game)
                if any(r == 1 for who, r in zip(assigned, ranks) if who == 1))
    return HeadToHeadResult(
        num_games=num_games,
        seat_assignments=seat_assignments,
        rank_per_game=rank_per_game,
        win_counts_per_agent=[win_a, win_b],
        name_a=name_a, name_b=name_b,
    )


def permutation_test(
    result_a_better: HeadToHeadResult,
    n_resamples: int = 2000,
    seed: int = 0,
) -> float:
    """Permutation test for the null hypothesis that agent A and B are
    equivalent. Returns a p-value (lower = more evidence A != B).
    """
    rng = np.random.default_rng(seed)
    observed = result_a_better.expected_score_a()
    null_ge = 0
    n = result_a_better.num_games
    if n == 0:
        return 1.0
    for _ in range(n_resamples):
        # Under null: swap A/B labels in each game with prob 0.5
        flips = rng.integers(0, 2, size=n)
        swapped_assigns = [
            tuple(1 - x if flips[i] else x for x in result_a_better.seat_assignments[i])
            for i in range(n)
        ]
        sub = HeadToHeadResult(
            num_games=n,
            seat_assignments=swapped_assigns,
            rank_per_game=result_a_better.rank_per_game,
            win_counts_per_agent=result_a_better.win_counts_per_agent,
            name_a="A", name_b="B",
        )
        if sub.expected_score_a() >= observed:
            null_ge += 1
    return null_ge / n_resamples
