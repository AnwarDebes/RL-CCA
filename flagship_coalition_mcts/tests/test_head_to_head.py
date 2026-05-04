"""Tests for the head-to-head evaluation harness."""

from __future__ import annotations

import numpy as np

from flagship_coalition_mcts.src.games.kingmaker import (
    KingmakerGame, KingmakerState, NUM_ACTIONS,
)
from flagship_coalition_mcts.src.head_to_head import (
    HeadToHeadResult, head_to_head, permutation_test,
)


def random_agent(seed_offset: int):
    rng = np.random.default_rng(seed_offset)
    def play(state):
        legal = KingmakerGame.legal_actions(state)
        return int(rng.integers(0, len(legal)))
    return play


def first_legal_agent(state):
    legal = KingmakerGame.legal_actions(state)
    return 0


def test_head_to_head_returns_result_with_correct_counts():
    res = head_to_head(
        game=KingmakerGame(),
        initial_state_fn=KingmakerState.initial,
        num_players=3,
        agent_a=random_agent(0),
        agent_b=first_legal_agent,
        name_a="rand", name_b="first",
        num_games=20,
        num_a_seats=1,
        seed=0,
    )
    assert isinstance(res, HeadToHeadResult)
    assert res.num_games == 20
    # Win counts non-negative
    assert all(c >= 0 for c in res.win_counts_per_agent)


def test_expected_score_in_unit_interval():
    res = head_to_head(
        game=KingmakerGame(),
        initial_state_fn=KingmakerState.initial,
        num_players=3,
        agent_a=random_agent(1), agent_b=random_agent(2),
        num_games=10, num_a_seats=2, seed=0,
    )
    s = res.expected_score_a()
    assert 0.0 <= s <= 1.0


def test_elo_gap_bounded():
    res = head_to_head(
        game=KingmakerGame(),
        initial_state_fn=KingmakerState.initial,
        num_players=3,
        agent_a=random_agent(3), agent_b=random_agent(4),
        num_games=10, num_a_seats=1, seed=0,
    )
    gap = res.elo_gap()
    # Bounded between extreme rare events
    assert -3000 < gap < 3000


def test_bootstrap_ci_brackets_point_estimate():
    res = head_to_head(
        game=KingmakerGame(),
        initial_state_fn=KingmakerState.initial,
        num_players=3,
        agent_a=random_agent(5), agent_b=random_agent(6),
        num_games=12, num_a_seats=1, seed=0,
    )
    point = res.elo_gap()
    lo, hi = res.bootstrap_elo_ci(n_resamples=200, seed=0)
    assert lo - 1 <= point <= hi + 1


def test_permutation_pvalue_in_unit_interval():
    res = head_to_head(
        game=KingmakerGame(),
        initial_state_fn=KingmakerState.initial,
        num_players=3,
        agent_a=random_agent(7), agent_b=random_agent(8),
        num_games=8, num_a_seats=1, seed=0,
    )
    p = permutation_test(res, n_resamples=200, seed=0)
    assert 0.0 <= p <= 1.0
