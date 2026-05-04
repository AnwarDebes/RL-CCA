"""Elo rating system for Phase 3 population training."""

import math
from typing import Dict


def expected_score(rating_a: float, rating_b: float) -> float:
    return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400.0))


def update_elo(
    rating_a: float,
    rating_b: float,
    score_a: float,
    k: float = 32.0,
) -> tuple[float, float]:
    """Update Elo ratings after a match.

    Args:
        rating_a, rating_b: current ratings.
        score_a: actual score for player A (1.0 win, 0.5 draw, 0.0 loss).
        k: K-factor.

    Returns:
        (new_rating_a, new_rating_b)
    """
    ea = expected_score(rating_a, rating_b)
    eb = 1.0 - ea
    score_b = 1.0 - score_a

    new_a = rating_a + k * (score_a - ea)
    new_b = rating_b + k * (score_b - eb)
    return new_a, new_b


class EloTracker:
    """Track Elo ratings for a population of agents."""

    def __init__(self, k: float = 32.0, initial_rating: float = 1000.0):
        self.k = k
        self.initial_rating = initial_rating
        self.ratings: Dict[str, float] = {}

    def register(self, agent_id: str):
        if agent_id not in self.ratings:
            self.ratings[agent_id] = self.initial_rating

    def record_match(self, agent_a: str, agent_b: str, score_a: float):
        self.register(agent_a)
        self.register(agent_b)
        new_a, new_b = update_elo(
            self.ratings[agent_a], self.ratings[agent_b], score_a, self.k
        )
        self.ratings[agent_a] = new_a
        self.ratings[agent_b] = new_b

    def get_rating(self, agent_id: str) -> float:
        return self.ratings.get(agent_id, self.initial_rating)

    def get_opponent_in_range(self, agent_id: str, elo_range: float = 200.0):
        """Find an opponent within elo_range of the given agent."""
        my_rating = self.get_rating(agent_id)
        candidates = [
            (aid, abs(r - my_rating))
            for aid, r in self.ratings.items()
            if aid != agent_id and abs(r - my_rating) <= elo_range
        ]
        if not candidates:
            return None
        candidates.sort(key=lambda x: x[1])
        return candidates[0][0]
