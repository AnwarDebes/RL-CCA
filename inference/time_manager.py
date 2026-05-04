"""Time management for tournament play.

Tournament constraints:
  TURN_TIMEOUT_SEC = 10 (per move)
  GAME_TIME_LIMIT_SEC = 60 (total)

We leave buffers: 8s per move, 55s total.
"""

import time

from config import Config


class TimeManager:
    """Manages time budget during tournament play."""

    def __init__(
        self,
        total_budget: float = Config.TOTAL_TIME_BUDGET,
        per_move_budget: float = Config.PER_MOVE_BUDGET,
    ):
        self.total_budget = total_budget
        self.per_move_budget = per_move_budget
        self.time_spent = 0.0
        self.move_count = 0
        self._move_start: float = 0.0

    def start_move(self):
        """Call at the start of each move."""
        self._move_start = time.time()

    def elapsed_this_move(self) -> float:
        """How much time has been spent on the current move."""
        return time.time() - self._move_start

    def record_move(self, elapsed: float = None):
        """Record that a move took `elapsed` seconds."""
        if elapsed is None:
            elapsed = self.elapsed_this_move()
        self.time_spent += elapsed
        self.move_count += 1

    def get_simulation_budget(self) -> int:
        """Determine MCTS simulation count based on remaining time."""
        remaining = self.total_budget - self.time_spent
        est_remaining_moves = max(10, 50 - self.move_count)
        time_per_move = remaining / est_remaining_moves

        if time_per_move > 2.0:
            return 64
        elif time_per_move > 1.0:
            return 32
        elif time_per_move > 0.5:
            return 16
        else:
            return 8

    def should_abort_search(self) -> bool:
        """Check if we need to abort MCTS early to stay within per-move budget."""
        return self.elapsed_this_move() > self.per_move_budget * 0.8

    def remaining_total(self) -> float:
        return self.total_budget - self.time_spent

    def remaining_this_move(self) -> float:
        return self.per_move_budget - self.elapsed_this_move()
