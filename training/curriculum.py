"""Progressive simulation schedule and curriculum helpers."""

from config import Config


def get_progressive_sims(iteration: int) -> int:
    """Return simulation count based on training iteration."""
    return Config.get_progressive_sims(iteration)


def get_temperature(iteration: int, total_iterations: int = 1000) -> float:
    """Get exploration temperature for self-play."""
    return Config.get_temperature(iteration, total_iterations)
