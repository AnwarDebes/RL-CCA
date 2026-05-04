"""Teacher's tournament scoring formula - exact replica.

This is the SINGLE source of truth for "score". All training rewards, value
targets, evaluation summaries, and rule alignment checks compute scores
through this module. It must remain in lockstep with the teacher's
`compute_scores` in:

  RLChineseCheckers/multi system single machine minimal/game.py:198-256

Formula (verified against teacher's README and game.py source):

  time_score     = max(0.0, 100.0 - time_taken_sec)
  move_score     = exp(-((move_count - 45)^2 / (2 * sigma^2)))
                   where sigma = 4 if move_count < 45 else 18
  pin_goal_score = 100.0 * pins_in_goal
  distance_score = max(0.0, 200.0 - total_distance)
  final_score    = time_score + move_score + pin_goal_score + distance_score

Notes:
- `time_taken_sec` is the player's accumulated thinking time, not the global
  game clock. A player who hasn't moved yet has time_taken_sec=0, score=0.
- `move_count` here is per-player move count.
- `pins_in_goal` counts pins of color C that have reached
  colour_opposites[C]'s zone.
- `total_distance` is the sum of axial-hex-distance from each NOT-yet-in-goal
  pin to its nearest target zone cell. Pins already in goal contribute 0.
"""

import math
from typing import Dict


def time_score(time_taken_sec: float) -> float:
    """Time component. Max 100 (instant move), drops linearly to 0 at 100s."""
    if time_taken_sec <= 0.0:
        return 0.0
    return max(0.0, 100.0 - time_taken_sec)


def move_score(move_count: int) -> float:
    """Asymmetric Gaussian peaking at 45 moves. sigma=4 below, sigma=18 above.

    Max value is 1.0 (negligible compared to pin/distance contributions).
    Returns 0 if move_count <= 0.
    """
    if move_count <= 0:
        return 0.0
    sigma = 4.0 if move_count < 45 else 18.0
    return math.exp(-((move_count - 45) ** 2) / (2.0 * sigma ** 2))


def pin_goal_score(pins_in_goal: int) -> float:
    """100 points per pin of this color in the opposite zone. Max 1000."""
    return 100.0 * pins_in_goal


def distance_score(total_distance: float) -> float:
    """200 minus total remaining distance (sum over not-in-goal pieces).

    Returns 0 if no moves taken (matches teacher: `if pl.move_count > 0`).
    Caller should pass the gating condition explicitly via final_score().
    """
    return max(0.0, 200.0 - total_distance)


def final_score(
    time_taken_sec: float,
    move_count: int,
    pins_in_goal: int,
    total_distance: float,
) -> float:
    """Compute the teacher's tournament final_score for one player.

    Mirrors game.py:198-256. Note: the teacher gates time_score and
    distance_score on `move_count > 0` - a player who never moved scores 0
    on those components.
    """
    if move_count <= 0:
        ts = 0.0
        ds = 0.0
    else:
        ts = time_score(time_taken_sec)
        ds = distance_score(total_distance)
    ms = move_score(move_count)
    ps = pin_goal_score(pins_in_goal)
    return ts + ms + ps + ds


def score_components(
    time_taken_sec: float,
    move_count: int,
    pins_in_goal: int,
    total_distance: float,
) -> Dict[str, float]:
    """Return the four components individually plus the sum.

    Useful for logging and for the rule-alignment monitor that compares
    component-by-component against the teacher's reported scores.
    """
    if move_count <= 0:
        ts = 0.0
        ds = 0.0
    else:
        ts = time_score(time_taken_sec)
        ds = distance_score(total_distance)
    ms = move_score(move_count)
    ps = pin_goal_score(pins_in_goal)
    return {
        "time_score": ts,
        "move_score": ms,
        "pin_goal_score": ps,
        "distance_score": ds,
        "final_score": ts + ms + ps + ds,
    }


# Sanity bounds - used by the value-target normalization and by tests.
SCORE_MIN = 0.0
SCORE_MAX = 100.0 + 1.0 + 1000.0 + 200.0  # 1301.0
SCORE_NORMALIZATION_MEAN = 500.0
SCORE_NORMALIZATION_RANGE = 700.0


def normalized_value_target(score: float) -> float:
    """Map a raw final_score in [0, ~1301] to a value target in [-1, 1].

    `(score - 500) / 700`, clipped. 500 chosen as approximate mid-pack score
    for a competitive player; 700 chosen so good play (~1200) maps near +1
    and weak play (~0-200) maps near -1.
    """
    v = (score - SCORE_NORMALIZATION_MEAN) / SCORE_NORMALIZATION_RANGE
    return max(-1.0, min(1.0, v))
