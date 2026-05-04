"""Chinese Checkers env adapter for CD-MCTS.

Wraps the existing nexus `core.game_env.GameEnv` as a CD-MCTS-compatible
game with the duck-typed interface expected by `mcts.py`. This is the
real bridge from the flagship research code to the user's actual
tournament game.

Design notes
------------
1. State representation: we use **the GameEnv instance itself** as the
   "state" object passed around in MCTS. Each MCTS step calls
   `state.clone()` before applying actions, preserving the immutable-
   state contract MCTS expects.
2. Action space: CC has 1210 actions (10 pieces × 121 cells). MCTS uses
   the legal-mask to restrict.
3. Feature encoding: GameEnv.get_state_tensor returns a torch tensor
   of shape (num_channels, 17, 17). The flagship's MLP encoder won't
   handle this; we provide `cc_state_to_features` that flattens to a
   1D ndarray for compatibility with the existing MLP encoder, AND a
   `cc_state_to_2d_features` for use with a CNN encoder.
4. Terminal marginal: derived from teacher's final_score per player,
   sorted descending (highest score → rank 1). Ties broken by player
   index ascending (stable, matches teacher rule semantics).
5. Win condition: a player who has all NUM_PIECES in goal-zone is the
   winner; remaining players rank by their final_score.
6. Draw rule (N-1 players stuck): GameEnv handles this internally;
   `is_done()` and `get_winner()` reflect it correctly.

Compute discipline note
-----------------------
This module imports `core.game_env` which imports torch. Importing it
during active v4 self-play training is safe (it has no side effects
beyond pure-Python module initialisation), but instantiating a GameEnv
is moderately heavy because it builds the HexBoard and StateEncoder.
For this reason, instantiate the board once and pass it explicitly.

Tests in tests/test_cc_adapter.py verify the adapter on small games.
"""

from __future__ import annotations

import os
import sys
from typing import List, Optional

import numpy as np

# We expect this module to be imported from the nexus project root.
# Add nexus root to path defensively for direct invocation.
_NEXUS_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..")
)
if _NEXUS_ROOT not in sys.path:
    sys.path.insert(0, _NEXUS_ROOT)


def _lazy_imports():
    """Defer the heavy imports until first use to avoid slowing import time
    when the adapter isn't actually exercised."""
    from core.board import HexBoard
    from core.game_env import GameEnv
    from core import teacher_score as ts
    from config import Config
    return HexBoard, GameEnv, ts, Config


def make_cc_env(num_players: int, seed: Optional[int] = None) -> "GameEnv":
    """Construct a fresh CC env and reset with random colors (matches
    teacher's tournament randomization)."""
    HexBoard, GameEnv, _, _ = _lazy_imports()
    import random
    rng = random.Random(seed)
    board = HexBoard()
    env = GameEnv(board=board, num_players=num_players)
    env.reset(num_players=num_players, random_colors=True, rng=rng)
    return env


# ----------------------------------------------------------------------
# Feature encoders
# ----------------------------------------------------------------------


def cc_state_to_features_flat(state: "GameEnv") -> np.ndarray:
    """Flatten the (C, 17, 17) state tensor to 1D ndarray.

    For use with the flagship's MLPEncoder. Larger games will want a
    CNN - see `cc_state_to_features_2d`.
    """
    p = state.current_player
    t = state.get_state_tensor(p)
    return t.detach().cpu().numpy().astype(np.float32).flatten()


def cc_state_to_features_2d(state: "GameEnv") -> np.ndarray:
    """Return the (C, 17, 17) ndarray suitable for a CNN encoder."""
    p = state.current_player
    t = state.get_state_tensor(p)
    return t.detach().cpu().numpy().astype(np.float32)


# ----------------------------------------------------------------------
# Game interface (duck-typed for CD-MCTS)
# ----------------------------------------------------------------------


class ChineseCheckersGame:
    """Adapter exposing the duck-typed interface CD-MCTS expects.

    Static methods take a GameEnv as `state`. We *clone* before any
    mutation so MCTS doesn't see side effects.
    """

    @staticmethod
    def num_players(state) -> int:
        return state.num_players

    @staticmethod
    def current_player(state) -> int:
        return state.current_player

    @staticmethod
    def legal_actions(state) -> List[int]:
        if state.is_done():
            return []
        mask = state.get_legal_mask(state.current_player)
        # mask is a torch boolean tensor of shape (1210,)
        idx = np.nonzero(mask.detach().cpu().numpy())[0]
        return [int(a) for a in idx.tolist()]

    @staticmethod
    def is_terminal(state) -> bool:
        return state.is_done()

    @staticmethod
    def step(state, action: int):
        """Returns (next_state, current_player_after_step).

        IMPORTANT: clones first so the input state is unchanged.
        """
        nxt = state.clone()
        _reward, _done = nxt.step(action)
        return nxt, nxt.current_player

    @staticmethod
    def terminal_marginal(state) -> np.ndarray:
        """N x N one-hot rank assignment.

        Computes ranks from teacher final scores: highest score → rank 1.
        Ties broken by player index ascending (stable rule matching the
        teacher's tournament tie-breaker semantics).
        """
        if not state.is_done():
            raise ValueError("terminal_marginal called on non-terminal state")
        N = state.num_players
        scores = [(state.compute_final_score(p), p) for p in range(N)]
        # Sort descending by score, ascending by player index for ties.
        scores.sort(key=lambda x: (-x[0], x[1]))
        rank_of = [0] * N
        for k, (_s, p) in enumerate(scores):
            rank_of[p] = k + 1
        M = np.zeros((N, N), dtype=np.float64)
        for p in range(N):
            M[p, rank_of[p] - 1] = 1.0
        return M

    @staticmethod
    def initial(num_players: int = 2, seed: Optional[int] = None):
        return make_cc_env(num_players=num_players, seed=seed)


# ----------------------------------------------------------------------
# Score-component decomposition for CMAZ training
# ----------------------------------------------------------------------


def cc_score_components(state, player: int) -> np.ndarray:
    """Return the 4 score components for a given player at game end.

    Components are (in order):
        0: pin_goal_score   - 1000 pts max
        1: distance_score   - 200 pts max
        2: time_score       - 100 pts max
        3: move_score       - ~1 pt max (negligible)

    Each is normalised to [0, 1] by dividing by the component's max,
    matching the convention used by teacher_score.normalized_value_target.

    For non-terminal states, returns the *current* component scores.
    """
    HexBoard, GameEnv, ts, Config = _lazy_imports()
    color = state.colors[player]
    pins = state.board.count_in_goal(state.pieces[player], color)
    total_dist = state.board.sum_distances_to_goal(state.pieces[player], color)
    pin_goal = pins * 100.0  # teacher_score uses 100 per pin → max 1000
    dist_score = max(0.0, 200.0 - float(total_dist))  # teacher distance_score
    time_score = max(
        0.0, 100.0 - state.player_time_taken[player]
    )
    move_score = max(0.0, 1.0 - state.player_move_counts[player] * 0.001)
    # Normalise each
    return np.array([
        pin_goal / 1000.0,
        dist_score / 200.0,
        time_score / 100.0,
        move_score / 1.0,
    ], dtype=np.float32)
