"""v3 phase-1 heuristic bootstrap: mixed-N games with full aux targets.

All seats are heuristic. Each (state, action) is recorded with:
  - policy_target: one-hot at heuristic action
  - value_target: normalized teacher final_score for that seat (scalar)
  - value_vec: full per-player vector (length 6, padded)
  - n_players, player
  - opp_action: next move's action (by next seat); -1 on terminal entry
  - plies_remaining: (total_plies - this_move_count) / 200, capped at 4.0
"""

from __future__ import annotations

import multiprocessing as mp
import random
from typing import Dict, List, Optional

import numpy as np

from config import Config
from core.board import HexBoard
from core.game_env import GameEnv
from core import teacher_score as ts
from training.heuristic_agent import HeuristicAgent


MAX_PLAYERS = Config.MAX_PLAYERS


def play_heuristic_game_v3(num_players: int) -> List[Dict]:
    """Single all-heuristic N-player game with v3 aux fields. Returns empty
    list if the game didn't conclude with a winner."""
    board = HexBoard()
    env = GameEnv(board, num_players=num_players)
    # v3-rebuild: random colors per game + teacher-style COLOUR_ORDER seating
    import random as _r
    env.reset(random_colors=True, rng=_r)
    agent = HeuristicAgent(board)
    trajectory = []
    while not env.is_done():
        p = env.current_player
        state = env.get_state_tensor(p).numpy()
        legal_mask = env.get_legal_mask(p).numpy()
        action = agent.choose_move(env, p)
        onehot = np.zeros(Config.ACTION_SPACE, dtype=np.float32)
        onehot[action] = 1.0
        trajectory.append({
            "state": state,
            "action": int(action),
            "policy_target": onehot,
            "legal_mask": legal_mask,
            "player": int(p),
            "n_players": int(num_players),
            "move_count": env.move_count,
            "is_heuristic": True,
            "opp_action": -1,
            "opp_legal_mask": np.zeros(Config.ACTION_SPACE, dtype=np.bool_),
            "plies_remaining": 0.0,
            "plies_valid": False,
        })
        env.step(action)

    if env.get_winner() is None:
        return []

    # Backfill opp_action: each entry's opp_action = next entry's action,
    # opp_legal_mask = next entry's legal_mask.
    for k in range(len(trajectory) - 1):
        trajectory[k]["opp_action"] = trajectory[k + 1]["action"]
        trajectory[k]["opp_legal_mask"] = trajectory[k + 1]["legal_mask"]

    # Per-player value vector (padded)
    value_vec = np.zeros(MAX_PLAYERS, dtype=np.float32)
    for p in range(num_players):
        value_vec[p] = ts.normalized_value_target(env.compute_final_score(p))

    total_plies = env.move_count
    for entry in trajectory:
        seat = entry["player"]
        entry["value_target"] = float(value_vec[seat])
        entry["value_vec"] = value_vec.copy()
        plies_left = max(0, total_plies - entry["move_count"])
        entry["plies_remaining"] = min(4.0, float(plies_left) / 200.0)
        entry["plies_valid"] = True

    return trajectory


def _worker(args):
    _idx, n = args
    return play_heuristic_game_v3(n)


def play_games_parallel_v3(
    num_games: int,
    num_workers: Optional[int] = None,
    num_players_distribution: Optional[Dict[int, float]] = None,
) -> List[List[Dict]]:
    if num_workers is None:
        num_workers = min(mp.cpu_count() // 4, 16)
    if num_players_distribution is None:
        num_players_distribution = {2: 1.0}

    target = num_games
    generate = int(num_games * 1.4) + 10
    Ns = list(num_players_distribution.keys())
    weights = [num_players_distribution[n] for n in Ns]
    sampled = random.choices(Ns, weights=weights, k=generate)
    args_list = list(zip(range(generate), sampled))

    results = []
    pool = mp.Pool(processes=num_workers)
    try:
        all_trajs = pool.map_async(
            _worker, args_list,
            chunksize=max(1, generate // (num_workers * 4)),
        ).get(timeout=7200)
        for traj in all_trajs:
            if traj:
                results.append(traj)
                if len(results) >= target:
                    break
    finally:
        pool.terminate()
        pool.join()
    return results[:target]
