"""v4 phase-1 heuristic bootstrap with all v4 aux targets.

Uses the simple HeuristicAgent (the 2-ply variant is too slow for 50k games).
For each game: all-heuristic play with random colors. Records all aux fields
including score_margin and pin_final.
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

# IMPORTANT: do NOT import from self_play_v4 here - that pulls in torch +
# MCTS, which makes spawn-context workers re-init CUDA on import (very slow,
# can hang). Inline the small helpers we actually need.


def _bucket_distance(d: int) -> int:
    if d == 0: return 0
    if d <= 4: return 1
    if d <= 8: return 2
    if d <= 15: return 3
    return 4


def _value_vec_from_env(env: GameEnv) -> np.ndarray:
    vec = np.zeros(Config.MAX_PLAYERS, dtype=np.float32)
    for p in range(env.num_players):
        vec[p] = ts.normalized_value_target(env.compute_final_score(p))
    return vec


def _score_margin_vec(env: GameEnv) -> np.ndarray:
    margin = np.zeros(Config.MAX_PLAYERS, dtype=np.float32)
    scores = [env.compute_final_score(p) for p in range(env.num_players)]
    total = sum(scores)
    n = env.num_players
    for p in range(n):
        others = total - scores[p]
        mean_others = others / max(1, n - 1)
        margin[p] = float((scores[p] - mean_others) / 1300.0)
    return np.clip(margin, -1.0, 1.0)


def _pin_final_vec(env: GameEnv, player: int) -> np.ndarray:
    color = env.colors[player]
    pieces = env.pieces[player]
    dist_table = env.board._min_dist_to_goal[color]
    out = np.zeros(Config.NUM_PIECES, dtype=np.int8)
    for i, idx in enumerate(sorted(pieces)[:Config.NUM_PIECES]):
        out[i] = _bucket_distance(int(dist_table[idx]))
    return out


MAX_PLAYERS = Config.MAX_PLAYERS


def play_heuristic_game_v4(num_players: int, seed: Optional[int] = None) -> List[Dict]:
    """Play one all-heuristic game with v4 aux targets.

    `seed` is per-game; combined with worker pid → unique RNG per worker
    per game. Avoids correlated games across the multi-process pool.
    """
    if seed is None:
        # Fall back to a unique-ish seed if the caller didn't pass one
        import os, time as _t
        seed = (os.getpid() * 1_000_003) ^ int(_t.time() * 1e6) & 0x7fffffff
    rng = random.Random(seed)
    board = HexBoard()
    env = GameEnv(board, num_players=num_players)
    env.reset(random_colors=True, rng=rng)
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
            "score_margin": np.zeros(MAX_PLAYERS, dtype=np.float32),
            "pin_final": np.zeros(Config.NUM_PIECES, dtype=np.int8),
            "pin_final_valid": False,
        })
        env.step(action)

    if env.get_winner() is None:
        return []

    # Backfill aux targets
    for k in range(len(trajectory) - 1):
        trajectory[k]["opp_action"] = trajectory[k + 1]["action"]
        trajectory[k]["opp_legal_mask"] = trajectory[k + 1]["legal_mask"]

    value_vec = _value_vec_from_env(env)
    margin_vec = _score_margin_vec(env)
    total_plies = env.move_count
    for entry in trajectory:
        seat = entry["player"]
        entry["value_target"] = float(value_vec[seat])
        entry["value_vec"] = value_vec.copy()
        entry["score_margin"] = margin_vec.copy()
        entry["pin_final"] = _pin_final_vec(env, seat)
        entry["pin_final_valid"] = True
        plies_left = max(0, total_plies - entry["move_count"])
        entry["plies_remaining"] = min(4.0, float(plies_left) / 200.0)
        entry["plies_valid"] = True
    return trajectory


def _worker(args):
    idx, n = args
    # Each game gets a unique seed = idx * prime XOR pid; ensures per-worker
    # per-game uniqueness without requiring a global lock.
    import os
    seed = (idx * 2654435761) ^ os.getpid()
    seed &= 0x7fffffff
    return play_heuristic_game_v4(n, seed=seed)


def play_games_parallel_v4(
    num_games: int,
    num_workers: Optional[int] = None,
    num_players_distribution: Optional[Dict[int, float]] = None,
) -> List[List[Dict]]:
    if num_workers is None:
        num_workers = min(mp.cpu_count() // 4, 16)
    if num_players_distribution is None:
        num_players_distribution = {2: 0.30, 3: 0.175, 4: 0.175, 5: 0.175, 6: 0.175}

    target = num_games
    generate = int(num_games * 1.4) + 10
    Ns = list(num_players_distribution.keys())
    weights = [num_players_distribution[n] for n in Ns]
    sampled = random.choices(Ns, weights=weights, k=generate)
    args_list = list(zip(range(generate), sampled))

    results = []
    # Use fork (default) - workers in phase1 are PURE CPU (HeuristicAgent +
    # GameEnv + numpy). They never touch CUDA, so fork is safe even after
    # the parent has initialized CUDA. Spawn context is much slower here
    # because each worker re-imports the whole module chain.
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
