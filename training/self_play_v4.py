"""v4 self-play - uses MCTS-IMPROVED policy targets (proper AlphaZero).

Differences vs v3 self_play_v3:
  - Each move runs MCTS (Gumbel AlphaZero v4) with `MCTS_TRAIN_SIMS_V4` sims.
  - Policy target = MCTS visit-count distribution (not raw network argmax).
  - Drops the v3 "self-imitation overwrite" - MCTS targets are the strongest signal.
  - Records score_margin and pin_final aux targets.
  - Per-game subtree reuse: tree carries forward across our own moves.
"""

from __future__ import annotations

import random as _random
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from config import Config
from core.board import HexBoard
from core.game_env import GameEnv
from core.action_space import get_legal_actions
from core import teacher_score as ts
from training.heuristic_agent import HeuristicAgent
from mcts.mcts_v4 import GumbelMCTSv4, advance_root


TEMP_SWITCH_MOVES = 30
MAX_PLAYERS = Config.MAX_PLAYERS
PIN_BUCKETS = Config.PIN_FINAL_BUCKETS_V4


def _bucket_distance(d: int) -> int:
    """Map a per-piece distance to a coarse bucket (matches PinFinalHead semantics)."""
    if d == 0:
        return 0
    if d <= 4:
        return 1
    if d <= 8:
        return 2
    if d <= 15:
        return 3
    return 4


def _score_margin_vec(env: GameEnv) -> np.ndarray:
    """For each seat, compute (final_score - mean(others)) / 1300 → [-1,1]."""
    margin = np.zeros(MAX_PLAYERS, dtype=np.float32)
    scores = [env.compute_final_score(p) for p in range(env.num_players)]
    total = sum(scores)
    n = env.num_players
    for p in range(n):
        others = total - scores[p]
        mean_others = others / max(1, n - 1)
        margin[p] = float((scores[p] - mean_others) / 1300.0)
    return np.clip(margin, -1.0, 1.0)


def _value_vec_from_env(env: GameEnv) -> np.ndarray:
    vec = np.zeros(MAX_PLAYERS, dtype=np.float32)
    for p in range(env.num_players):
        vec[p] = ts.normalized_value_target(env.compute_final_score(p))
    return vec


def _pin_final_vec(env: GameEnv, player: int) -> np.ndarray:
    """Per-piece final-distance bucket for `player`'s pieces."""
    color = env.colors[player]
    pieces = env.pieces[player]
    dist_table = env.board._min_dist_to_goal[color]
    out = np.zeros(Config.NUM_PIECES, dtype=np.int8)
    for i, idx in enumerate(sorted(pieces)[:Config.NUM_PIECES]):
        out[i] = _bucket_distance(int(dist_table[idx]))
    return out


def _maybe_pick_heuristic_seat(num_players: int, rng) -> Optional[int]:
    if rng.random() < Config.VS_HEURISTIC_FRACTION:
        return rng.randrange(num_players)
    return None


def _maybe_pick_frozen_seats(num_players: int, frozen_pool, rng) -> Dict[int, object]:
    if not frozen_pool or rng.random() >= Config.FREEZE_OPP_FRACTION:
        return {}
    seat = rng.randrange(num_players)
    return {seat: rng.choice(frozen_pool)}


def _summarize_game(env: GameEnv, iteration: int, game_id: int,
                    nexus_seats: List[int],
                    heuristic_seat: Optional[int]) -> Dict:
    scores = {}
    for p in range(env.num_players):
        color = env.colors[p]
        pins = env.board.count_in_goal(env.pieces[p], color)
        dist = env.board.sum_distances_to_goal(env.pieces[p], color)
        scores[str(p)] = {
            "color": color,
            "pins": pins,
            "dist": dist,
            "moves": env.player_move_counts[p],
            "final_score": env.compute_final_score(p),
        }
    return {
        "iter": iteration,
        "game_id": game_id,
        "N": env.num_players,
        "seats": list(env.colors),
        "nexus_seats": nexus_seats,
        "heuristic_seat": heuristic_seat,
        "winner": env.get_winner(),
        "ended_by": ("win" if env.get_winner() is not None else "cap_or_draw"),
        "total_moves": env.move_count,
        "scores": scores,
    }


def _sample_action_from_visits(visits: np.ndarray, temperature: float, rng) -> int:
    legal = np.where(visits > 0)[0]
    if len(legal) == 0:
        return int(np.argmax(visits))
    if temperature <= 0:
        return int(legal[np.argmax(visits[legal])])
    if abs(temperature - 1.0) < 1e-6:
        p = visits[legal] / visits[legal].sum()
    else:
        log_v = np.log(visits[legal] + 1e-9) / temperature
        log_v = log_v - log_v.max()
        ev = np.exp(log_v)
        p = ev / ev.sum()
    s = p.sum()
    if s <= 0 or not np.isfinite(s):
        return int(legal[np.argmax(visits[legal])])
    p = p / s
    return int(rng.choices(legal.tolist(), weights=p.tolist(), k=1)[0])


def generate_games_with_mcts(
    network,
    device: torch.device,
    board: HexBoard,
    num_games: int,
    iteration: int,
    rng: _random.Random,
    num_simulations: int = Config.MCTS_TRAIN_SIMS_V4,
    m: int = Config.MCTS_TRAIN_M_V4,
    temperature: float = 1.0,
    start_game_id: int = 0,
    frozen_pool: Optional[List] = None,
) -> Tuple[List[List[Dict]], List[Dict]]:
    """Generate self-play games using MCTS for policy targets.

    Each (game, move) calls mcts.search() once, runs sims, and records the
    visit-count distribution as the policy target.
    """
    network.eval()
    if frozen_pool is None:
        frozen_pool = []

    trajectories: List[List[Dict]] = []
    summaries: List[Dict] = []
    heuristic_agent = HeuristicAgent(board)

    import time as _time
    _sp_start = _time.time()
    for i in range(num_games):
        _g_start = _time.time()
        # N from curriculum
        N = Config.sample_num_players(iteration, rng=rng, v3=True)
        env = GameEnv(board, num_players=N)
        env.reset(random_colors=True, rng=rng)

        heuristic_seat = _maybe_pick_heuristic_seat(N, rng)
        frozen_seats = _maybe_pick_frozen_seats(N, frozen_pool, rng)
        if heuristic_seat is not None and heuristic_seat in frozen_seats:
            del frozen_seats[heuristic_seat]
        nexus_seats = [p for p in range(N)
                       if p != heuristic_seat and p not in frozen_seats]

        # Per-seat MCTS (independent search trees for live + frozen seats).
        # NOTE: reverted to non-batched MCTS - batched variant stalled iter 0
        # on full Halma games despite passing small-game equivalence tests.
        # Real GPU saturation requires parallel-game workers (TBD).
        live_mcts = GumbelMCTSv4(network, device,
                                 num_simulations=num_simulations, m=m)
        frozen_mcts = {seat: GumbelMCTSv4(net, device,
                                          num_simulations=num_simulations, m=m)
                       for seat, net in frozen_seats.items()}

        # Persistent root for each network seat (subtree reuse)
        live_roots: Dict[int, object] = {p: None for p in nexus_seats}
        frozen_roots: Dict[int, object] = {p: None for p in frozen_seats}

        traj: List[Dict] = []
        # Pending opp-action backfill: list of trajectory entry indices
        pending_opp: List[int] = []

        while not env.is_done():
            p = env.current_player

            # Pick action by seat type
            if heuristic_seat is not None and p == heuristic_seat:
                state = env.get_state_tensor(p).numpy()
                mask = env.get_legal_mask(p).numpy()
                action = heuristic_agent.choose_move(env, p)
                policy_target = np.zeros(Config.ACTION_SPACE, dtype=np.float32)
                policy_target[action] = 1.0
                is_heuristic = True
            elif p in frozen_seats:
                state = env.get_state_tensor(p).numpy()
                mask = env.get_legal_mask(p).numpy()
                action, policy_target, _, frozen_roots[p] = frozen_mcts[p].search(
                    env, root=frozen_roots[p]
                )
                is_heuristic = False
            else:  # live seat
                state = env.get_state_tensor(p).numpy()
                mask = env.get_legal_mask(p).numpy()
                action, mcts_policy, _, live_roots[p] = live_mcts.search(
                    env, root=live_roots[p]
                )
                # Sample by temperature for early moves; greedy after
                use_greedy = env.move_count >= TEMP_SWITCH_MOVES
                if use_greedy:
                    final_action = int(np.argmax(mcts_policy))
                else:
                    # mcts_policy is already a normalized distribution
                    visits = mcts_policy.copy()
                    final_action = _sample_action_from_visits(visits, temperature, rng)
                action = final_action
                policy_target = mcts_policy.astype(np.float32)
                is_heuristic = False

            train_this_entry = (p in nexus_seats) or (p == heuristic_seat)

            # Backfill any pending opp_action with THIS move
            if pending_opp:
                for idx in pending_opp:
                    traj[idx]["opp_action"] = int(action)
                    traj[idx]["opp_legal_mask"] = mask
                pending_opp = []

            if train_this_entry:
                traj.append({
                    "state": state,
                    "action": int(action),
                    "policy_target": np.asarray(policy_target, dtype=np.float32),
                    "legal_mask": mask,
                    "player": int(p),
                    "n_players": int(N),
                    "move_count": env.move_count,
                    "game_id": start_game_id + i,
                    "is_heuristic": is_heuristic,
                    "opp_action": -1,
                    "opp_legal_mask": np.zeros(Config.ACTION_SPACE, dtype=np.bool_),
                    "plies_remaining": 0.0,
                    "plies_valid": False,
                    "score_margin": np.zeros(MAX_PLAYERS, dtype=np.float32),
                    "pin_final": np.zeros(Config.NUM_PIECES, dtype=np.int8),
                    "pin_final_valid": False,
                })
                pending_opp.append(len(traj) - 1)

            env.step(int(action))

            # Subtree reuse: descend each seat's root by the played action
            for seat_p, mcts_root in list(live_roots.items()):
                if mcts_root is not None:
                    live_roots[seat_p] = advance_root(mcts_root, action)
            for seat_p, mcts_root in list(frozen_roots.items()):
                if mcts_root is not None:
                    frozen_roots[seat_p] = advance_root(mcts_root, action)

        # Backfill terminal aux targets
        value_vec = _value_vec_from_env(env)
        margin_vec = _score_margin_vec(env)
        total_plies = env.move_count
        for entry in traj:
            seat = entry["player"]
            entry["value_vec"] = value_vec.copy()
            entry["value_target"] = float(value_vec[seat])
            entry["score_margin"] = margin_vec.copy()
            entry["pin_final"] = _pin_final_vec(env, seat)
            entry["pin_final_valid"] = True
            plies_left = max(0, total_plies - entry["move_count"])
            entry["plies_remaining"] = min(4.0, float(plies_left) / 200.0)
            entry["plies_valid"] = True

        trajectories.append(traj)
        summaries.append(_summarize_game(
            env, iteration=iteration, game_id=start_game_id + i,
            nexus_seats=nexus_seats, heuristic_seat=heuristic_seat,
        ))
        _g_dt = _time.time() - _g_start
        _sp_dt = _time.time() - _sp_start
        print(f"[selfplay] iter={iteration} game {i+1}/{num_games} "
              f"moves={env.move_count} N={N} dt={_g_dt:.1f}s "
              f"cum={_sp_dt:.1f}s", flush=True)

    return trajectories, summaries
