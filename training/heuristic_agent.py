"""Heuristic agent for Phase 1 bootstrap data generation.

Strong heuristic with:
- Move rear pieces first (piece ordering)
- Chain-building: prefer moves that create hop opportunities
- Goal zone management: shuffle pieces to make room for stragglers
- Anti-blocking: avoid gridlocking friendly pieces
- Parallel game generation across multiple CPU cores
"""

import random
import multiprocessing as mp
from typing import Dict, List, Optional, Tuple

from core.board import HexBoard
from core.game_env import GameEnv
from core.action_space import encode_action
from config import Config


class HeuristicAgent:
    """Strong hand-crafted heuristic agent for generating training data."""

    def __init__(self, board: HexBoard):
        self.board = board
        self._goal_list: Dict[str, List[int]] = {}
        for color in Config.COLOR_OPPOSITES:
            self._goal_list[color] = sorted(board.goal_zones[color])

    def choose_move(
        self,
        env: GameEnv,
        player: Optional[int] = None,
    ) -> int:
        if player is None:
            player = env.current_player

        color = env.colors[player]
        pieces = env.pieces[player]
        legal_moves = env.get_legal_moves(player)
        occupied = env.get_occupied()

        board = self.board
        dist_table = board._min_dist_to_goal[color]
        goal_set = board._goal_set[color]
        pairwise = board._dist_table
        directions = Config.DIRECTIONS
        index_of = board.index_of
        cell_q = board.cell_q
        cell_r = board.cell_r

        piece_dists = [(dist_table[p], p) for p in pieces]
        max_piece_dist = max(d for d, _ in piece_dists) if piece_dists else 1
        in_goal_count = sum(1 for p in pieces if p in goal_set)

        # Find which goal cells are empty (available to land in)
        occupied_goal = set(p for p in pieces if p in goal_set)
        empty_goal = goal_set - occupied_goal - (occupied - set(pieces))

        # Find pieces NOT in goal, sorted by distance (closest first)
        stragglers = sorted(
            [(dist_table[p], p) for p in pieces if p not in goal_set],
            key=lambda x: x[0]
        )

        pos_to_id = {pos: pid for pid, pos in enumerate(pieces)}
        friendly_set = set(pieces)

        best_action = None
        best_score = float('-inf')

        for piece_pos, destinations in legal_moves.items():
            pid = pos_to_id[piece_pos]
            piece_dist_to_goal = dist_table[piece_pos]
            in_goal_already = piece_pos in goal_set

            rear_bonus = (piece_dist_to_goal / max(max_piece_dist, 1)) * 4.0

            for dest in destinations:
                score = 0.0
                dest_dist_to_goal = dist_table[dest]

                # 1. Distance reduction
                dist_reduction = piece_dist_to_goal - dest_dist_to_goal
                score += dist_reduction * 10.0

                # 2. Rear piece bonus
                score += rear_bonus * max(0, dist_reduction)

                # 3. Hop length bonus
                hop_length = pairwise[piece_pos][dest]
                if hop_length > 1:
                    score += hop_length * 3.0

                # 4. Goal zone management
                if in_goal_already:
                    if dest in goal_set:
                        # Repositioning within goal - only good if it opens
                        # a path for a straggler
                        if stragglers:
                            closest_straggler_dist, closest_straggler = stragglers[0]
                            # Does moving free up piece_pos for the straggler?
                            straggler_dist_to_old = pairwise[closest_straggler][piece_pos]
                            straggler_dist_to_dest = pairwise[closest_straggler][dest]
                            if straggler_dist_to_old < straggler_dist_to_dest:
                                score += 8.0  # good shuffle - opens space
                            else:
                                score += 0.5  # neutral shuffle
                        else:
                            score += 0.5
                    else:
                        score -= 100.0  # NEVER leave goal
                else:
                    # Not in goal yet
                    if dest in goal_set:
                        # Landing in goal - high priority
                        score += 20.0
                        if in_goal_count >= 7:
                            score += 15.0  # urgent when close to winning
                        if in_goal_count >= 9:
                            score += 30.0  # last piece - critical
                    elif dest_dist_to_goal == 0:
                        # At distance 0 but dest not in goal_set? Shouldn't happen
                        score += 10.0

                # 5. Chain building
                dq, dr = cell_q[dest], cell_r[dest]
                for ddq, ddr in directions:
                    src_q, src_r = dq - ddq, dr - ddr
                    land_q, land_r = dq + ddq, dr + ddr
                    src_idx = index_of.get((src_q, src_r))
                    land_idx = index_of.get((land_q, land_r))
                    if (src_idx is not None and land_idx is not None
                            and src_idx in friendly_set
                            and src_idx != piece_pos
                            and land_idx not in occupied):
                        if dist_table[land_idx] < dist_table[src_idx]:
                            score += 2.5
                        else:
                            score += 0.5

                # 6. Anti-blocking
                for ddq, ddr in directions:
                    adj_q, adj_r = dq + ddq, dr + ddr
                    adj_idx = index_of.get((adj_q, adj_r))
                    if (adj_idx is not None and adj_idx in friendly_set
                            and adj_idx != piece_pos):
                        if dist_table[adj_idx] > dest_dist_to_goal:
                            score -= 1.0

                # 7. Center control early game
                if env.move_count < 20:
                    center_idx = index_of.get((0, 0))
                    if center_idx is not None:
                        center_dist = pairwise[dest][center_idx]
                        score += max(0, 4 - center_dist) * 0.3

                # 8. Randomness for diversity
                score += random.gauss(0, 0.8)

                if score > best_score:
                    best_score = score
                    best_action = encode_action(pid, dest)

        return best_action


_COLOR_PAIRS = [('red', 'blue'), ('lawn green', 'gray0'), ('yellow', 'purple')]


def play_heuristic_game(board: Optional[HexBoard] = None,
                        num_players: int = 2) -> List[Dict]:
    """Play a full N-player game where every seat is the heuristic agent.

    Returns:
        List of trajectory dicts, one per move (across ALL seats).
        Each entry has a single-scalar value_target = normalized teacher score.
        Returns empty list if the game ended without a winner (cap or full draw).
    """
    if board is None:
        board = HexBoard()

    env = GameEnv(board, num_players=num_players)
    env.reset()
    agent = HeuristicAgent(board)
    trajectory = []

    while not env.is_done():
        player = env.current_player
        state = env.get_state_tensor(player).numpy()
        legal_mask = env.get_legal_mask(player).numpy()
        action = agent.choose_move(env, player)

        trajectory.append({
            'state': state,
            'action': action,
            'player': player,
            'legal_mask': legal_mask,
            'move_count': env.move_count,
        })

        reward, done = env.step(action)
        trajectory[-1]['reward'] = reward

    # Filter: only games that actually completed with some winner are useful for
    # bootstrap. Games that hit the move cap (winner=None) are noisy.
    if env.get_winner() is None:
        return []

    # Single-scalar value target = normalized teacher final_score per player
    for entry in trajectory:
        entry['value_target'] = float(env.compute_value_target(entry['player']))

    return trajectory


def _play_heuristic_game_worker(args) -> List[Dict]:
    """Worker function for parallel game generation. args = (idx, num_players)."""
    _idx, num_players = args
    board = HexBoard()
    return play_heuristic_game(board, num_players=num_players)


def play_heuristic_games_parallel(
    num_games: int,
    num_workers: int = None,
    num_players_distribution: Optional[Dict[int, float]] = None,
) -> List[List[Dict]]:
    """Generate heuristic games in parallel across N-player counts.

    num_players_distribution: optional {N: weight} dict; defaults to N=2 only.
    Filters out games that ended without a winner. Generates extras to compensate.
    """
    if num_workers is None:
        num_workers = min(mp.cpu_count() // 4, 16)
    if num_players_distribution is None:
        num_players_distribution = {2: 1.0}

    target = num_games
    generate = int(num_games * 1.4) + 10

    # Pre-sample N for each game so workers don't need access to RNG/Config
    Ns = list(num_players_distribution.keys())
    weights = [num_players_distribution[n] for n in Ns]
    sampled_Ns = random.choices(Ns, weights=weights, k=generate)
    args_list = list(zip(range(generate), sampled_Ns))

    results = []
    pool = mp.Pool(processes=num_workers)
    try:
        async_result = pool.map_async(
            _play_heuristic_game_worker,
            args_list,
            chunksize=max(1, generate // (num_workers * 4)),
        )
        all_trajs = async_result.get(timeout=7200)
        for traj in all_trajs:
            if traj:
                results.append(traj)
                if len(results) >= target:
                    break
    finally:
        pool.terminate()
        pool.join()

    return results[:target]
