"""v3 self-play. Differences vs v2 self_play.py:

  • Records per-player value vector (length 6, padded), n_players, current
    player seat, opponent's NEXT action, plies-remaining target - for the
    aux heads.
  • Supports a frozen opponent pool: a fraction of seats use a frozen
    snapshot from a previous iteration instead of the live network.
  • Uses Config.NUM_PLAYERS_CURRICULUM_V3 (N=2 floor at 20%).
  • Keeps the within-game greedy switch (TEMP_SWITCH_MOVES) - that fix is
    load-bearing for finishing games.

Returns trajectories where each entry has the keys consumed by ReplayBufferV3:
  state, action, policy_target, legal_mask, player, n_players, move_count,
  game_id, reward, value_target (current-player slot, scalar in [-1, 1]),
  value_vec (length 6, normalized), opp_action (-1 if no follow-up move),
  opp_legal_mask, plies_remaining (raw plies), plies_valid (bool),
  is_heuristic (bool, optional)
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


TEMP_SWITCH_MOVES: int = 30
MAX_PLAYERS = Config.MAX_PLAYERS


# ── Helpers ───────────────────────────────────────────────────────────


def _sample_num_players(iteration: int, rng=None) -> int:
    return Config.sample_num_players(iteration, rng=rng, v3=True)


def _maybe_pick_heuristic_seat(num_players: int, rng) -> Optional[int]:
    if rng.random() < Config.VS_HEURISTIC_FRACTION:
        return rng.randrange(num_players)
    return None


def _maybe_pick_frozen_seats(num_players: int, frozen_pool, rng) -> Dict[int, object]:
    """Optionally assign 0..1 seats per game to a randomly-picked frozen network.
    Returns dict {seat: frozen_net}. If frozen_pool is empty, returns {}.
    """
    if not frozen_pool or rng.random() >= Config.FREEZE_OPP_FRACTION:
        return {}
    seat = rng.randrange(num_players)
    net = rng.choice(frozen_pool)
    return {seat: net}


def _value_vec_from_env(env: GameEnv) -> np.ndarray:
    """Per-player normalized teacher final_score, padded to MAX_PLAYERS with 0."""
    vec = np.zeros(MAX_PLAYERS, dtype=np.float32)
    for p in range(env.num_players):
        vec[p] = ts.normalized_value_target(env.compute_final_score(p))
    return vec


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


# ── Batched self-play (FAST, used in phase 2 by default) ─────────────


def _net_choose_action(network, device, env: GameEnv, player: int,
                       temperature: float, use_greedy: bool):
    state = env.get_state_tensor(player)
    mask = env.get_legal_mask(player)
    seat_t = torch.tensor([player], device=device)
    with torch.no_grad():
        out = network(state.unsqueeze(0).to(device),
                      mask.unsqueeze(0).to(device),
                      current_seat=seat_t)
    policy = out["policy"][0].cpu().numpy()
    legal = get_legal_actions(mask)

    if use_greedy:
        action = int(np.argmax(policy))
    elif temperature != 1.0 and temperature > 0:
        log_p = np.log(policy + 1e-8) / temperature
        exp_p = np.zeros_like(policy)
        if legal:
            mx = log_p[legal].max()
            for a in legal:
                exp_p[a] = np.exp(log_p[a] - mx)
            total = exp_p.sum()
            if total > 0:
                exp_p /= total
            else:
                for a in legal:
                    exp_p[a] = 1.0 / len(legal)
        # Defensive renorm - np.random.choice raises if p doesn't sum to 1.0
        # exactly. fp32 softmax can drift by a few ULPs.
        s = exp_p.sum()
        if s > 0 and abs(s - 1.0) > 1e-9:
            exp_p = exp_p / s
        action = int(np.random.choice(len(exp_p), p=exp_p))
    else:
        # Same defensive renorm for the temperature=1.0 / direct-softmax path
        p = policy.astype(np.float64)
        s = p.sum()
        if s > 0:
            p = p / s
            action = int(np.random.choice(len(p), p=p))
        else:
            action = int(np.argmax(policy))
    return action, policy, state, mask


def generate_games_batched_v3(
    network,                                # the live network
    device: torch.device,
    board: HexBoard,
    num_games: int = 32,
    temperature: float = 1.0,
    start_game_id: int = 0,
    iteration: int = 0,
    rng: Optional[_random.Random] = None,
    frozen_pool: Optional[List] = None,     # list of frozen NexusNetV3 (or None)
) -> Tuple[List[List[Dict]], List[Dict]]:
    """Run num_games sequentially with per-step batched GPU inference.

    Sequential per-game (not interleaved-batched) for v3 to keep the
    aux-target bookkeeping simple. We still use batched forward passes
    inside each game's per-move call since GPU latency dominates.
    """
    network.eval()
    rng = rng or _random.Random()
    if frozen_pool is None:
        frozen_pool = []

    trajectories: List[List[Dict]] = [[] for _ in range(num_games)]
    summaries: List[Dict] = []

    heuristic_agent = HeuristicAgent(board)

    for i in range(num_games):
        N = _sample_num_players(iteration, rng=rng)
        env = GameEnv(board, num_players=N)
        env.reset(random_colors=True, rng=rng)
        heuristic_seat = _maybe_pick_heuristic_seat(N, rng)
        frozen_seats = _maybe_pick_frozen_seats(N, frozen_pool, rng)
        # Ensure a heuristic seat doesn't double-assign onto a frozen seat
        if heuristic_seat is not None and heuristic_seat in frozen_seats:
            del frozen_seats[heuristic_seat]
        nexus_seats = [p for p in range(N)
                       if p != heuristic_seat and p not in frozen_seats]

        traj = trajectories[i]
        # Pending entries awaiting the *next* move from another seat -
        # the action chosen on the very next iteration of this loop will
        # fill in `opp_action` and `opp_legal_mask` for these entries.
        pending_opp_entries: List[int] = []

        while not env.is_done():
            p = env.current_player
            use_greedy = env.move_count >= TEMP_SWITCH_MOVES

            # 2) Pick an action depending on seat type
            if heuristic_seat is not None and p == heuristic_seat:
                state = env.get_state_tensor(p)
                mask = env.get_legal_mask(p)
                action = heuristic_agent.choose_move(env, p)
                policy_target = np.zeros(Config.ACTION_SPACE, dtype=np.float32)
                policy_target[action] = 1.0
                is_heuristic = True
            elif p in frozen_seats:
                action, policy_target, state, mask = _net_choose_action(
                    frozen_seats[p], device, env, p, temperature, use_greedy
                )
                is_heuristic = False    # don't keep frozen-net entries for training
            else:
                action, policy_target, state, mask = _net_choose_action(
                    network, device, env, p, temperature, use_greedy
                )
                is_heuristic = False

            # First - fill any pending entries with this current move's
            # (action, legal_mask). Those entries were appended on the
            # PREVIOUS turn and were waiting for the next seat to move.
            mask_np = mask.numpy() if hasattr(mask, "numpy") else np.asarray(mask)
            if pending_opp_entries:
                for idx in pending_opp_entries:
                    traj[idx]["opp_action"] = int(action)
                    traj[idx]["opp_legal_mask"] = mask_np
                pending_opp_entries = []

            # We only train on entries from heuristic seats and live-network
            # seats. Frozen-pool seats provide diverse opponents but we
            # don't use their states as training data (they would bias the
            # buffer toward stale-policy decisions).
            train_this_entry = (p in nexus_seats) or (p == heuristic_seat)

            if train_this_entry:
                entry_idx = len(traj)
                traj.append({
                    "state": state.numpy() if hasattr(state, "numpy") else np.asarray(state),
                    "action": int(action),
                    "policy_target": np.asarray(policy_target, dtype=np.float32),
                    "legal_mask": mask_np,
                    "player": int(p),
                    "n_players": int(N),
                    "move_count": env.move_count,
                    "game_id": start_game_id + i,
                    "reward": 0.0,
                    "is_heuristic": is_heuristic,
                    "opp_action": -1,
                    "opp_legal_mask": np.zeros(Config.ACTION_SPACE, dtype=np.bool_),
                    "plies_remaining": 0.0,
                    "plies_valid": False,
                })
                pending_opp_entries.append(entry_idx)

            env.step(int(action))

        # ── End of game: backfill value vector + plies remaining ──
        value_vec = _value_vec_from_env(env)
        total_plies = env.move_count

        for entry in traj:
            entry["value_vec"] = value_vec.copy()
            seat = entry["player"]
            entry["value_target"] = float(value_vec[seat])
            plies_left = max(0, total_plies - entry["move_count"])
            # Cap at 4.0 (i.e. 800 plies) - beyond this the regression
            # target is meaningless. Most games < 200 plies.
            entry["plies_remaining"] = min(4.0, float(plies_left) / 200.0)
            entry["plies_valid"] = True

        # Self-imitation for top-half scoring players (kept from v2 - works)
        scores = [env.compute_final_score(p) for p in range(env.num_players)]
        if scores:
            sorted_scores = sorted(scores, reverse=True)
            cutoff = sorted_scores[max(0, (len(scores) - 1) // 2)]
            for entry in traj:
                if entry.get("is_heuristic"):
                    continue
                if scores[entry["player"]] >= cutoff:
                    onehot = np.zeros(entry["policy_target"].shape, dtype=np.float32)
                    onehot[entry["action"]] = 1.0
                    entry["policy_target"] = onehot

        summaries.append(_summarize_game(
            env, iteration=iteration, game_id=start_game_id + i,
            nexus_seats=nexus_seats, heuristic_seat=heuristic_seat,
        ))

    return trajectories, summaries
