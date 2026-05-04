"""Self-play game generation for N-player training (v3).

Two modes:
1. Batched policy self-play: runs N_games simultaneously with batched GPU
   inference. ~100x faster than MCTS - used to fill the buffer quickly.
2. Sequential MCTS self-play: one game at a time with Gumbel MCTS. Slower,
   but produces MCTS-improved policy targets.

Both modes:
- Sample N (number of players) per game per Config.NUM_PLAYERS_CURRICULUM.
- Optionally replace one seat with HeuristicAgent (vs-heuristic anchor).
- Compute value_target as a single scalar (normalized teacher final_score).
"""

from __future__ import annotations

import random as _random
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from config import Config
from core.board import HexBoard
from core.game_env import GameEnv
from core.action_space import get_legal_actions, encode_action
from mcts.gumbel_mcts import GumbelMCTS
from training.heuristic_agent import HeuristicAgent


# After this many game moves, switch from temperature sampling to greedy.
# Empirically critical: without this, 99%+ of games timed out at MAX_MOVES
# because both players sampled noisily forever and never committed to a
# winning sequence. With switch=30, each player gets ~15-30 exploration
# moves (depending on N) then commits to argmax → games actually finish.
# Standard AlphaZero technique.
TEMP_SWITCH_MOVES: int = 30


# ── Helpers ──────────────────────────────────────────────────────


def _sample_num_players(iteration: int, rng=None) -> int:
    """Sample N according to the curriculum (Config.NUM_PLAYERS_CURRICULUM)."""
    return Config.sample_num_players(iteration, rng=rng)


def _maybe_pick_heuristic_seat(num_players: int, rng) -> Optional[int]:
    """With Config.VS_HEURISTIC_FRACTION probability, replace one seat with
    the heuristic agent. Returns the seat index, or None."""
    if rng.random() < Config.VS_HEURISTIC_FRACTION:
        return rng.randrange(num_players)
    return None


def _to_action(env: GameEnv, player: int, piece_pos: int, dest: int) -> int:
    """Convert (piece_pos, dest) to encoded action for nexus's action space."""
    pieces = env.pieces[player]
    piece_id = pieces.index(piece_pos)
    return encode_action(piece_id, dest)


def _heuristic_action(env: GameEnv, heuristic: HeuristicAgent, player: int) -> int:
    """Pick an action via the HeuristicAgent for `player`."""
    return heuristic.choose_move(env, player)


def _summarize_game(env: GameEnv, iteration: int, game_id: int,
                    nexus_seats: List[int],
                    heuristic_seat: Optional[int],
                    num_moves_played: int) -> Dict:
    """Build a per-game summary record (one line for self_play_summaries.jsonl)."""
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


# ── Batched Policy Self-Play (FAST) ──────────────────────────────


def generate_games_batched(
    network,
    device: torch.device,
    board: HexBoard,
    num_games: int = 32,
    temperature: float = 1.0,
    start_game_id: int = 0,
    iteration: int = 0,
    rng: Optional[_random.Random] = None,
) -> Tuple[List[List[Dict]], List[Dict]]:
    """Run num_games simultaneously with batched GPU forward passes.

    Returns (trajectories, summaries) - both lists of length num_games.
    """
    network.eval()
    rng = rng or _random.Random()

    envs: List[GameEnv] = []
    heuristic_seats: List[Optional[int]] = []
    heuristic_agent = HeuristicAgent(board)
    for _ in range(num_games):
        N = _sample_num_players(iteration, rng=rng)
        env = GameEnv(board, num_players=N)
        env.reset()
        envs.append(env)
        heuristic_seats.append(_maybe_pick_heuristic_seat(N, rng))

    trajectories: List[List[Dict]] = [[] for _ in range(num_games)]
    nexus_seats: List[List[int]] = [
        [p for p in range(env.num_players) if p != heuristic_seats[i]]
        for i, env in enumerate(envs)
    ]
    active = list(range(num_games))

    with torch.no_grad():
        while active:
            # Split games by current-player role: heuristic seat or NEXUS seat.
            heur_now = []  # (i, env, player)
            net_now = []   # (i, env, player)
            for i in active:
                env = envs[i]
                p = env.current_player
                if heuristic_seats[i] is not None and p == heuristic_seats[i]:
                    heur_now.append((i, env, p))
                else:
                    net_now.append((i, env, p))

            # Apply heuristic seats first (no batched GPU call needed)
            # DAgger: train on heuristic's moves too - heuristic is a strong
            # teacher whose decisions in Phase-2 state distributions are
            # different from Phase-1 heuristic-vs-heuristic. This exposes the
            # network to heuristic responses against weak network play, which
            # the network needs to learn to imitate.
            next_active = set()
            for (i, env, p) in heur_now:
                # Capture state BEFORE the move (training input)
                state = env.get_state_tensor(p)
                legal_mask = env.get_legal_mask(p)
                action = _heuristic_action(env, heuristic_agent, p)
                # One-hot policy target at heuristic's choice (DAgger / IL)
                onehot = np.zeros(Config.ACTION_SPACE, dtype=np.float32)
                onehot[action] = 1.0
                trajectories[i].append({
                    "state": state.numpy(),
                    "action": action,
                    "policy_target": onehot,
                    "legal_mask": legal_mask.numpy(),
                    "player": p,
                    "move_count": env.move_count,
                    "game_id": start_game_id + i,
                    "reward": 0.0,
                    "is_heuristic": True,  # marker; backfill skips overwrite
                })
                reward, done = env.step(action)
                trajectories[i][-1]["reward"] = reward
                if not done:
                    next_active.add(i)

            # Now batched forward pass for NEXUS seats
            if net_now:
                states = [env.get_state_tensor(p) for (_, env, p) in net_now]
                masks = [env.get_legal_mask(p) for (_, env, p) in net_now]
                state_batch = torch.stack(states).to(device)
                mask_batch = torch.stack(masks).to(device)
                out = network(state_batch, mask_batch)
                policies = out["policy"].cpu().numpy()

                for j, (i, env, p) in enumerate(net_now):
                    policy = policies[j]
                    legal = get_legal_actions(masks[j])

                    # ── Within-game temperature schedule (AlphaZero) ──
                    # First TEMP_SWITCH_MOVES game-moves: explore with given
                    # `temperature`. After: switch to greedy (argmax).
                    # Without this, games never commit to a winning sequence
                    # and 99% time out. Empirically: pre-fix only 0.4% of
                    # games had a winner. Greedy late-game lets the agent
                    # actually finish what it started.
                    use_greedy = env.move_count >= TEMP_SWITCH_MOVES

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
                        action = int(np.random.choice(len(exp_p), p=exp_p))
                    else:
                        action = int(np.random.choice(len(policy), p=policy))

                    trajectories[i].append({
                        "state": states[j].numpy(),
                        "action": action,
                        "policy_target": policy,
                        "legal_mask": masks[j].numpy(),
                        "player": p,
                        "move_count": env.move_count,
                        "game_id": start_game_id + i,
                        "reward": 0.0,
                    })
                    reward, done = env.step(action)
                    trajectories[i][-1]["reward"] = reward
                    if not done:
                        next_active.add(i)

            active = sorted(next_active)

    # Backfill targets:
    # - value_target: scalar normalized teacher_score (standard AlphaZero)
    # - policy_target: SELF-IMITATION - for top-half-scoring players, replace
    #   the raw network policy with a one-hot at the action they actually took.
    #   This gives the policy head a real learning signal (imitate winning
    #   moves) instead of distilling itself. Bottom-half players keep their
    #   raw policy target (effectively zero gradient - same as before).
    import numpy as _np
    summaries = []
    for i in range(num_games):
        env = envs[i]
        scores = [env.compute_final_score(p) for p in range(env.num_players)]
        # Top-half cutoff: rank 1 in N=2, ranks 1 in N=3, ranks 1-2 in N=4..6
        # We use median: above-or-equal-median = "top half" = winners to imitate.
        if len(scores) > 0:
            sorted_scores = sorted(scores, reverse=True)
            cutoff_idx = max(0, (len(scores) - 1) // 2)
            cutoff = sorted_scores[cutoff_idx]
        else:
            cutoff = 0.0

        for entry in trajectories[i]:
            entry["value_target"] = float(env.compute_value_target(entry["player"]))
            # Heuristic-derived entries (DAgger) already have one-hot targets;
            # leave them alone.
            if entry.get("is_heuristic"):
                continue
            # Self-imitation for top-half players: one-hot at the action taken
            if scores[entry["player"]] >= cutoff:
                onehot = _np.zeros(entry["policy_target"].shape, dtype=_np.float32)
                onehot[entry["action"]] = 1.0
                entry["policy_target"] = onehot

        summaries.append(_summarize_game(
            env, iteration=iteration, game_id=start_game_id + i,
            nexus_seats=nexus_seats[i], heuristic_seat=heuristic_seats[i],
            num_moves_played=env.move_count,
        ))

    return trajectories, summaries


# ── Sequential MCTS Self-Play (QUALITY) ──────────────────────────


def generate_self_play_game(
    network,
    device: torch.device,
    board: HexBoard,
    num_simulations: int = 32,
    temperature: float = 1.0,
    add_noise: bool = True,
    game_id: int = 0,
    iteration: int = 0,
    rng: Optional[_random.Random] = None,
) -> Tuple[List[Dict], Dict]:
    """Generate one self-play game using MCTS. Returns (trajectory, summary)."""
    network.eval()
    rng = rng or _random.Random()

    N = _sample_num_players(iteration, rng=rng)
    heuristic_seat = _maybe_pick_heuristic_seat(N, rng)
    heuristic_agent = HeuristicAgent(board) if heuristic_seat is not None else None

    env = GameEnv(board, num_players=N)
    env.reset()

    mcts = GumbelMCTS(
        network, device,
        num_simulations=num_simulations,
        add_noise=add_noise,
    )

    trajectory = []
    nexus_seats = [p for p in range(N) if p != heuristic_seat]

    with torch.no_grad():
        while not env.is_done():
            player = env.current_player

            if player == heuristic_seat:
                action = heuristic_agent.choose_move(env, player)
                env.step(action)
                continue

            state = env.get_state_tensor(player)
            legal_mask = env.get_legal_mask(player)

            best_action, improved_policy, root_value = mcts.search(env)

            if temperature > 0 and temperature != 1.0:
                log_policy = np.log(improved_policy + 1e-8) / temperature
                legal = get_legal_actions(legal_mask)
                exp_vals = np.zeros_like(improved_policy)
                if legal:
                    mx = log_policy[legal].max()
                    for a in legal:
                        exp_vals[a] = np.exp(log_policy[a] - mx)
                    total = exp_vals.sum()
                    if total > 0:
                        exp_vals /= total
                    else:
                        for a in legal:
                            exp_vals[a] = 1.0 / len(legal)
                action = int(np.random.choice(len(exp_vals), p=exp_vals))
            else:
                action = int(best_action)

            trajectory.append({
                "state": state.numpy(),
                "action": action,
                "policy_target": improved_policy,
                "legal_mask": legal_mask.numpy(),
                "player": player,
                "move_count": env.move_count,
                "game_id": game_id,
                "reward": 0.0,
            })
            reward, done = env.step(action)
            trajectory[-1]["reward"] = reward

    for entry in trajectory:
        entry["value_target"] = float(env.compute_value_target(entry["player"]))

    summary = _summarize_game(
        env, iteration=iteration, game_id=game_id,
        nexus_seats=nexus_seats, heuristic_seat=heuristic_seat,
        num_moves_played=env.move_count,
    )
    return trajectory, summary


def generate_games_sequential(
    network,
    device: torch.device,
    board: HexBoard,
    num_games: int = 32,
    num_simulations: int = 32,
    temperature: float = 1.0,
    start_game_id: int = 0,
    iteration: int = 0,
    rng: Optional[_random.Random] = None,
) -> Tuple[List[List[Dict]], List[Dict]]:
    """Generate sequential MCTS self-play games. Returns (trajectories, summaries)."""
    network.eval()
    rng = rng or _random.Random()
    trajectories = []
    summaries = []
    for i in range(num_games):
        traj, summ = generate_self_play_game(
            network, device, board,
            num_simulations=num_simulations,
            temperature=temperature,
            add_noise=True,
            game_id=start_game_id + i,
            iteration=iteration,
            rng=rng,
        )
        trajectories.append(traj)
        summaries.append(summ)
        if (i + 1) % 10 == 0:
            print(f"    MCTS self-play: {i+1}/{num_games} games done")
    return trajectories, summaries


# ── Main Entry Point (used by trainer) ───────────────────────────


def generate_games_parallel(
    network,
    device: torch.device,
    board: HexBoard,
    num_games: int = 32,
    num_simulations: int = 32,
    temperature: float = 1.0,
    start_game_id: int = 0,
    num_workers: int = None,
    use_mcts: bool = True,
    iteration: int = 0,
    rng: Optional[_random.Random] = None,
) -> Tuple[List[List[Dict]], List[Dict]]:
    """Generate self-play games. Returns (trajectories, summaries)."""
    if use_mcts:
        return generate_games_sequential(
            network, device, board,
            num_games=num_games,
            num_simulations=num_simulations,
            temperature=temperature,
            start_game_id=start_game_id,
            iteration=iteration,
            rng=rng,
        )
    else:
        return generate_games_batched(
            network, device, board,
            num_games=num_games,
            temperature=temperature,
            start_game_id=start_game_id,
            iteration=iteration,
            rng=rng,
        )
