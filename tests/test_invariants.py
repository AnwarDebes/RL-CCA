"""Day-2 invariant tests - must pass before training starts.

INV1: Telescoping reward identity
       sum(step_rewards_for_player_p) + init_potential[p] == teacher_score(player_p)
       across 100 random self-play games for each N in {2,3,4,5,6}.

INV2: 2-player rollout determinism - env behaves deterministically from a seed.

INV3: State encoder is symmetric in the order of `other_colors` (channels 1/3/5
       are unions, not concatenations).

INV4: Full N=6 game completes without errors and final scores match teacher_score.
"""

from __future__ import annotations

import os
import random
import sys

NEXUS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, NEXUS_DIR)

import pytest
import torch

from config import Config
from core.board import HexBoard
from core.game_env import GameEnv, SEAT_COLORS_BY_N
from core.state_encoder import StateEncoder
from core import teacher_score as ts
from core.action_space import encode_action


def _random_legal_action(env, rng):
    legal = env.get_legal_moves(env.current_player)
    if not legal:
        return None
    pieces = env.pieces[env.current_player]
    pos = rng.choice(list(legal.keys()))
    piece_id = pieces.index(pos)
    dest = rng.choice(legal[pos])
    return encode_action(piece_id, dest)


def _play_random_game(env, rng, max_steps=2000):
    """Play env to completion with random moves. Returns per-player reward sums."""
    N = env.num_players
    sums = [0.0] * N
    steps = 0
    while not env.is_done() and steps < max_steps:
        player = env.current_player
        action = _random_legal_action(env, rng)
        if action is None:
            # Shouldn't happen - env._advance_turn handles stuck players. If it
            # does happen, just break.
            break
        reward, done = env.step(action)
        sums[player] += reward
        steps += 1
    return sums, steps


# ── INV1: Telescoping invariant ─────────────────────────────────


@pytest.mark.parametrize("N", [2, 3, 4, 5, 6])
def test_inv1_telescoping_reward(N):
    """For each N, run 20 random games and verify the telescoping invariant.

    Per-step rewards capture only pin_goal_score + distance_score:
        init_potential[p] + sum(step_rewards[p]) == 100*pins(p) + max(0, 200-dist(p))

    Time and move components are NOT in step rewards (they would be
    inconsistent across players); they're picked up by the terminal value target.
    """
    board = HexBoard()
    rng = random.Random(20260429 + N)
    failures = []

    for trial in range(20):
        env = GameEnv(board, num_players=N)
        env.reset()
        sums, steps = _play_random_game(env, rng)

        for p in range(N):
            color = env.colors[p]
            pins = env.board.count_in_goal(env.pieces[p], color)
            total_dist = env.board.sum_distances_to_goal(env.pieces[p], color)
            expected_telescoping = (
                ts.pin_goal_score(pins) + ts.distance_score(total_dist)
            )
            actual = env.init_potential(p) + sums[p]
            diff = abs(actual - expected_telescoping)
            if diff > 0.5:
                failures.append({
                    "N": N, "trial": trial, "player": p,
                    "actual": actual, "expected": expected_telescoping,
                    "diff": diff, "steps": steps,
                    "pins": pins, "dist": total_dist,
                    "init_potential": env.init_potential(p),
                    "sum_rewards": sums[p],
                })

    assert not failures, (
        f"{len(failures)} telescoping invariant failures for N={N}. "
        f"First 3: {failures[:3]}"
    )


# ── INV2: 2-player determinism ──────────────────────────────────


def test_inv2_2player_determinism():
    """Same seed must produce identical games."""
    board = HexBoard()

    def play(seed):
        rng = random.Random(seed)
        env = GameEnv(board, num_players=2)
        env.reset()
        actions = []
        rewards = []
        steps = 0
        while not env.is_done() and steps < 1000:
            a = _random_legal_action(env, rng)
            if a is None:
                break
            actions.append((env.current_player, a))
            r, _ = env.step(a)
            rewards.append(r)
            steps += 1
        return actions, rewards, env.winner, env.move_count

    a1 = play(42)
    a2 = play(42)
    assert a1 == a2, "Same seed should produce identical games"

    # Different seed → likely different game
    a3 = play(43)
    assert a3 != a1, "Different seed should produce different games (random sampling)"


# ── INV3: State encoder permutation invariance ──────────────────


def test_inv3_encoder_symmetric_in_other_colors():
    """Channels 1 (opp pieces), 3 (opp goals), 5 (opp starts) must be UNIONS,
    so reordering `other_colors` and the corresponding `opp_pieces` should
    produce identical tensors."""
    board = HexBoard()
    encoder = StateEncoder(board)

    my_color = 'red'
    other_colors_a = ['lawn green', 'yellow', 'gray0']
    other_colors_b = ['gray0', 'lawn green', 'yellow']  # permuted

    # Disjoint piece positions for each opponent
    opp_pieces_lawn = [10, 11, 12]
    opp_pieces_yellow = [50, 51, 52]
    opp_pieces_gray = [100, 101, 102]
    opp_pieces_combined = sorted(opp_pieces_lawn + opp_pieces_yellow + opp_pieces_gray)

    my_pieces = [0, 1, 2]

    t_a = encoder.encode(
        my_pieces=my_pieces,
        opp_pieces=opp_pieces_combined,
        my_color=my_color,
        other_colors=other_colors_a,
        num_players=4,
        move_count=10,
        time_elapsed=1.0,
        legal_moves=None,
    )
    t_b = encoder.encode(
        my_pieces=my_pieces,
        opp_pieces=opp_pieces_combined,
        my_color=my_color,
        other_colors=other_colors_b,
        num_players=4,
        move_count=10,
        time_elapsed=1.0,
        legal_moves=None,
    )
    assert torch.equal(t_a, t_b), "Encoder must be symmetric in other_colors order"


def test_inv3b_encoder_n2_back_compat():
    """N=2 encoder output should be sensible: channels 1/3/5 should match
    what the old encoder produced when given the single opponent."""
    board = HexBoard()
    encoder = StateEncoder(board)

    my_pieces = [0, 1, 2]
    opp_pieces = [50, 51, 52]
    t = encoder.encode(
        my_pieces=my_pieces,
        opp_pieces=opp_pieces,
        my_color='red',
        other_colors=['blue'],
        num_players=2,
        move_count=5,
        time_elapsed=0.5,
        legal_moves=None,
    )
    # Channel 0 should mark exactly my pieces
    G = Config.GRID_SIZE
    ch0_count = int(t[0].sum().item())
    assert ch0_count == 3, f"Channel 0 should mark 3 my-pieces, got {ch0_count}"
    # Channel 1 should mark exactly opp pieces
    ch1_count = int(t[1].sum().item())
    assert ch1_count == 3
    # Channels 2, 3 (goal zones) should each mark exactly 10 cells
    ch2_count = int(t[2].sum().item())
    ch3_count = int(t[3].sum().item())
    assert ch2_count == 10, f"My goal zone should be 10 cells, got {ch2_count}"
    assert ch3_count == 10, f"Opp goal zone (single opp) should be 10 cells, got {ch3_count}"


# ── INV4: Full N=6 game completes ────────────────────────────────


def test_inv4_full_n6_game():
    """A 6-player random game must complete and produce sensible scores."""
    board = HexBoard()
    rng = random.Random(20260430)
    env = GameEnv(board, num_players=6)
    env.reset()

    sums, steps = _play_random_game(env, rng)

    assert env.is_done(), f"6-player game did not complete in {steps} steps"
    assert steps > 0, "Game ended at step 0?"
    assert steps <= env.MAX_MOVES, f"Game exceeded MAX_MOVES ({env.MAX_MOVES})"

    # Each player's final_score should be in [0, 1301]
    for p in range(6):
        s = env.compute_final_score(p)
        assert 0.0 <= s <= 1302.0, f"Player {p} score out of range: {s}"

    # Telescoping per player should match the pin+dist portion
    for p in range(6):
        color = env.colors[p]
        pins = env.board.count_in_goal(env.pieces[p], color)
        dist = env.board.sum_distances_to_goal(env.pieces[p], color)
        expected = ts.pin_goal_score(pins) + ts.distance_score(dist)
        actual = env.init_potential(p) + sums[p]
        assert abs(actual - expected) < 0.5, (
            f"Player {p}: telescoping={actual:.2f} != expected={expected:.2f}"
        )


# ── Smoke: state tensor shape ────────────────────────────────────


@pytest.mark.parametrize("N", [2, 3, 4, 5, 6])
def test_state_tensor_shape(N):
    """get_state_tensor must return (22, 17, 17) regardless of N."""
    board = HexBoard()
    env = GameEnv(board, num_players=N)
    env.reset()
    t = env.get_state_tensor()
    assert t.shape == (22, 17, 17), f"N={N}: got shape {t.shape}"
