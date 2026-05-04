"""Chinese Checkers game environment for N-player self-play (N=2..6).

Aligned with teacher's tournament rules (verified by tests/test_pre_flight.py):
- Win condition: a player has all NUM_PIECES pins in opposite-color zone.
- Draw rule: when N-1 players have no legal moves, the remaining one wins.
- Move rule: each turn picks ONE destination (single-step OR hop chain).
- Move-type exclusivity is implicit in the legal-move generator.

Telescoping reward - design invariant:

  sum_over_steps( reward[player_p, step] ) + init_potential[p]
    == pin_goal_score(p) + distance_score(p)
    == teacher_score(p) - time_score(p) - move_score(p)

where:
  init_potential[p] = max(0, 200 - sum_distances_to_goal(start_pieces, color_p))

The per-step reward is JUST the dense shaping signal:
  step_reward = (prev_dist - new_dist)  +  100 * (new_pins - prev_pins)

time_score and move_score are NOT in the per-step reward - they would be
inconsistent (only the terminal-move player would receive them). Instead,
they are captured by the terminal value target = normalized full teacher_score.
The value head learns to predict the full teacher_score, including time/move
components (which are deterministic functions of move count).

This is verified in tests/test_invariants.py::test_inv1_telescoping_reward.
"""

from __future__ import annotations

import bisect
from collections import deque
from typing import Dict, List, Optional, Set, Tuple

import torch

from config import Config
from core.board import HexBoard
from core.action_space import decode_action, build_legal_mask
from core.state_encoder import StateEncoder
from core import teacher_score as ts


# Color seating order - matches teacher's PRIMARY + COMPLEMENT pairing
# (game.py:51-52). For N players, the first N entries below are seated.
# Used as the default when randomized colors are not requested.
SEAT_COLORS_BY_N: Dict[int, List[str]] = {
    2: ['red', 'blue'],
    3: ['red', 'lawn green', 'yellow'],
    4: ['red', 'blue', 'lawn green', 'gray0'],
    5: ['red', 'blue', 'lawn green', 'gray0', 'yellow'],
    6: ['red', 'blue', 'lawn green', 'gray0', 'yellow', 'purple'],
}

# Teacher's color/turn constants (game.py:48-49)
PRIMARY_COLOURS: List[str] = ['red', 'lawn green', 'yellow']
COLOUR_ORDER: List[str] = ['red', 'lawn green', 'yellow', 'blue', 'gray0', 'purple']

# Virtual seconds per move during training (proxy for tournament wall clock).
# Combined with GAME_TIME_LIMIT_SEC below, this caps game length to match
# what teacher's 60-second wall-clock would produce.
VIRTUAL_TIME_PER_MOVE: float = 0.10
GAME_TIME_LIMIT_SEC: float = 60.0    # teacher's GAME_TIME_LIMIT (game.py:54)


def _sample_random_colors(num_players: int, rng) -> List[str]:
    """Mirror teacher's color assignment (game.py:107-118): shuffle
    PRIMARY_COLOURS, then pair each primary with its complement.

    For N=2: 1 primary + 1 complement = 1 random pair (e.g. red/blue,
    lawn green/gray0, or yellow/purple).
    For N=3: 3 primaries (1 of each).
    For N=4: 2 primaries + complements = 2 random pairs.
    For N=5: 2 primaries + complements + 1 third primary.
    For N=6: all 6 colors.
    """
    primaries = list(PRIMARY_COLOURS)
    rng.shuffle(primaries)
    complements = {p: Config.COLOR_OPPOSITES[p] for p in primaries}
    seats: List[str] = []
    # Teacher's join logic alternates primary/complement on odd/even seats
    for k in range(num_players):
        if k % 2 == 0:
            seats.append(primaries[k // 2])
        else:
            seats.append(complements[primaries[k // 2]])
    return seats


class GameEnv:
    """N-player Chinese Checkers environment (2 ≤ N ≤ 6)."""

    def __init__(self, board: Optional[HexBoard] = None, num_players: int = 2):
        if num_players not in SEAT_COLORS_BY_N:
            raise ValueError(f"num_players must be in {sorted(SEAT_COLORS_BY_N)}")
        self.board = board or HexBoard()
        self.encoder = StateEncoder(self.board)
        self.num_players: int = num_players
        self.colors: List[str] = list(SEAT_COLORS_BY_N[num_players])
        self.pieces: List[List[int]] = [[] for _ in range(num_players)]
        self.current_player: int = 0
        self.move_count: int = 0
        self.player_move_counts: List[int] = [0] * num_players
        self.player_time_taken: List[float] = [0.0] * num_players
        self.done: bool = False
        self.winner: Optional[int] = None  # player index, or None for cap/total-draw
        self.last_hop_length: int = 0
        # Safety cap. Teacher uses GAME_TIME_LIMIT_SEC=60s wall-clock with no
        # move cap. With VIRTUAL_TIME_PER_MOVE=0.10s, 60s ≈ 600 moves total.
        # We keep a generous upper bound so well-behaved training games can
        # still finish; the time-based cap below mirrors teacher's behavior.
        self.MAX_MOVES: int = max(600, 200 * num_players)
        self.GAME_TIME_LIMIT_SEC: float = GAME_TIME_LIMIT_SEC

        # Internal caches
        self._occupied: Set[int] = set()
        self._legal_cache: Dict[int, Optional[Dict[int, List[int]]]] = {
            p: None for p in range(num_players)
        }
        self._stuck: List[bool] = [False] * num_players
        self._prev_dist: List[int] = [0] * num_players
        self._prev_pins: List[int] = [0] * num_players
        # Constant offsets so sum_rewards + init_potential == teacher_score
        self._init_potential: List[float] = [0.0] * num_players

    # ── reset ────────────────────────────────────────────────────────

    def reset(self, num_players: Optional[int] = None,
              colors: Optional[List[str]] = None,
              random_colors: bool = False,
              rng=None) -> None:
        """Reset to a fresh start position. May change num_players or colors.

        random_colors=True samples a teacher-style random color set (matches
        game.py:107-118). When True, `colors` is ignored.
        """
        if num_players is not None and num_players != self.num_players:
            if num_players not in SEAT_COLORS_BY_N:
                raise ValueError(f"num_players must be in {sorted(SEAT_COLORS_BY_N)}")
            self.num_players = num_players
            self.MAX_MOVES = max(600, 200 * num_players)
        if random_colors:
            import random as _r
            colors = _sample_random_colors(self.num_players, rng or _r)
        elif colors is None:
            colors = SEAT_COLORS_BY_N[self.num_players]
        if len(colors) != self.num_players:
            raise ValueError(
                f"colors len ({len(colors)}) != num_players ({self.num_players})"
            )
        # Match teacher's compute_turn_order (game.py:131-140): take the set
        # of present colors, filter through COLOUR_ORDER, and (optionally)
        # rotate so the "first joiner" goes first. We rotate to a random
        # color when random_colors=True so seat-0 isn't always 'red'.
        present = set(colors)
        ordered = [c for c in COLOUR_ORDER if c in present]
        if random_colors and rng is not None:
            first = rng.choice(ordered)
            i = ordered.index(first)
            ordered = ordered[i:] + ordered[:i]
        self.colors = ordered
        self.pieces = [sorted(self.board.start_zones[c]) for c in self.colors]
        self.current_player = 0
        self.move_count = 0
        self.player_move_counts = [0] * self.num_players
        self.player_time_taken = [0.0] * self.num_players
        self.done = False
        self.winner = None
        self.last_hop_length = 0
        self._rebuild_occupied()
        self._legal_cache = {p: None for p in range(self.num_players)}
        self._stuck = [False] * self.num_players
        self._prev_dist = [
            self.board.sum_distances_to_goal(self.pieces[p], self.colors[p])
            for p in range(self.num_players)
        ]
        self._prev_pins = [
            self.board.count_in_goal(self.pieces[p], self.colors[p])
            for p in range(self.num_players)
        ]
        # init_potential: starting distance_score for each player.
        self._init_potential = [
            max(0.0, 200.0 - float(self._prev_dist[p]))
            for p in range(self.num_players)
        ]

    # ── geometry helpers ─────────────────────────────────────────────

    def _rebuild_occupied(self) -> None:
        s: Set[int] = set()
        for p_pieces in self.pieces:
            s.update(p_pieces)
        self._occupied = s

    def get_occupied(self) -> Set[int]:
        return self._occupied

    def get_legal_moves(self, player: Optional[int] = None) -> Dict[int, List[int]]:
        """Dict {piece_pos: [destination_indices]} for one player."""
        if player is None:
            player = self.current_player

        cached = self._legal_cache[player]
        if cached is not None:
            return cached

        moves: Dict[int, List[int]] = {}
        occupied = self._occupied
        board_neighbors = self.board.neighbors
        board_index = self.board.index_of
        cell_q = self.board.cell_q
        cell_r = self.board.cell_r
        directions = Config.DIRECTIONS

        for piece_pos in self.pieces[player]:
            dests: List[int] = []

            # Single-step moves
            for nbr in board_neighbors[piece_pos]:
                if nbr not in occupied:
                    dests.append(nbr)

            # Multi-hop chains via BFS
            visited: Set[int] = {piece_pos}
            queue: deque = deque([piece_pos])
            while queue:
                cur = queue.popleft()
                cq, cr = cell_q[cur], cell_r[cur]
                for dq, dr in directions:
                    mq, mr = cq + dq, cr + dr
                    mid = board_index.get((mq, mr))
                    if mid is None or mid not in occupied:
                        continue
                    lq, lr = mq + dq, mr + dr
                    land = board_index.get((lq, lr))
                    if land is None or land in occupied or land in visited:
                        continue
                    visited.add(land)
                    dests.append(land)
                    queue.append(land)

            if dests:
                moves[piece_pos] = dests

        self._legal_cache[player] = moves
        return moves

    # ── step / turn rotation ─────────────────────────────────────────

    def step(self, action: int) -> Tuple[float, bool]:
        """Apply `action` (encoded as piece_id * 121 + dest_idx) for current player.
        Returns (reward_for_current_player, done)."""
        if self.done:
            return 0.0, True

        piece_id, dest_idx = decode_action(action)
        player = self.current_player
        color = self.colors[player]
        pieces = self.pieces[player]
        if piece_id >= len(pieces):
            raise ValueError(
                f"Invalid piece_id {piece_id} (player {player} has {len(pieces)} pieces)"
            )
        piece_pos = pieces[piece_id]

        # Pre-move metrics
        dist_table = self.board._min_dist_to_goal[color]
        goal_set = self.board._goal_set[color]
        prev_total_dist = self._prev_dist[player]
        prev_pins_in_goal = self._prev_pins[player]
        hop_length = self.board._dist_table[piece_pos][dest_idx]

        # Apply move
        pieces.pop(piece_id)
        bisect.insort(pieces, dest_idx)
        self._occupied.discard(piece_pos)
        self._occupied.add(dest_idx)
        self.move_count += 1
        self.player_move_counts[player] += 1
        self.player_time_taken[player] += VIRTUAL_TIME_PER_MOVE
        self.last_hop_length = hop_length

        # Post-move metrics for the moving player
        new_total_dist = sum(dist_table[p] for p in pieces)
        new_pins_in_goal = sum(1 for p in pieces if p in goal_set)
        self._prev_dist[player] = new_total_dist
        self._prev_pins[player] = new_pins_in_goal

        # Invalidate legal-move caches and unstuck the moving player
        for p in range(self.num_players):
            self._legal_cache[p] = None
        self._stuck[player] = False

        # Per-step (non-terminal portion) reward
        reward = self._step_reward_components(
            prev_total_dist, new_total_dist,
            prev_pins_in_goal, new_pins_in_goal,
        )

        # Check win condition for the moving player
        terminal = False
        if new_pins_in_goal == Config.NUM_PIECES:
            self.winner = player
            self.done = True
            terminal = True

        if not terminal:
            # Advance to next non-stuck player; updates self._stuck as it goes
            advanced_ok = self._advance_turn()

            if not advanced_ok:
                # All N players are stuck → total draw, no winner
                self.winner = None
                self.done = True
                terminal = True
            else:
                # "N-1 stuck = remaining wins" - STATELESS check, mirrors
                # teacher (game.py:464-474): re-derive who can move RIGHT
                # NOW from the live board, don't rely on cached flags. (v2
                # used a stale-flag cache that could declare premature wins.)
                live_movers = [
                    p for p in range(self.num_players)
                    if len(self.get_legal_moves(p)) > 0
                ]
                if len(live_movers) == 1:
                    self.winner = live_movers[0]
                    self.done = True
                    terminal = True
                elif len(live_movers) == 0:
                    self.winner = None    # full DRAW
                    self.done = True
                    terminal = True

            # Time-based termination - mirrors teacher's GAME_TIME_LIMIT
            # (game.py:54, ensure_time_limits). With virtual 0.10s/move, 60s
            # ≈ 600 moves. Whichever fires first.
            total_virtual_time = sum(self.player_time_taken)
            if not terminal and total_virtual_time >= self.GAME_TIME_LIMIT_SEC:
                self.winner = None
                self.done = True
                terminal = True

            # Move-count safety cap (last-resort)
            if not terminal and self.move_count >= self.MAX_MOVES:
                self.winner = None
                self.done = True
                terminal = True

        return reward, self.done

    def _step_reward_components(self, prev_dist: int, new_dist: int,
                                prev_pins: int, new_pins: int) -> float:
        """Per-step shaping reward (dense signal).

        Sum across a player's turns + init_potential equals
        (pin_goal_score + distance_score) - the score components the player
        directly affects via moves. time_score and move_score are picked up
        by the terminal value target (= normalized full teacher_score)."""
        delta_dist = prev_dist - new_dist           # positive = closer to goal
        delta_pins = new_pins - prev_pins           # positive = pin reached goal
        return float(delta_dist) + 100.0 * float(delta_pins)

    def _advance_turn(self) -> bool:
        """Rotate current_player to next non-stuck player. Updates self._stuck.
        Returns False if every player is stuck (no one can move)."""
        for _ in range(self.num_players):
            self.current_player = (self.current_player + 1) % self.num_players
            legal = self.get_legal_moves(self.current_player)
            if legal:
                self._stuck[self.current_player] = False
                return True
            self._stuck[self.current_player] = True
        return False

    # ── observations ─────────────────────────────────────────────────

    def get_state_tensor(self, player: Optional[int] = None) -> torch.Tensor:
        """Return the NUM_CHANNELS state tensor from `player`'s perspective.

        v3-rebuild: also passes per-opponent slots ordered by relative seat
        offset from `player` (slot 0 = next-to-move opponent, slot 1 =
        two-after-me, etc.). This gives the encoder N=6-aware channels and
        eliminates the v2 "all opponents unioned" information loss.
        """
        if player is None:
            player = self.current_player
        my_color = self.colors[player]
        my_pieces = self.pieces[player]
        opp_pieces: List[int] = []
        other_colors: List[str] = []
        opp_pieces_by_slot: List[List[int]] = []
        opp_colors_by_slot: List[str] = []
        for off in range(1, self.num_players):
            p = (player + off) % self.num_players
            opp_pieces.extend(self.pieces[p])
            other_colors.append(self.colors[p])
            opp_pieces_by_slot.append(list(self.pieces[p]))
            opp_colors_by_slot.append(self.colors[p])
        opp_pieces.sort()
        legal = self.get_legal_moves(player)
        return self.encoder.encode(
            my_pieces=my_pieces,
            opp_pieces=opp_pieces,
            my_color=my_color,
            other_colors=other_colors,
            num_players=self.num_players,
            move_count=self.player_move_counts[player],
            time_elapsed=self.player_time_taken[player],
            opp_pieces_by_slot=opp_pieces_by_slot,
            opp_colors_by_slot=opp_colors_by_slot,
            legal_moves=legal,
        )

    def get_legal_mask(self, player: Optional[int] = None) -> torch.Tensor:
        if player is None:
            player = self.current_player
        legal = self.get_legal_moves(player)
        return build_legal_mask(legal, self.pieces[player])

    # ── status ───────────────────────────────────────────────────────

    def is_done(self) -> bool:
        return self.done

    def get_winner(self) -> Optional[int]:
        return self.winner

    # ── teacher-score-aligned value targets ──────────────────────────

    def init_potential(self, player: int) -> float:
        return self._init_potential[player]

    def compute_final_score(self, player: int) -> float:
        """Player's teacher final_score given the env's current state."""
        color = self.colors[player]
        total_dist = self.board.sum_distances_to_goal(self.pieces[player], color)
        pins = self.board.count_in_goal(self.pieces[player], color)
        return ts.final_score(
            time_taken_sec=self.player_time_taken[player],
            move_count=self.player_move_counts[player],
            pins_in_goal=pins,
            total_distance=float(total_dist),
        )

    def compute_value_target(self, player: int) -> float:
        """Single-scalar value target in [-1, 1] = normalized teacher final_score."""
        return ts.normalized_value_target(self.compute_final_score(player))

    # ── deep copy ────────────────────────────────────────────────────

    def clone(self) -> 'GameEnv':
        env = GameEnv.__new__(GameEnv)
        env.board = self.board
        env.encoder = self.encoder
        env.num_players = self.num_players
        env.MAX_MOVES = self.MAX_MOVES
        env.GAME_TIME_LIMIT_SEC = self.GAME_TIME_LIMIT_SEC
        env.colors = list(self.colors)
        env.pieces = [list(p) for p in self.pieces]
        env.current_player = self.current_player
        env.move_count = self.move_count
        env.player_move_counts = list(self.player_move_counts)
        env.player_time_taken = list(self.player_time_taken)
        env.done = self.done
        env.winner = self.winner
        env.last_hop_length = self.last_hop_length
        env._occupied = set(self._occupied)
        env._legal_cache = {p: None for p in range(env.num_players)}
        env._stuck = list(self._stuck)
        env._prev_dist = list(self._prev_dist)
        env._prev_pins = list(self._prev_pins)
        env._init_potential = list(self._init_potential)
        return env
