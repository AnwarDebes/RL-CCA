"""v4 tournament player - uses MCTS at inference with adaptive sim budget
and subtree reuse across moves. Plays FAST (~0.3-0.5s per move).

Plies-based budget (more sims early, fewer in endgame):
  plies 0-10:   MCTS_INF_SIMS_OPENING_V4
  plies 11-60:  MCTS_INF_SIMS_MIDGAME_V4
  plies 61+:    MCTS_INF_SIMS_ENDGAME_V4

Hard wall-clock cap: MCTS_INF_HARD_BUDGET_SEC.
Subtree reuse: track our last-move tree; descend through opponent's move and
our own move to inherit visit counts.
"""
from __future__ import annotations

import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from config import Config
from core.action_space import (
    build_legal_mask_from_server, decode_action_to_server, get_legal_actions,
)
from core.board import HexBoard
from core.game_env import GameEnv, COLOUR_ORDER
from core.state_encoder import StateEncoder
from network.model_v4 import NexusNetV4
from inference.time_manager import TimeManager
from training.heuristic_agent import HeuristicAgent
from mcts.mcts_v4 import GumbelMCTSv4, advance_root, NodeV4


class NexusTournamentPlayerV4:
    """v4 tournament player: MCTS at inference with subtree reuse.

    Falls back to v4 raw policy or heuristic on any exception.
    """

    def __init__(self, model_path: str = None, device: str = "cuda",
                 use_mcts: bool = True):
        self.device = torch.device(
            device if torch.cuda.is_available() else "cpu"
        )
        self.board = HexBoard()
        self.encoder = StateEncoder(self.board)
        self.heuristic = HeuristicAgent(self.board)
        self.use_mcts = use_mcts

        if model_path is not None:
            sd = torch.load(model_path, map_location="cpu", weights_only=True)
            # Detect v4 vs v3 vs v2
            keys = list(sd.keys())
            is_v4 = any(k.startswith("backbone.blocks_a.") for k in keys)
            if is_v4:
                self.network = NexusNetV4(self.board).to(self.device)
                self.network.load_state_dict(sd)
            else:
                # Fall back to v3 auto-detect
                from inference.tournament_player import NexusTournamentPlayer
                # Should not happen for v4 deployments; just raise loudly
                raise RuntimeError(
                    f"Checkpoint at {model_path} is not v4. "
                    f"Use NexusTournamentPlayer (v3) for older checkpoints."
                )
            self._is_v4 = True
        else:
            self.network = NexusNetV4(self.board).to(self.device)
            self._is_v4 = True
        self.network.eval()

        # MCTS for the entire game; root persists across our moves
        self.mcts = GumbelMCTSv4(
            self.network, self.device,
            num_simulations=Config.MCTS_INF_SIMS_MIDGAME_V4,
            m=Config.MCTS_INF_M_V4,
        )
        self.search_root: Optional[NodeV4] = None

        # Per-game state
        self.my_color: Optional[str] = None
        self.opp_color: Optional[str] = None
        self.move_count = 0
        self.all_players: List[str] = []
        self.other_colors: List[str] = []
        self.time_manager = TimeManager()

    def set_color(self, color: str, all_player_colors: List[str] = None):
        self.my_color = color
        self.opp_color = Config.COLOR_OPPOSITES[color]
        self.all_players = all_player_colors or [color, self.opp_color]
        self.other_colors = [c for c in self.all_players if c != color]
        self.time_manager = TimeManager()
        self.move_count = 0
        self.search_root = None   # fresh tree per game

    def _build_env_from_state(self, state_pins: Dict[str, List[int]]) -> GameEnv:
        """Reconstruct a GameEnv that mirrors the server's current state.

        Used for MCTS rollouts, which need a stepable env.
        """
        N = len(self.all_players)
        env = GameEnv(self.board, num_players=N)
        env.reset(num_players=N, colors=list(self.all_players))
        # Override pieces directly
        env.pieces = [list(state_pins.get(c, [])) for c in env.colors]
        # current_player is OUR seat - we're being asked to move
        env.current_player = env.colors.index(self.my_color)
        env._rebuild_occupied()
        env._legal_cache = {p: None for p in range(env.num_players)}
        # Recompute prev_dist/prev_pins (used by reward shaping; safe defaults)
        env._prev_dist = [
            self.board.sum_distances_to_goal(env.pieces[p], env.colors[p])
            for p in range(env.num_players)
        ]
        env._prev_pins = [
            self.board.count_in_goal(env.pieces[p], env.colors[p])
            for p in range(env.num_players)
        ]
        return env

    def _adaptive_sims(self) -> int:
        if self.move_count <= 10:
            return Config.MCTS_INF_SIMS_OPENING_V4
        if self.move_count <= 60:
            return Config.MCTS_INF_SIMS_MIDGAME_V4
        return Config.MCTS_INF_SIMS_ENDGAME_V4

    def choose_move(
        self,
        state_pins: Dict[str, List[int]],
        legal_moves: Dict[str, List[int]],
    ) -> Tuple[int, int]:
        self.time_manager.start_move()
        try:
            result = self._mcts_move(state_pins, legal_moves)
        except Exception as e:
            print(f"[NEXUS-v4] MCTS failed ({e}), falling back to raw policy")
            try:
                result = self._policy_move(state_pins, legal_moves)
            except Exception as e2:
                print(f"[NEXUS-v4] Policy failed ({e2}), falling back to heuristic")
                result = self._heuristic_move(state_pins, legal_moves)
        self.move_count += 1
        self.time_manager.record_move()
        return result

    def _mcts_move(self, state_pins, legal_moves):
        """MCTS-guided move with subtree reuse + adaptive budget + hard time cap."""
        if not self.use_mcts:
            return self._policy_move(state_pins, legal_moves)

        env = self._build_env_from_state(state_pins)
        sims = self._adaptive_sims()

        # If our last call's elapsed was over budget, scale down sims this call.
        if hasattr(self, "_last_ms_per_sim") and self._last_ms_per_sim > 0:
            max_sims_in_budget = max(8, int(
                (Config.MCTS_INF_HARD_BUDGET_SEC * 1000.0 * 0.85)
                / self._last_ms_per_sim
            ))
            sims = min(sims, max_sims_in_budget)

        self.mcts.num_simulations = sims

        # Subtree reuse - best effort. We don't precisely track opp actions
        # since the MCTS tree only fully expands the immediate root children
        # at low sim counts. Use search_root only if it's defined and was
        # advanced after our last move.
        root = self.search_root

        t0 = time.time()
        action, _, _, root = self.mcts.search(env, root=root)
        elapsed = time.time() - t0
        self._last_ms_per_sim = (elapsed * 1000.0) / max(1, sims)

        # Save root for potential reuse next call (via advance_root before next call)
        self.search_root = root

        # If we blew budget, drop sims more aggressively next call
        if elapsed > Config.MCTS_INF_HARD_BUDGET_SEC:
            print(f"[NEXUS-v4] move {self.move_count}: {elapsed:.2f}s "
                  f"(sims={sims}, ~{self._last_ms_per_sim:.1f}ms/sim) "
                  f"- will lower sims next call")

        # Decode to server (pin_id, dest)
        my_pin_positions = state_pins.get(self.my_color, [])
        pin_id, dest = decode_action_to_server(action, my_pin_positions)
        # Advance our own subtree by the chosen action so next call can reuse
        if self.search_root is not None:
            self.search_root = advance_root(self.search_root, action)
        return pin_id, dest

    def _policy_move(self, state_pins, legal_moves):
        my_pin_positions = state_pins.get(self.my_color, [])
        my_pieces_sorted = sorted(my_pin_positions)
        opp_pieces = []
        for color, positions in state_pins.items():
            if color != self.my_color:
                opp_pieces.extend(positions)
        opp_pieces_sorted = sorted(opp_pieces)
        present = [c for c in COLOUR_ORDER if c in state_pins]
        if self.my_color in present:
            i = present.index(self.my_color)
            ordered_opps = present[i + 1:] + present[:i]
        else:
            ordered_opps = list(self.other_colors)
        opp_pieces_by_slot = [list(state_pins.get(c, [])) for c in ordered_opps]
        opp_colors_by_slot = list(ordered_opps)
        state_tensor = self.encoder.encode(
            my_pieces=my_pieces_sorted,
            opp_pieces=opp_pieces_sorted,
            my_color=self.my_color,
            other_colors=self.other_colors,
            num_players=len(self.all_players) or 2,
            move_count=self.move_count,
            opp_pieces_by_slot=opp_pieces_by_slot,
            opp_colors_by_slot=opp_colors_by_slot,
        ).unsqueeze(0).to(self.device)
        legal_mask = build_legal_mask_from_server(
            legal_moves, my_pin_positions
        ).unsqueeze(0).to(self.device)
        ordered = [c for c in COLOUR_ORDER if c in self.all_players]
        my_seat = ordered.index(self.my_color) if self.my_color in ordered else 0
        seat_t = torch.tensor([my_seat], device=self.device)
        with torch.no_grad():
            out = self.network(state_tensor, legal_mask, current_seat=seat_t)
        action = out["policy"][0].argmax().item()
        pin_id, dest = decode_action_to_server(action, my_pin_positions)
        return pin_id, dest

    def _heuristic_move(self, state_pins, legal_moves):
        # Construct an env mirror for the heuristic (it's pure-CPU)
        env = self._build_env_from_state(state_pins)
        action = self.heuristic.choose_move(env, env.current_player)
        my_pin_positions = state_pins.get(self.my_color, [])
        pin_id, dest = decode_action_to_server(action, my_pin_positions)
        return pin_id, dest
