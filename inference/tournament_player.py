"""Tournament player: drop-in for CAIR game server interface.

Handles 2-6 player games. In multi-player mode, all non-self players'
pieces are merged into the "opponent" channel.

Fallback hierarchy: policy-only -> heuristic.
MCTS is too slow for the tournament's tight time budget (60s total).
"""

import time
from typing import Dict, List, Optional, Tuple

import torch
import numpy as np

from config import Config
from core.board import HexBoard
from core.action_space import (
    encode_action, decode_action, build_legal_mask_from_server,
    decode_action_to_server, get_legal_actions,
)
from core.state_encoder import StateEncoder
from network.model import NexusNet
from network.model_v3 import NexusNetV3
from inference.time_manager import TimeManager
from training.heuristic_agent import HeuristicAgent
from core.game_env import GameEnv


class NexusTournamentPlayer:
    """Tournament-ready player with fallback hierarchy.

    Works with 2-6 players. Uses raw policy (no MCTS) for speed.
    Fallback: neural network policy -> heuristic agent.
    """

    def __init__(self, model_path: str = None, device: str = 'cuda'):
        self.device = torch.device(
            device if torch.cuda.is_available() else 'cpu'
        )
        self.board = HexBoard()
        self.encoder = StateEncoder(self.board)
        self.heuristic = HeuristicAgent(self.board)

        # Load network - auto-detect v2 vs v3 from checkpoint key prefixes.
        if model_path is not None:
            sd = torch.load(model_path, map_location="cpu", weights_only=True)
            is_v3 = any(k.startswith("backbone.res_blocks_a.") for k in sd.keys())
            if is_v3:
                self.network = NexusNetV3(self.board).to(self.device)
                self.network.load_state_dict(sd)
            else:
                self.network = NexusNet.load(model_path, device)
            self._is_v3 = is_v3
        else:
            self.network = NexusNet(self.board).to(self.device)
            self._is_v3 = False
        self.network.eval()

        self.time_manager = TimeManager()

        # Per-game state
        self.my_color: Optional[str] = None
        self.opp_color: Optional[str] = None  # complement color
        self.move_count = 0
        self.all_players: List[str] = []  # all colors in the game

    def set_color(self, color: str, all_player_colors: List[str] = None):
        """Set our color at the start of a game.

        Args:
            color: our assigned color.
            all_player_colors: list of all player colors in the game.
        """
        self.my_color = color
        # Backward-compat: opp_color was the 2-player complement; in N-player
        # the encoder now takes a full `other_colors` list, so opp_color is
        # only kept as a sentinel for old code paths.
        self.opp_color = Config.COLOR_OPPOSITES[color]
        self.all_players = all_player_colors or [color, self.opp_color]
        # Set of OTHER players' colors (handles N=2..6 uniformly)
        self.other_colors = [c for c in self.all_players if c != color]
        self.time_manager = TimeManager()
        self.move_count = 0

    def choose_move(
        self,
        state_pins: Dict[str, List[int]],
        legal_moves: Dict[str, List[int]],
    ) -> Tuple[int, int]:
        """Choose a move given server state.

        Args:
            state_pins: {"color": [cell_indices]} for all colors.
                pin_positions[pin_id] = cell_index (ordered by pin_id).
            legal_moves: {"pin_id_str": [dest_indices]} for our pieces.

        Returns:
            (pin_id, dest_index) as ints to send to server.
        """
        self.time_manager.start_move()

        try:
            result = self._policy_move(state_pins, legal_moves)
        except Exception as e:
            print(f"[NEXUS] Policy failed ({e}), falling back to heuristic")
            result = self._heuristic_move(state_pins, legal_moves)

        self.move_count += 1
        self.time_manager.record_move()
        return result

    def _policy_move(
        self, state_pins: Dict[str, List[int]], legal_moves: Dict[str, List[int]]
    ) -> Tuple[int, int]:
        """Neural network policy move (no MCTS)."""
        my_pin_positions = state_pins.get(self.my_color, [])
        my_pieces_sorted = sorted(my_pin_positions)

        # Merge ALL other players' pieces into "opponent" (channel 1 union)
        opp_pieces = []
        for color, positions in state_pins.items():
            if color != self.my_color:
                opp_pieces.extend(positions)
        opp_pieces_sorted = sorted(opp_pieces)

        # v3-rebuild: per-opponent slots ordered by relative seat offset.
        # Use the COLOUR_ORDER turn rotation to produce stable slot indices.
        from core.game_env import COLOUR_ORDER
        present = [c for c in COLOUR_ORDER if c in state_pins]
        if self.my_color in present:
            i = present.index(self.my_color)
            ordered_opps = present[i+1:] + present[:i]
        else:
            ordered_opps = [c for c in self.other_colors]
        opp_pieces_by_slot = [list(state_pins.get(c, [])) for c in ordered_opps]
        opp_colors_by_slot = list(ordered_opps)

        # Encode state - pass full other_colors list so channels 3/5 union
        # the goal/start zones of every actual opponent in the game.
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

        # Build legal mask with proper pin_id -> piece_id mapping
        legal_mask = build_legal_mask_from_server(
            legal_moves, my_pin_positions
        ).unsqueeze(0).to(self.device)

        with torch.no_grad():
            if self._is_v3:
                # v3's value head is per-seat. We must pass our actual seat
                # in COLOUR_ORDER so the value slot belongs to us. The policy
                # may also depend on seat through shared backbone activations
                # - passing the right seat is correct regardless of head split.
                from core.game_env import COLOUR_ORDER
                ordered = [c for c in COLOUR_ORDER if c in self.all_players]
                my_seat = ordered.index(self.my_color) if self.my_color in ordered else 0
                seat_t = torch.tensor([my_seat], device=self.device)
                out = self.network(state_tensor, legal_mask, current_seat=seat_t)
            else:
                out = self.network(state_tensor, legal_mask)

        action = out['policy'][0].argmax().item()

        # Decode NEXUS action -> server's (pin_id, dest)
        pin_id, dest = decode_action_to_server(action, my_pin_positions)
        return pin_id, dest

    def _heuristic_move(
        self, state_pins: Dict[str, List[int]], legal_moves: Dict[str, List[int]]
    ) -> Tuple[int, int]:
        """Heuristic fallback - picks move that reduces distance to goal most."""
        my_pin_positions = state_pins.get(self.my_color, [])
        best_score = float('-inf')
        best_move = None

        for pid_str, dests in legal_moves.items():
            pin_id = int(pid_str)
            if pin_id >= len(my_pin_positions):
                continue
            piece_pos = my_pin_positions[pin_id]

            for dest in dests:
                dist_before = self.board.min_distance_to_goal(piece_pos, self.my_color)
                dist_after = self.board.min_distance_to_goal(dest, self.my_color)
                score = (dist_before - dist_after) * 10.0
                hop_len = self.board.axial_distance(piece_pos, dest)
                if hop_len > 1:
                    score += hop_len * 5.0
                # Bonus for landing in goal
                if self.board.is_in_goal(dest, self.my_color):
                    score += 20.0

                if score > best_score:
                    best_score = score
                    best_move = (pin_id, dest)

        if best_move:
            return best_move

        # Absolute fallback: first legal move
        for pid_str, dests in legal_moves.items():
            if dests:
                return int(pid_str), dests[0]
        raise RuntimeError("No legal moves")
