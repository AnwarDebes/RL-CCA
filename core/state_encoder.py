"""State encoder: N-player game state -> 32-channel 17x17 tensor.

v3-rebuild: adds per-opponent channels (no more "all opponents unioned"
information loss). The agent now sees each opponent individually, keyed by
relative seat offset from the current player, so positions, rankings, and
threats are distinguishable across N=2..6.

Channel layout (32 channels total):

  0   my pieces
  1   union of all OTHER players' pieces (kept for back-compat / global context)
  2   my goal zone
  3   union of all OTHER players' goal zones
  4   my start zone
  5   union of all OTHER players' start zones
  6   empty cells
  7   board mask
  8-17 per-piece distance-to-goal (10 channels)
  18  pieces that can land in goal in one move
  19  N / 6 (broadcast)
  20  player elapsed virtual-time fraction (broadcast)
  21  in-goal fraction (broadcast)
  22  opp slot 0 - pieces (next-to-move opponent after me)
  23  opp slot 0 - goal zone
  24  opp slot 1 - pieces
  25  opp slot 1 - goal zone
  26  opp slot 2 - pieces
  27  opp slot 2 - goal zone
  28  opp slot 3 - pieces
  29  opp slot 3 - goal zone
  30  opp slot 4 - pieces
  31  opp slot 4 - goal zone

Slot ordering: relative seat offset from current player. Slot 0 is the
opponent who plays NEXT (seat = my_seat+1 mod N), slot 1 is two-after-me,
and so on. Unused slots (when N<6) are zeroed.
"""

from typing import Dict, List, Optional

import torch

from config import Config
from core.board import HexBoard


class StateEncoder:
    """Encodes game state into a 32x17x17 tensor - works for N=2..6."""

    def __init__(self, board: HexBoard):
        self.board = board
        n = board.num_cells
        G = Config.GRID_SIZE
        self._gx = board.cell_gx
        self._gy = board.cell_gy

        self._board_mask = torch.zeros(G, G)
        for i in range(n):
            self._board_mask[self._gx[i], self._gy[i]] = 1.0

        self._goal_masks: Dict[str, torch.Tensor] = {}
        self._start_masks: Dict[str, torch.Tensor] = {}
        for color in Config.COLOR_OPPOSITES:
            gm = torch.zeros(G, G)
            for idx in board.goal_zones[color]:
                gm[self._gx[idx], self._gy[idx]] = 1.0
            self._goal_masks[color] = gm
            sm = torch.zeros(G, G)
            for idx in board.start_zones[color]:
                sm[self._gx[idx], self._gy[idx]] = 1.0
            self._start_masks[color] = sm

        self._max_dist = 16.0

    def encode(
        self,
        my_pieces: List[int],
        opp_pieces: List[int],
        my_color: str,
        other_colors: List[str],
        num_players: int = 2,
        move_count: int = 0,
        time_elapsed: float = 0.0,
        legal_moves: Optional[Dict[int, List[int]]] = None,
        opp_pieces_by_slot: Optional[List[List[int]]] = None,
        opp_colors_by_slot: Optional[List[str]] = None,
    ) -> torch.Tensor:
        """Encode the full game state. Returns (NUM_CHANNELS, 17, 17) float tensor.

        New v3-rebuild args (back-compat: when None, opp slots are zeros):
            opp_pieces_by_slot: list of length up to 5; entry k = piece cell
                indices of the opponent at relative seat offset k+1.
            opp_colors_by_slot: list of length up to 5; entry k = color of
                that opponent (used to fetch their goal zone).
        """
        G = Config.GRID_SIZE
        tensor = torch.zeros(Config.NUM_CHANNELS, G, G)
        gx, gy = self._gx, self._gy

        # Channel 0: my pieces
        for idx in my_pieces:
            tensor[0, gx[idx], gy[idx]] = 1.0

        # Channel 1: union of all OTHER players' pieces
        for idx in opp_pieces:
            tensor[1, gx[idx], gy[idx]] = 1.0

        # Channel 2: my goal zone
        tensor[2] = self._goal_masks[my_color]

        # Channel 3: union of all OTHER players' goal zones
        if other_colors:
            ch3 = torch.zeros_like(self._board_mask)
            for c in other_colors:
                ch3 = torch.maximum(ch3, self._goal_masks[c])
            tensor[3] = ch3

        # Channel 4: my start zone
        tensor[4] = self._start_masks[my_color]

        # Channel 5: union of all OTHER players' start zones
        if other_colors:
            ch5 = torch.zeros_like(self._board_mask)
            for c in other_colors:
                ch5 = torch.maximum(ch5, self._start_masks[c])
            tensor[5] = ch5

        # Channel 6: empty cells
        tensor[6] = self._board_mask - tensor[0] - tensor[1]

        # Channel 7: board mask
        tensor[7] = self._board_mask

        # Channels 8-17: per-piece distance-to-goal (1 - normalized distance)
        dist_table = self.board._min_dist_to_goal[my_color]
        sorted_pieces = sorted(my_pieces)
        for i, piece_idx in enumerate(sorted_pieces[:10]):
            norm_dist = dist_table[piece_idx] / self._max_dist
            tensor[8 + i, gx[piece_idx], gy[piece_idx]] = 1.0 - norm_dist

        # Channel 18: pieces that can land directly in goal in one move
        if legal_moves is not None:
            goal_set = self.board._goal_set[my_color]
            for piece_pos, dests in legal_moves.items():
                for dest in dests:
                    if dest in goal_set:
                        tensor[18, gx[piece_pos], gy[piece_pos]] = 1.0
                        break

        # Channel 19: N / 6 broadcast
        tensor[19] = self._board_mask * (float(num_players) / 6.0)

        # Channel 20: own move-count fraction
        norm_moves = min(move_count / 100.0, 1.0)
        tensor[20] = self._board_mask * norm_moves

        # Channel 21: in-goal fraction
        in_goal = self.board.count_in_goal(my_pieces, my_color)
        tensor[21] = self._board_mask * (in_goal / Config.NUM_PIECES)

        # Channels 22-31: per-opponent (slot k = relative seat offset k+1)
        if opp_pieces_by_slot is not None and opp_colors_by_slot is not None:
            num_slots = min(Config.NUM_OPP_SLOTS, len(opp_pieces_by_slot))
            for k in range(num_slots):
                base = 22 + 2 * k
                pieces = opp_pieces_by_slot[k] if k < len(opp_pieces_by_slot) else []
                color = opp_colors_by_slot[k] if k < len(opp_colors_by_slot) else None
                for idx in pieces:
                    tensor[base, gx[idx], gy[idx]] = 1.0
                if color is not None and color in self._goal_masks:
                    tensor[base + 1] = self._goal_masks[color]

        return tensor
