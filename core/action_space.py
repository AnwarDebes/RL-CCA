"""Action space encoding/decoding and legal move masking for Chinese Checkers.

Action encoding: action = piece_id * 121 + dest_index
  piece_id: 0-9 (index into the player's sorted piece list)
  dest_index: 0-120 (cell index on the board)
Total action space: 1210
"""

import torch
import numpy as np
from typing import Dict, List, Tuple

from config import Config


def encode_action(piece_id: int, dest_index: int) -> int:
    """Encode (piece_id, dest_index) into a single action integer."""
    return piece_id * Config.NUM_CELLS + dest_index


def decode_action(action: int) -> Tuple[int, int]:
    """Decode action integer into (piece_id, dest_index)."""
    piece_id = action // Config.NUM_CELLS
    dest_index = action % Config.NUM_CELLS
    return piece_id, dest_index


def build_legal_mask(
    legal_moves: Dict[int, List[int]],
    piece_positions: List[int],
) -> torch.Tensor:
    """Build a binary legal move mask of shape (1210,).

    Args:
        legal_moves: {piece_cell_index: [dest_cell_indices]} - the raw legal
            moves keyed by the cell index where the piece currently sits.
        piece_positions: sorted list of the player's 10 piece cell indices.
            piece_id i corresponds to piece_positions[i].

    Returns:
        Boolean tensor of shape (ACTION_SPACE,) - True for legal actions.
    """
    mask = torch.zeros(Config.ACTION_SPACE, dtype=torch.bool)
    pos_to_id = {pos: pid for pid, pos in enumerate(piece_positions)}

    for piece_pos, dests in legal_moves.items():
        pid = pos_to_id.get(piece_pos)
        if pid is None:
            continue
        for dest in dests:
            action = encode_action(pid, dest)
            mask[action] = True

    return mask


def build_legal_mask_from_server(
    server_legal_moves: Dict[str, List[int]],
    pin_positions: List[int],
) -> torch.Tensor:
    """Build legal mask from server-format legal moves.

    Server returns {pin_id_str: [dest_indices]}.
    pin_positions: list where pin_positions[pin_id] = cell_index of that pin.
        This is state['pins'][my_colour] from the server.

    NEXUS uses piece_id = index into SORTED piece list.
    Server's pin_id is a stable ID that doesn't change when pieces move.
    We must map: pin_id -> pin's cell -> sorted position -> piece_id.
    """
    mask = torch.zeros(Config.ACTION_SPACE, dtype=torch.bool)
    sorted_positions = sorted(pin_positions)

    for pid_str, dests in server_legal_moves.items():
        pin_id = int(pid_str)
        pin_cell = pin_positions[pin_id]
        piece_id = sorted_positions.index(pin_cell)
        for dest in dests:
            action = encode_action(piece_id, dest)
            mask[action] = True
    return mask


def decode_action_to_server(
    action: int,
    pin_positions: List[int],
) -> Tuple[int, int]:
    """Decode NEXUS action to server's (pin_id, dest_index).

    NEXUS action = piece_id * 121 + dest, where piece_id indexes sorted pieces.
    Server expects (pin_id, dest_index) where pin_id is the original stable ID.
    """
    piece_id, dest = decode_action(action)
    sorted_positions = sorted(pin_positions)
    piece_cell = sorted_positions[piece_id]
    pin_id = pin_positions.index(piece_cell)
    return pin_id, dest


def mask_policy_logits(logits: torch.Tensor, legal_mask: torch.Tensor) -> torch.Tensor:
    """Mask illegal actions to -inf BEFORE softmax.

    Args:
        logits: raw policy logits, shape (..., 1210)
        legal_mask: boolean mask, shape (..., 1210), True = legal

    Returns:
        Masked logits with illegal actions set to -inf.
    """
    return logits.masked_fill(~legal_mask, float('-inf'))


def get_legal_actions(legal_mask: torch.Tensor) -> List[int]:
    """Return list of legal action indices from a mask."""
    return legal_mask.nonzero(as_tuple=False).squeeze(-1).tolist()
