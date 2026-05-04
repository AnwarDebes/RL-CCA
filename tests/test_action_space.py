"""Tests for core/action_space.py."""
import sys
sys.path.insert(0, '/home/coder/nexus')

import torch
from core.action_space import (
    encode_action, decode_action, build_legal_mask, mask_policy_logits,
    get_legal_actions,
)
from config import Config


def test_encode_decode_roundtrip():
    """encode(decode(a)) == a for all valid actions."""
    for a in range(Config.ACTION_SPACE):
        pid, dest = decode_action(a)
        assert encode_action(pid, dest) == a


def test_decode_ranges():
    """piece_id in [0,9], dest in [0,120]."""
    for a in range(Config.ACTION_SPACE):
        pid, dest = decode_action(a)
        assert 0 <= pid < Config.NUM_PIECES
        assert 0 <= dest < Config.NUM_CELLS


def test_legal_mask_shape():
    mask = torch.zeros(Config.ACTION_SPACE, dtype=torch.bool)
    assert mask.shape == (1210,)


def test_legal_mask_has_legal_actions():
    """Mask has at least 1 legal action for a non-empty legal_moves dict."""
    # Simulate: piece 0 at cell 50 can go to cells 51, 52
    piece_positions = list(range(10))  # pieces at cells 0-9
    piece_positions[0] = 50
    legal_moves = {50: [51, 52]}
    mask = build_legal_mask(legal_moves, piece_positions)
    assert mask.any(), "Should have at least one legal action"
    actions = get_legal_actions(mask)
    assert len(actions) == 2
    # Verify the actions decode correctly
    for a in actions:
        pid, dest = decode_action(a)
        assert pid == 0
        assert dest in [51, 52]


def test_mask_policy_logits():
    """Illegal actions masked to -inf, legal ones untouched."""
    logits = torch.zeros(Config.ACTION_SPACE)
    mask = torch.zeros(Config.ACTION_SPACE, dtype=torch.bool)
    mask[0] = True
    mask[121] = True  # piece 1, cell 0
    masked = mask_policy_logits(logits, mask)
    assert masked[0] == 0.0
    assert masked[121] == 0.0
    assert masked[1] == float('-inf')
    # softmax should only have mass on legal actions
    probs = torch.softmax(masked, dim=-1)
    assert probs[0] > 0
    assert probs[1] == 0.0
    assert abs(probs.sum().item() - 1.0) < 1e-5


if __name__ == '__main__':
    import pytest
    pytest.main([__file__, '-v'])
