"""Tests for core/board.py."""
import sys
sys.path.insert(0, '/home/coder/nexus')

from core.board import HexBoard
from config import Config


def test_board_has_121_cells():
    board = HexBoard()
    assert board.num_cells == 121, f"Expected 121 cells, got {board.num_cells}"


def test_central_cell_has_6_neighbors():
    board = HexBoard()
    # Center cell at (0, 0)
    center = board.index_of[(0, 0)]
    nbrs = board.get_neighbors(center)
    assert len(nbrs) == 6, f"Center cell should have 6 neighbors, got {len(nbrs)}"


def test_axial_to_grid_and_back():
    board = HexBoard()
    for idx in range(board.num_cells):
        gx, gy = board.axial_to_grid(idx)
        assert 0 <= gx < 17 and 0 <= gy < 17, f"Grid pos out of range: ({gx}, {gy})"
        back = board.grid_to_cell_idx(gx, gy)
        assert back == idx, f"Round-trip failed for cell {idx}"


def test_goal_zones_have_10_cells():
    board = HexBoard()
    for color in Config.COLOR_OPPOSITES:
        zone = board.goal_zones[color]
        assert len(zone) == 10, f"{color} goal zone has {len(zone)} cells, expected 10"


def test_start_zones_have_10_cells():
    board = HexBoard()
    for color in Config.COLOR_OPPOSITES:
        zone = board.start_zones[color]
        assert len(zone) == 10, f"{color} start zone has {len(zone)} cells, expected 10"


def test_color_opposites_are_correct():
    expected = {
        'red': 'blue', 'blue': 'red',
        'lawn green': 'gray0', 'gray0': 'lawn green',
        'yellow': 'purple', 'purple': 'yellow',
    }
    assert Config.COLOR_OPPOSITES == expected


def test_goal_is_opponent_start():
    board = HexBoard()
    for color, opposite in Config.COLOR_OPPOSITES.items():
        assert board.goal_zones[color] == board.start_zones[opposite], \
            f"{color}'s goal zone should be {opposite}'s start zone"


def test_no_overlapping_zones():
    board = HexBoard()
    all_zone_cells = []
    for color in Config.COLOR_OPPOSITES:
        all_zone_cells.extend(board.start_zones[color])
    # 6 zones * 10 cells = 60 cells, all should be unique
    assert len(all_zone_cells) == 60
    assert len(set(all_zone_cells)) == 60


def test_board_mask_121_valid():
    """Board mask channel should have exactly 121 valid cells."""
    board = HexBoard()
    grid = [[0] * 17 for _ in range(17)]
    for idx in range(board.num_cells):
        gx, gy = board.grid_pos(idx)
        grid[gx][gy] = 1
    total = sum(sum(row) for row in grid)
    assert total == 121, f"Expected 121 valid grid cells, got {total}"


def test_axial_distance():
    board = HexBoard()
    center = board.index_of[(0, 0)]
    # Distance from center to (1, 0) should be 1
    adj = board.index_of[(1, 0)]
    assert board.axial_distance(center, adj) == 1
    # Distance from center to (4, 0) should be 4
    far = board.index_of[(4, 0)]
    assert board.axial_distance(center, far) == 4


if __name__ == '__main__':
    import pytest
    pytest.main([__file__, '-v'])
