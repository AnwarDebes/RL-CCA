"""Wreath-equivariant network - public API."""

__version__ = "0.1.0"
__title__ = "Wreath-equivariant network for star-hex Chinese Checkers"
__author__ = "(see paper)"

from .seat_equivariant import (
    SeatEquivariantBlock, SeatInvariantPool, WreathSeatNet, make_seat_mask,
)
from .c6_spatial import (
    C6EquivariantLinear, make_rotation_permutation, rotate_axial,
    rotate_feature_map,
)
from .wreath_fuse import WreathFuseLayer, permute_seats

# CC integration (defer if nexus core/* unavailable)
try:
    from .cc_wreath_encoder import (
        CCWreathEncoder, _build_cc_rotation_permutation, _get_board_axial_coords,
    )
    from .wreath_network import WreathCCNetwork, cc_seat_features
    from .cc_runner import WreathCCEvaluator, play_one_wreath_cc_game
except ImportError:
    CCWreathEncoder = None
    WreathCCNetwork = None

__all__ = [
    "SeatEquivariantBlock", "SeatInvariantPool", "WreathSeatNet", "make_seat_mask",
    "C6EquivariantLinear", "make_rotation_permutation", "rotate_axial",
    "rotate_feature_map",
    "WreathFuseLayer", "permute_seats",
    "CCWreathEncoder", "WreathCCNetwork", "cc_seat_features",
    "WreathCCEvaluator", "play_one_wreath_cc_game",
]
