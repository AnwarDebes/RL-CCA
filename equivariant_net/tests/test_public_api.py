"""Tests for the wreath equivariant public API surface."""

from __future__ import annotations

import pytest


def test_main_classes_importable():
    from equivariant_net.src import (
        SeatEquivariantBlock,
        SeatInvariantPool,
        WreathSeatNet,
        C6EquivariantLinear,
        WreathFuseLayer,
    )
    assert isinstance(SeatEquivariantBlock, type)
    assert isinstance(C6EquivariantLinear, type)
    assert isinstance(WreathFuseLayer, type)


def test_main_functions_importable():
    from equivariant_net.src import (
        make_seat_mask, make_rotation_permutation,
        rotate_axial, rotate_feature_map, permute_seats,
    )
    assert callable(make_seat_mask)
    assert callable(rotate_axial)
    assert callable(rotate_feature_map)


def test_cc_imports():
    try:
        from equivariant_net.src import (
            CCWreathEncoder, WreathCCNetwork, WreathCCEvaluator,
            cc_seat_features, play_one_wreath_cc_game,
        )
    except ImportError:
        pytest.skip("CC integration needs nexus core/* modules")


def test_all_export_consistent():
    from equivariant_net import src as pkg
    for name in pkg.__all__:
        attr = getattr(pkg, name, "MISSING")
        assert attr != "MISSING", f"__all__ lists {name!r} but it's not importable"


def test_version_metadata_present():
    from equivariant_net import src as pkg
    assert pkg.__version__
    assert pkg.__title__
    assert pkg.__author__
