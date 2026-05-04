"""Regression tests for the 3 real bugs caught during testing.

Each bug has a paired test elsewhere in the suite, but these regression
tests act as a single cross-cutting safety net to catch any backslide
in a single failing pytest invocation.

See BUGS_FOUND.md for the full post-mortem.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest
import torch

_NEXUS_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)
if _NEXUS_ROOT not in sys.path:
    sys.path.insert(0, _NEXUS_ROOT)


def test_regression_bug_1_wreath_rotation_equivariance():
    """Bug #1: wreath_fuse seat→spatial bias broke C6 equivariance.

    The fix: bias is broadcast identically across all 6 rotation
    channels. Test that a 60° rotation BEFORE the layer == rotation
    AFTER the layer (on interior cells).
    """
    from equivariant_net.src.c6_spatial import (
        make_rotation_permutation, rotate_axial, rotate_feature_map,
    )
    from equivariant_net.src.wreath_fuse import WreathFuseLayer, permute_seats

    coords = []
    for q in range(-2, 3):
        for r in range(-2, 3):
            if abs(q) + abs(r) + abs(q + r) <= 4:
                coords.append((q, r))
    L = len(coords)
    perm_rot = torch.from_numpy(make_rotation_permutation(coords, num_rotations=6))

    torch.manual_seed(0)
    layer = WreathFuseLayer(in_spatial=2, in_seat=3, out_spatial=4, out_seat=5)
    layer.eval()
    spatial = torch.randn(1, 3, L, 6 * 2)
    seat = torch.randn(1, 3, 3)

    # Path 1: rotate then layer
    rot_idx = 1
    sp_rot = torch.zeros_like(spatial)
    for s in range(spatial.shape[1]):
        sp_rot[:, s] = rotate_feature_map(spatial[:, s], perm_rot, 2, rot_idx)
    sp_a, _ = layer(sp_rot, seat)

    # Path 2: layer then rotate
    sp_b, _ = layer(spatial, seat)
    sp_b_rot = torch.zeros_like(sp_b)
    for s in range(sp_b.shape[1]):
        sp_b_rot[:, s] = rotate_feature_map(sp_b[:, s], perm_rot, 4, rot_idx)

    # Interior-only check
    coord_set = set(coords)
    interior = [
        rotate_axial(c) in coord_set for c in coords
    ]
    if any(interior):
        m = torch.tensor(interior)
        err = (sp_a[:, :, m] - sp_b_rot[:, :, m]).abs().max().item()
        assert err < 1e-5, (
            f"Bug #1 regressed: rotation equivariance violated, max diff = {err:.2e}. "
            f"See BUGS_FOUND.md::Bug-1."
        )


def test_regression_bug_2_cc_adapter_shape():
    """Bug #2: CC adapter returned (max_players, max_players) instead of
    (N, N), crashing the first MCTS rollout on real CC.

    The fix: evaluator returns (N, N) matching active player count.
    """
    try:
        from flagship_coalition_mcts.src.cc_runner import build_cc_evaluator
        from flagship_coalition_mcts.src.games.chinese_checkers import make_cc_env
    except ImportError:
        pytest.skip("CC adapter requires nexus core/* modules")

    torch.manual_seed(0)
    _, ev = build_cc_evaluator(
        num_players_max=6, channels=4, num_blocks=1, hidden_dim=8,
    )
    state = make_cc_env(num_players=2, seed=0)
    out = ev.evaluate(state)
    assert out.placement_marginals.shape == (2, 2), (
        f"Bug #2 regressed: placement_marginals.shape={out.placement_marginals.shape}, "
        f"expected (2, 2). See BUGS_FOUND.md::Bug-2."
    )
    assert out.coalition_alignment.shape == (2,), (
        f"Bug #2 regressed: coalition_alignment.shape={out.coalition_alignment.shape}, "
        f"expected (2,). See BUGS_FOUND.md::Bug-2."
    )


def test_regression_bug_3_cmaz_mixer_gradient():
    """Bug #3: CMAZ mixer received zero gradient - dead weight.

    The fix: cmaz_loss includes a value loss term that exercises the
    mixer's hypernetwork output.
    """
    from decomposed_mcts.src.network import (
        CMAZEncoder, CMAZNetwork, cmaz_loss,
    )

    torch.manual_seed(0)
    enc = CMAZEncoder(input_dim=8, hidden_dim=16, num_layers=2)
    net = CMAZNetwork(encoder=enc, action_space_size=4, num_components=3)

    B = 4
    feats = torch.randn(B, 8)
    target_pol = torch.softmax(torch.randn(B, 4), dim=-1)
    legal_mask = torch.ones(B, 4, dtype=torch.bool)
    target_comp = torch.rand(B, 3)

    total, _ = cmaz_loss(net, feats, target_pol, legal_mask, target_comp)
    total.backward()

    # Mixer's hypernetwork must receive non-None gradient
    for n, p in net.named_parameters():
        if "mixer" in n:
            assert p.grad is not None, (
                f"Bug #3 regressed: mixer parameter {n} has no gradient. "
                f"See BUGS_FOUND.md::Bug-3."
            )
            assert torch.isfinite(p.grad).all(), (
                f"Bug #3 regressed: non-finite gradient on {n}."
            )
