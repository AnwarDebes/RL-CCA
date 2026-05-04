"""End-to-end integration test: exercises all 3 subprojects on real CC.

Runs the same logic as `experiments/full_integration_demo.py` but as a
test (so any regression is caught automatically). Bounds simulation
counts tightly so the test finishes in <30s.

This catches integration-level regressions that unit tests miss - e.g.,
shape mismatches between subprojects, import cycles, evaluator API
drift.
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

try:
    from decomposed_mcts.src.cc_adapter import (
        CMAZCCEvaluator, build_cmaz_cc_network,
    )
    from decomposed_mcts.src.cmaz_mcts import run_mcts_cmaz
    from equivariant_net.src.cc_wreath_encoder import CCWreathEncoder
    from flagship_coalition_mcts.src.cc_runner import build_cc_evaluator
    from flagship_coalition_mcts.src.games.chinese_checkers import (
        ChineseCheckersGame, cc_state_to_features_2d, make_cc_env,
    )
    from flagship_coalition_mcts.src.mcts import run_mcts
    HAVE = True
except ImportError:
    HAVE = False

pytestmark = pytest.mark.skipif(
    not HAVE, reason="Integration test requires nexus core/* modules",
)


def test_flagship_cdmcts_runs_on_real_cc():
    """CD-MCTS produces a valid policy on a real CC state."""
    torch.manual_seed(0)
    cd_net, cd_ev = build_cc_evaluator(
        num_players_max=6, channels=8, num_blocks=1, hidden_dim=16,
    )
    state = make_cc_env(num_players=2, seed=0)
    _, pi = run_mcts(
        state=state, network=cd_ev, game=ChineseCheckersGame(),
        num_simulations=2, seed=0,
    )
    assert pi.shape[0] > 0
    assert abs(pi.sum() - 1.0) < 1e-9
    assert (pi >= 0).all()


def test_cmaz_runs_on_real_cc():
    """CMAZ produces a valid policy on a real CC state."""
    torch.manual_seed(1)
    cmaz_net = build_cmaz_cc_network(channels=8, num_blocks=1, hidden_dim=16)
    cmaz_ev = CMAZCCEvaluator(cmaz_net)
    state = make_cc_env(num_players=2, seed=1)
    _, pi = run_mcts_cmaz(
        state=state, network=cmaz_ev, game=ChineseCheckersGame(),
        mixer_apply=cmaz_ev.mixer_apply,
        num_simulations=2,
    )
    assert pi.shape[0] > 0
    assert abs(pi.sum() - 1.0) < 1e-9


def test_cmaz_inference_override_changes_apply_function():
    """Different mixer overrides yield different mixer_apply outputs."""
    torch.manual_seed(2)
    cmaz_net = build_cmaz_cc_network(channels=8, num_blocks=1, hidden_dim=16)
    state = make_cc_env(num_players=2, seed=2)

    ev_a = CMAZCCEvaluator(cmaz_net, override_weights=np.array([1.0, 0.0, 0.0, 0.0]))
    ev_b = CMAZCCEvaluator(cmaz_net, override_weights=np.array([0.0, 0.0, 0.0, 1.0]))
    out_a = ev_a.evaluate_cmaz(state)
    v = np.array([0.8, 0.4, 0.1, 0.3])
    qa = ev_a.mixer_apply(v, out_a.encoder_features)
    qb = ev_b.mixer_apply(v, out_a.encoder_features)
    assert qa != qb, "mixer override didn't change Q"


def test_wreath_encoder_runs_on_real_cc():
    """Wreath encoder forward pass works on a real CC state tensor."""
    torch.manual_seed(3)
    enc = CCWreathEncoder(
        in_channels=32, c_spatial=4, hidden_dim=8, num_blocks=1,
    )
    enc.eval()
    state = make_cc_env(num_players=2, seed=3)
    feats_2d = cc_state_to_features_2d(state)
    x = torch.from_numpy(feats_2d).float().unsqueeze(0)
    h = enc(x)
    assert h.shape == (1, 8)
    assert torch.isfinite(h).all()


def test_three_subprojects_compose_on_same_state():
    """All three networks accept the same CC state and produce outputs."""
    torch.manual_seed(4)
    state = make_cc_env(num_players=2, seed=4)
    # Flagship
    cd_net, cd_ev = build_cc_evaluator(num_players_max=6, channels=4, num_blocks=1, hidden_dim=8)
    out_cd = cd_ev.evaluate(state)
    assert out_cd.placement_marginals.shape == (2, 2)
    # CMAZ
    cmaz_net = build_cmaz_cc_network(channels=4, num_blocks=1, hidden_dim=8)
    cmaz_ev = CMAZCCEvaluator(cmaz_net)
    out_cmaz = cmaz_ev.evaluate_cmaz(state)
    assert out_cmaz.component_values.shape == (4,)
    # Wreath
    enc = CCWreathEncoder(in_channels=32, c_spatial=4, hidden_dim=8, num_blocks=1)
    feats_2d = cc_state_to_features_2d(state)
    x = torch.from_numpy(feats_2d).float().unsqueeze(0)
    enc_out = enc(x)
    assert enc_out.shape == (1, 8)
