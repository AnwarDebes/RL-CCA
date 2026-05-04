"""Day-3 verification gates.

Verifies the rewritten model + self-play modules:

D3.1 - NexusNet forward pass works on a fresh untrained model
D3.2 - Self-play game completes for each N ∈ {2,3,4,5,6}
D3.3 - value_target on every trajectory entry is in [-1, 1]
D3.4 - Heuristic-mix mode produces a valid trajectory (one seat plays heuristic)
"""

from __future__ import annotations

import os
import random
import sys

NEXUS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, NEXUS_DIR)

import pytest
import torch

from config import Config
from core.board import HexBoard
from core.game_env import GameEnv
from network.model import NexusNet
from training.self_play import generate_self_play_game, generate_games_batched


def _device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── D3.1 - Forward pass ──────────────────────────────────────────


def test_d3_1_forward_pass():
    """Fresh NexusNet runs forward on a random batched state."""
    device = _device()
    net = NexusNet().to(device)
    net.eval()
    state = torch.zeros(2, 22, 17, 17, device=device)
    mask = torch.ones(2, 1210, dtype=torch.bool, device=device)
    with torch.no_grad():
        out = net(state, mask)
    assert "policy" in out and "value" in out and "logits" in out
    assert out["policy"].shape == (2, 1210)
    assert out["value"].shape == (2,)
    assert out["logits"].shape == (2, 1210)
    # Value in [-1, 1] (tanh)
    assert torch.all(out["value"] >= -1.0) and torch.all(out["value"] <= 1.0)


# ── D3.2 - Self-play per N ───────────────────────────────────────


@pytest.mark.parametrize("N", [2, 3, 4, 5, 6])
def test_d3_2_self_play_completes_per_N(N, monkeypatch):
    """generate_games_batched with N=fixed completes one game per N."""
    device = _device()
    net = NexusNet().to(device)
    net.eval()

    # Force the curriculum to return this N for the test.
    monkeypatch.setattr(Config, "NUM_PLAYERS_CURRICULUM", {0: {N: 1.0}})
    # Disable heuristic mix so the test hits the pure-policy path
    monkeypatch.setattr(Config, "VS_HEURISTIC_FRACTION", 0.0)

    board = HexBoard()
    rng = random.Random(20260429 + N)
    trajectories, summaries = generate_games_batched(
        net, device, board,
        num_games=1, temperature=1.0, start_game_id=0,
        iteration=0, rng=rng,
    )
    assert len(trajectories) == 1
    assert len(summaries) == 1
    summ = summaries[0]
    assert summ["N"] == N
    assert summ["total_moves"] >= 1


# ── D3.3 - value_target range ────────────────────────────────────


def test_d3_3_value_target_range(monkeypatch):
    """Every trajectory entry has value_target in [-1, 1]."""
    device = _device()
    net = NexusNet().to(device)
    net.eval()
    monkeypatch.setattr(Config, "NUM_PLAYERS_CURRICULUM", {0: {3: 1.0}})
    monkeypatch.setattr(Config, "VS_HEURISTIC_FRACTION", 0.0)

    board = HexBoard()
    rng = random.Random(20260430)
    trajs, _ = generate_games_batched(
        net, device, board, num_games=1, iteration=0, rng=rng,
    )
    for traj in trajs:
        for entry in traj:
            v = entry["value_target"]
            assert isinstance(v, float), f"value_target should be float, got {type(v)}"
            assert -1.0 <= v <= 1.0, f"value_target {v} out of range"


# ── D3.4 - Heuristic mix ─────────────────────────────────────────


def test_d3_4_heuristic_mix(monkeypatch):
    """When VS_HEURISTIC_FRACTION=1.0, every game has a heuristic seat.
    DAgger design: heuristic seat's moves DO generate trajectory entries
    with one-hot policy targets at the heuristic's chosen action - the
    network learns to imitate the heuristic on the in-distribution states
    Phase 2 generates."""
    device = _device()
    net = NexusNet().to(device)
    net.eval()
    monkeypatch.setattr(Config, "NUM_PLAYERS_CURRICULUM", {0: {3: 1.0}})
    monkeypatch.setattr(Config, "VS_HEURISTIC_FRACTION", 1.0)

    board = HexBoard()
    rng = random.Random(20260501)
    trajs, summaries = generate_games_batched(
        net, device, board, num_games=2, iteration=0, rng=rng,
    )
    for traj, summ in zip(trajs, summaries):
        # heuristic_seat must be set
        assert summ["heuristic_seat"] is not None
        h_seat = summ["heuristic_seat"]
        # Heuristic-seat entries should have is_heuristic=True and one-hot
        # policy_target at the action taken
        h_entries = [e for e in traj if e["player"] == h_seat]
        assert len(h_entries) > 0, "Heuristic seat should have trajectory entries"
        for entry in h_entries:
            assert entry.get("is_heuristic") is True
            # One-hot at action
            pt = entry["policy_target"]
            assert pt[entry["action"]] > 0.99
            assert pt.sum() < 1.01
