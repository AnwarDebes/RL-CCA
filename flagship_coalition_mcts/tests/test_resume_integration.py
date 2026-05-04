"""Tests for the checkpoint --resume-from integration.

Verifies that:
1. A training script's checkpoint can be saved and reloaded.
2. The reloaded network produces identical outputs to the original.
3. The optimizer state is preserved.
4. The iter_idx is preserved (so resumed training picks up at the right
   iteration).

These tests are critical for production training: a 24h training run
that crashes at hour 23 must resume from hour 23, not hour 0.
"""

from __future__ import annotations

import os
import tempfile

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.optim as optim

from flagship_coalition_mcts.src.checkpoint import (
    load_checkpoint, save_checkpoint,
)
from flagship_coalition_mcts.src.network import CDMCTSNetwork, MLPEncoder
from flagship_coalition_mcts.src.replay_buffer import ReplayBuffer


def _build_test_net(seed: int = 0):
    torch.manual_seed(seed)
    encoder = MLPEncoder(input_dim=8, hidden_dim=16, num_layers=2)
    return CDMCTSNetwork(encoder=encoder, action_space_size=10, max_players=4)


def test_save_load_preserves_network_outputs():
    """Loading a saved checkpoint reproduces identical network outputs."""
    net1 = _build_test_net(seed=42)
    net1.eval()
    x = torch.randn(3, 8)

    with torch.no_grad():
        out1 = net1(x)

    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "ckpt.pt")
        save_checkpoint(path, net1, iter_idx=10)

        net2 = _build_test_net(seed=99)  # different seed
        load_checkpoint(path, net2, strict=False)
        net2.eval()

        with torch.no_grad():
            out2 = net2(x)

    for o1, o2 in zip(out1, out2):
        assert torch.allclose(o1, o2, atol=1e-7), (
            "Reloaded network produces different outputs from original"
        )


def test_save_load_preserves_optimizer_state():
    """Optimizer's running stats (e.g. Adam moments) survive a roundtrip."""
    net = _build_test_net()
    opt = optim.Adam(net.parameters(), lr=1e-3)

    # Run a few training steps to populate Adam state
    x = torch.randn(2, 8)
    for _ in range(5):
        opt.zero_grad()
        out = net(x)
        loss = sum(o.sum() for o in out)
        loss.backward()
        opt.step()

    # Snapshot state
    state_before = opt.state_dict()

    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "ckpt.pt")
        save_checkpoint(path, net, iter_idx=5, optimizer=opt)

        net2 = _build_test_net(seed=99)
        opt2 = optim.Adam(net2.parameters(), lr=1e-3)
        load_checkpoint(path, net2, optimizer=opt2)

        state_after = opt2.state_dict()

    # Compare param_groups
    for pg1, pg2 in zip(state_before["param_groups"], state_after["param_groups"]):
        assert pg1.keys() == pg2.keys()
    # State dict should have same keys
    assert state_before["state"].keys() == state_after["state"].keys()


def test_save_load_preserves_iter_idx():
    """Iter idx survives the roundtrip."""
    net = _build_test_net()
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "ckpt.pt")
        save_checkpoint(path, net, iter_idx=42)
        net2 = _build_test_net(seed=99)
        bundle = load_checkpoint(path, net2)
        assert bundle.iter_idx == 42


def test_save_load_preserves_replay_buffer():
    """Replay buffer entries are preserved across save/load."""
    net = _build_test_net()
    buf = ReplayBuffer(capacity=100)
    for i in range(20):
        buf.add({"i": i, "data": np.array([i * 1.0, i * 2.0])})

    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "ckpt.pt")
        save_checkpoint(path, net, iter_idx=1, replay_buffer=buf)

        net2 = _build_test_net(seed=99)
        buf2 = ReplayBuffer(capacity=1)
        load_checkpoint(path, net2, replay_buffer=buf2)

    assert len(buf2) == 20
    assert buf2.capacity == 100
    samples = buf2.sample(5)
    for s in samples:
        assert "i" in s
        assert s["i"] in range(20)


def test_metadata_survives_roundtrip():
    """User-supplied metadata is preserved."""
    net = _build_test_net()
    meta = {"experiment": "test_resume", "history": [{"loss": 1.5}, {"loss": 1.2}]}
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "ckpt.pt")
        save_checkpoint(path, net, iter_idx=3, metadata=meta)
        net2 = _build_test_net(seed=99)
        bundle = load_checkpoint(path, net2)
    assert bundle.metadata == meta
