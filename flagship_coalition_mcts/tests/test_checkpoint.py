"""Tests for the checkpoint utility."""

from __future__ import annotations

import os
import tempfile

import pytest
import torch
import torch.nn as nn
import torch.optim as optim

from flagship_coalition_mcts.src.checkpoint import (
    CHECKPOINT_VERSION,
    CheckpointBundle,
    latest_checkpoint,
    list_checkpoints,
    load_checkpoint,
    save_checkpoint,
)
from flagship_coalition_mcts.src.replay_buffer import ReplayBuffer


def _make_dummy_net():
    return nn.Sequential(nn.Linear(4, 8), nn.GELU(), nn.Linear(8, 2))


def test_save_load_roundtrip():
    net = _make_dummy_net()
    opt = optim.Adam(net.parameters(), lr=0.001)
    buf = ReplayBuffer(capacity=10)
    buf.add({"i": 1})
    buf.add({"i": 2})
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "ckpt.pt")
        save_checkpoint(path, net, iter_idx=42, optimizer=opt, replay_buffer=buf,
                        metadata={"experiment": "smoke"})
        # New net + opt + buffer
        net2 = _make_dummy_net()
        opt2 = optim.Adam(net2.parameters(), lr=0.001)
        buf2 = ReplayBuffer(capacity=99)
        bundle = load_checkpoint(path, net2, optimizer=opt2, replay_buffer=buf2)
        # Iter idx
        assert bundle.iter_idx == 42
        # Metadata
        assert bundle.metadata == {"experiment": "smoke"}
        # Buffer roundtrip
        assert len(buf2) == 2
        # Network parameters match
        for p1, p2 in zip(net.parameters(), net2.parameters()):
            assert torch.allclose(p1, p2)


def test_save_atomic_replaces_old():
    net = _make_dummy_net()
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "ckpt.pt")
        save_checkpoint(path, net, iter_idx=1)
        first_mtime = os.path.getmtime(path)
        # Modify net and save again
        with torch.no_grad():
            net[0].weight.zero_()
        save_checkpoint(path, net, iter_idx=2)
        # File still exists; .tmp should be cleaned up
        assert os.path.exists(path)
        assert not os.path.exists(path + ".tmp")
        bundle = load_checkpoint(path, _make_dummy_net())
        assert bundle.iter_idx == 2


def test_list_and_latest_checkpoints():
    net = _make_dummy_net()
    with tempfile.TemporaryDirectory() as d:
        for i in [1, 5, 10, 3]:
            save_checkpoint(os.path.join(d, f"iter_{i:04d}.pt"), net, iter_idx=i)
        cps = list_checkpoints(d)
        assert [it for it, _ in cps] == [1, 3, 5, 10]
        latest = latest_checkpoint(d)
        assert latest.endswith("iter_0010.pt")


def test_load_strict_rejects_wrong_version():
    net = _make_dummy_net()
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "ckpt.pt")
        save_checkpoint(path, net, iter_idx=1)
        # Manually rewrite version
        bundle = torch.load(path, weights_only=False)
        bundle["version"] = 999
        torch.save(bundle, path)
        with pytest.raises(ValueError):
            load_checkpoint(path, _make_dummy_net(), strict=True)


def test_load_non_strict_accepts_wrong_version():
    net = _make_dummy_net()
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "ckpt.pt")
        save_checkpoint(path, net, iter_idx=1)
        bundle = torch.load(path, weights_only=False)
        bundle["version"] = 999
        torch.save(bundle, path)
        # Should not raise
        load_checkpoint(path, _make_dummy_net(), strict=False)
